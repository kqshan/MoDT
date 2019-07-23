/*==============================================================================
 * Compute the squared Mahalanobis distance, delta = sum((L\X).^2,1)', on GPU
 *   delta = calcMahalDistGpu(L, X)
 * 
 * Returns:
 *   delta   [N x 1] squared Mahalanobis distance (gpuArray)
 * Required arguments:
 *   L       [D x D] lower triangular Cholesky-factorized covariance matrix
 *   X       [D x N] data matrix (gpuArray)
 * 
 * L and X may be single- or double-precision, but they must be the same type.
 * 
 * For small D, this uses a custom CUDA kernel that multiplies the matrices and
 * computes the squared norm, without requiring intermediate storage.
 * For large D, we compute the explicit inverse on the CPU, perform GEMM on the
 * GPU, and then square and sum across columns using a custom kernel.
 *
 * Kevin Shan
 * 2017-04-27  Switch from TRSM to explicit inverse 
 * 2017-04-26  Add single-precision support
 * 2016-06-16  Initial version
 *============================================================================*/

#ifdef MATLAB_MEX_FILE
    #include "mex.h"
    #include "gpu/mxGPUArray.h"
#else
    #include <unistd.h>
#endif

#include "lapack.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <thrust/device_vector.h>

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <utility>
#include <limits>
#include <memory>
#include <vector>

/* Shuffle operations for single- and double-precision. Unfortunately,
 * I can't just call it __shfl() because there's an undocumented version
 * of __shfl() for double-precision, and yet I'm having trouble linking
 * to it.
 */
__device__ double shfl(double var, int srcLane, int width=32) {
    int hi = __shfl( __double2hiint(var), srcLane, width );
    int lo = __shfl( __double2loint(var), srcLane, width );
    return __hiloint2double( hi, lo );
}

__device__ double shfl_down(double var, unsigned int delta, int width=32) {
    int hi = __shfl_down( __double2hiint(var), delta, width );
    int lo = __shfl_down( __double2loint(var), delta, width );
    return __hiloint2double( hi, lo );
}

__device__ float shfl(float var, int srcLane, int width=32) {
   return __shfl( var, srcLane, width );
}

__device__ float shfl_down(float var, unsigned int delta, int width=32) {
    return __shfl_down(var, delta, width);
}


/* Simple wrapper classes so we can do RAII-style cleanup 
 */
#ifdef MATLAB_MEX_FILE
struct mxGPUArrayCleanupWrapper {
    mxGPUArray const *mxgpu_ptr;
    mxGPUArrayCleanupWrapper(mxGPUArray const *p) {mxgpu_ptr = p;}
    ~mxGPUArrayCleanupWrapper(void) {mxGPUDestroyGPUArray(mxgpu_ptr);}
};
struct mxArrayCleanupWrapper {
    mxArray *mx_ptr;
    mxArrayCleanupWrapper(mxArray *p) {mx_ptr = p;}
    ~mxArrayCleanupWrapper(void) {mxDestroyArray(mx_ptr);}
};
#endif
struct cublasHandleCleanupWrapper {
    cublasHandle_t handle;
    cublasHandleCleanupWrapper(cublasHandle_t h) {handle = h;}
    ~cublasHandleCleanupWrapper(void) {cublasDestroy(handle);}
};
struct CudaDeleter {
    void operator() (void *d_ptr) { cudaFree(d_ptr); }
};


/* Helpers for triSolveNorm
 * 
 * The triSolveNorm kernel is kinda complicated: there are many different things
 * that need to be done on each iteration.
 *
 * For a D=16 triangular solve, there are 10 [4 x 4] sub-blocks of L, and on each
 * iteration we have 4 loads from global memory (LD), 4 triangular solves (TS), 
 * and 6 matrix multiplications (MM).
 * 
 * LD_0 TS_00  |              |             |             |
 * LD_1        | MM_10 TS_11  |             |             |
 * LD_2        | MM_20        | MM_21 TS_22 |             |
 * LD_3        | MM_30        | MM_31       | MM_32 TS_33 | 
 *
 * Each operation acts on a [4 x 8] workspace, and the vertical lines indicate 
 * syncthreads() barriers. After performing the triangular solve, we need to 
 * square and sum across columns.
 */

template <typename numeric_t>
__device__ void tsn_Load(int x, int y, int tIdx, numeric_t &val,
        numeric_t *W, numeric_t const *X, int D, int N, int xOffset) {
    // Load a [4 x 8] block of X into W, padding with zeroes
    int xRead = x + xOffset;
    val = ((xRead<D)&&(y<N)) ? X[xRead+D*y] : 0;
    W[tIdx] = val;
}

template <typename numeric_t>
__device__ void tsn_TriSolve(int x, int y, int tIdx, numeric_t &val,
        numeric_t volatile *W, numeric_t const *L) {
    // Triangular solve with unit-diagonal [4 x 4] L and [4 x 8] W
    // Assume val = W[tIdx];
    numeric_t coeff = L[tIdx%16];
    for (int ii=0; ii<4-1; ii++) {
        // if (x>ii) W[x,y] -= L[x,ii] * W[ii,y];
        // But we need to keep the shfl() outside the conditional
        numeric_t adjustment = shfl(coeff,x+ii*4,16) * shfl(val,ii,4);
        if (x>ii) val -= adjustment;
    }
    // Apply scaling (produced when we unit-diagonalized L)
    val *= shfl(coeff,5*x,16); // W[x,y] *= L[x,x];
    W[tIdx] = val;
}

template <typename numeric_t>
__device__ void tsn_MatMult(int x, int y, int tIdx, numeric_t &val,
        numeric_t *W, numeric_t const *L, numeric_t const *Y) {
    // Matrix multiplication: W -= L * Y, where L is [4x4] and W,Y are [4x8]
    val = W[tIdx];
    numeric_t y_val = Y[tIdx];
    numeric_t coeff = L[tIdx%16];
    for (int ii=0; ii<4; ii++) {
        // W[x,y] -= L[x,ii] * Y[ii,y];
        val -= shfl(coeff,x+ii*4,16) * shfl(y_val,ii,4);
    }
    W[tIdx] = val;
}


/* Convert L into a blocked representation suitable for the triSolveNorm kernel
 * 
 * This copies L into 4 x 4 blocks in column-major order
 * i.e. [00, 10, 20, 30, 11, 21, 31, 22, 32, 33]
 * But it also scales the diagonal blocks so that they are unit-diagonal, and
 * the scaling factor is stored on the diagonal. 
 */
template <typename numeric_t>
std::vector<numeric_t> copyLToBlocks(int D, numeric_t const *L)
{
    // Allocate and initialize to zeros
    int M = (D+4-1)/4; // ceil(D/4)
    std::vector<numeric_t> L_blocks(4*4*M*(M+1)/2, 0.0);
    // Copy L into blocks
    int blockIdx = 0;
    for (int block_j=0; block_j<M; block_j++) {
        // For blocks on the diagonal, scale so that it is unit-diagonal
        for (int jj=0; jj<4; jj++) {
            // Set the diagonal equal to the scaling
            int src_j = jj + 4*block_j;
            if (src_j>=D) continue;
            double scaling = 1 / L[src_j+D*src_j];
            L_blocks[jj + 4*jj + 4*4*blockIdx] = scaling;
            // Scale the rest of that column
            for (int ii=jj+1; ii<4; ii++) {
                int src_i = ii + 4*block_j;
                int tgt_idx = ii + 4*jj + 4*4*blockIdx;
                if (src_i<D)
                    L_blocks[tgt_idx] = scaling * L[src_i + D*src_j];
            }
        }
        blockIdx++;
        // Sub-diagonal blocks are a simple copy
        for (int block_i=block_j+1; block_i<M; block_i++) {
            for (int jj=0; jj<4; jj++) {
                int src_j = jj + 4*block_j;
                if (src_j>=D) continue;
                for (int ii=0; ii<4; ii++) {
                    int src_i = ii + 4*block_i;
                    int tgt_idx = ii + 4*jj + 4*4*blockIdx;
                    if (src_i<D)
                        L_blocks[tgt_idx] = L[src_i + D*src_j];
                }
            }
            blockIdx++;
        }
    }
    return L_blocks;
}


/* Shared memory layout
 */
template <typename numeric_t>
__device__ void tsn_SetupSharedMem(int M, int nBanks, 
        int* S, numeric_t const *L_global, 
        numeric_t* &workspaces, numeric_t* &L_shared)
{
    // Lay out memory
    int offset = 0;
    workspaces = reinterpret_cast<numeric_t*>(&S[offset]);
    offset += 4*8 * M * nBanks * sizeof(numeric_t)/sizeof(*S);
    L_shared = reinterpret_cast<numeric_t*>(&S[offset]);
    int L_size = 4*4 * M*(M+1)/2;
    // Read L into shared memory
    int readOffset = threadIdx.x + 32*threadIdx.y;
    int readStride = 32 * blockDim.y;
    for (int ii=readOffset; ii<L_size; ii+=readStride)
        L_shared[ii] = L_global[ii];
    __syncthreads();
}


/* triSolveNorm - CUDA kernel to perform triangular matrix solve and the compute
 *                the squared 2-norm over the columns of the result.
 *
 * This expects:
 *   - 1-D block grid of any size
 *   - Thread block of size [32 x nWarps]
 *   - Shared memory for the following (M = ceil(D/4)):
 *     - [4 x 4 x M*(M+1)/2] sub-blocks of L
 *     - [4 x 8 x M x nBanks] workspaces
 * 
 * This processes 8*nBanks columns of X at a time. There are two versions of
 * this kernel: synchronized and unsynchronized.
 * 
 * The unsynchronized version has no __syncthreads() barriers, so is
 * potentially more efficient, but requires that nWarps == nBanks.
 *
 * The synchronized version has M __syncthreads() barriers per iteration, but
 * can have an arbitrary relationship between nWarps and nBanks (this is only
 * useful if nWarps > nBanks)
 *
 * Inputs:
 *   D          # of rows in X, and dimension of L
 *   N          # of columns in X
 *   nBanks     # of banks of data to process in each block
 *   X          [D x N] data matrix
 *   L_blk      [4 x 4 x M*(M+1)/2] sub-blocks of L
 * Outputs:
 *   sqNorm     [N] output vector; y = sum((L\X).^2,1)
 */
template <typename numeric_t>
__global__ void triSolveNorm_unsynchronized( int D, int N,
        numeric_t const *X, numeric_t const *L_blk, numeric_t *sqNorm)
{
    int M = (D+4-1)/4;
    int nBanks = blockDim.y; // Also = nWarps
    // Set up the shared memory
    extern __shared__ __align__(sizeof(int4)) int S[];
    numeric_t *workspaces, *L_shared;
    tsn_SetupSharedMem(M, nBanks, S, L_blk, workspaces, L_shared);
    // Get the identity of this thread
    int tIdx = threadIdx.x;
    int tx = tIdx % 4;
    int ty = tIdx / 4;
    int bankIdx = threadIdx.y;
    // Adjust the pointers based on the bank offset
    int bankOffset = 8 * (bankIdx + nBanks * blockIdx.x);
    N -= bankOffset;
    sqNorm += bankOffset;
    X += D*bankOffset;
    workspaces += 4*8*M*bankIdx;
    // Grid stride through the data
    int blockStride = 8 * nBanks * gridDim.x;
    while (N > 0) {
        numeric_t val;
        // Load rows (ending with row 0)
        for (int i=M-1; i>=0; i--)
            tsn_Load(tx,ty,tIdx,val, workspaces+4*8*i, X,D,N, 4*i);
        // Perform the solve, moving across the columns of L
        int m = 0;
        for (int j=0; j<M; j++) {
            // Triangular solve for the diagonal element, assuming that it is
            // currently loaded into <val>
            numeric_t *Y = workspaces + 4*8*j;
            tsn_TriSolve(tx,ty,tIdx,val, Y, L_shared+4*4*m);
            // Matrix multiplications for the rest of the rows, ending with j+1
            for (int i=M-1; i>j; i--)
                tsn_MatMult(tx,ty,tIdx,val, workspaces+4*8*i, L_shared+4*4*(m+i-j), Y);
            // Increment m
            m += M-j;
        }
        // Sum the squared elements across columns
        val *= val;
        for (int i=M-2; i>=0; i--)
            val += workspaces[tIdx + 4*8*i] * workspaces[tIdx + 4*8*i];
        // Reduce within the workspace and write to global memory
        val += shfl_down(val,2,4);
        val += shfl_down(val,1,4);
        // After shfl_down, only tx==0 has a valid sum
        if ((tx==0) && (ty < N))
            sqNorm[ty] = val;
        // Increment the pointers/counters
        N -= blockStride;
        sqNorm += blockStride;
        X += D*blockStride;
    }
}

template <typename numeric_t>
__global__ void triSolveNorm( int D, int N, int nBanks,
        numeric_t const *X, numeric_t const *L_blk, numeric_t *sqNorm)
{
    int M = (D+4-1)/4;
    // Set up the shared memory
    extern __shared__ __align__(sizeof(int4)) int S[];
    numeric_t *workspaces, *L_shared;
    tsn_SetupSharedMem(M, nBanks, S, L_blk, workspaces, L_shared);
    // Get the identity of this thread
    int tIdx = threadIdx.x;
    int tx = tIdx % 4;
    int ty = tIdx / 4;
    int nWarps = blockDim.y;
    int warpIdx = threadIdx.y;
    // Adjust the pointers based on the block offset
    int blockOffset = 8 * nBanks * blockIdx.x;
    N -= blockOffset;
    sqNorm += blockOffset;
    X += D*blockOffset;
    // Grid stride through the data
    int blockStride = 8 * nBanks * gridDim.x;
    while (N > 0) {
        // Load
        for (int opIdx = warpIdx; opIdx < M*nBanks; opIdx += nWarps) {
            int bankIdx = opIdx % nBanks;
            int ii = opIdx / nBanks;
            numeric_t val;
            numeric_t *W = workspaces + 4*8*(ii+M*bankIdx);
            tsn_Load(tx,ty,tIdx,val, W, X+D*8*bankIdx, D, N, 4*ii);
            // Perform triangular solve
            if (ii==0)
                tsn_TriSolve(tx,ty,tIdx,val, W, L_shared);
        }
        __syncthreads();
        // Perform the solve, moving across the columns of L
        int m = 0;
        for (int jj=0; jj<M-1; jj++) {
            for (int opIdx = warpIdx; opIdx < (M-jj-1)*nBanks; opIdx += nWarps) {
                int bankIdx = opIdx % nBanks;
                int increment = (opIdx / nBanks) + 1;
                int ii = jj + increment;
                // Matrix multiplication
                numeric_t val;
                numeric_t *Y = workspaces + 4*8*(jj + M*bankIdx);
                numeric_t *W = workspaces + 4*8*(ii + M*bankIdx);
                tsn_MatMult(tx,ty,tIdx,val, W, L_shared+4*4*(m+increment), Y);
                // Triangular solve
                if (increment == 1)
                    tsn_TriSolve(tx,ty,tIdx,val, W, L_shared+4*4*(m+M-jj));
            }
            __syncthreads();
            m += M-jj;
        }
        // Sum the squared elements and write to global memory
        if (warpIdx < nBanks) {
            // Square and sum across workspaces
            numeric_t val = 0;
            int bankIdx = warpIdx;
            numeric_t *W_bank = workspaces + 4*8*M*bankIdx;
            for (int ii=0; ii<M; ii++)
                val += W_bank[tIdx+4*8*ii] * W_bank[tIdx+4*8*ii];
            // Reduce within the workspace and write to global memory
            val += shfl_down(val,2,4);
            val += shfl_down(val,1,4);
            // After shfl_down, only tx==0 has a valid sum
            if ((tx==0) && (ty+8*bankIdx<N))
                sqNorm[ty+8*bankIdx] = val;
        }
        // Increment the pointers/counters
        N -= blockStride;
        sqNorm += blockStride;
        X += D*blockStride;
    }
}


/* sumSqCols - CUDA kernel for computing the squared 2-norm over columns of A
 *
 * This expects:
 *   - 1-D block grid of any size
 *   - Thread block of size [32 x P]
 *   - Shared memory to hold [P] doubles
 *
 * Inputs:
 *   D       # of rows in A
 *   N       # of columns in A
 *   A       [D x N] data matrix
 * Outputs:
 *   b       [N] output vector; b = sum(A.^2,1)
 */
int const sumSqCols_blockDim_y = 16; // Easier if known at compile time
template <typename numeric_t>
__global__ void sumSqCols( int D, int N, 
        numeric_t const * __restrict__ A, numeric_t * b )
{
    // Shared memory helps us coalesce the writes
    __shared__ numeric_t S[sumSqCols_blockDim_y];
    // Some dimension-related constants
    int linearIdx = threadIdx.x + threadIdx.y*blockDim.x;
    int P = sumSqCols_blockDim_y;
    int D_eff = ((D+31)/32) * 32; // Round up to next multiple of 32
    // Shift the start 
    int readOffset = blockIdx.x * P;
    N -= readOffset;
    b += readOffset;
    A += D*readOffset;
    // Grid-stride loop over the columns
    int yStride = gridDim.x * P;
    while (N > 0) {
        // Compute the sum over this column
        numeric_t running_sum = 0;
        int readIdx_y = threadIdx.y;
        if (readIdx_y < N) {
            // For loop over the rows, 32 at a time
            /* No need to synchronize because each y belongs to a single warp */
            for (int readIdx_x=threadIdx.x; readIdx_x<D_eff; readIdx_x+=32) {
                // Read and square the data
                numeric_t value;
                value = (readIdx_x<D) ? A[readIdx_x+D*readIdx_y] : 0;
                value *= value;
                // Reduce across rows, noting that they belong to the same warp
                for (int shflOffset=16; shflOffset>0; shflOffset/=2)
                    value += shfl_down(value, shflOffset);
                // Note that this value is only valid for threadIdx.x==0
                running_sum += value;
            }
        }
        // Synchronize so that we have a coalesced write
        __syncthreads();
        if (threadIdx.x==0)
            S[threadIdx.y] = running_sum;
        __syncthreads();
        if ((linearIdx < P) && (linearIdx < N))
            b[linearIdx] = S[linearIdx];
        // Increment the pointers/counters
        N -= yStride;
        b += yStride;
        A += D*yStride;
    }
}


/* Define a "mexErrMsgIdAndTxt" function for the non-MATLAB case
 */
#ifndef MATLAB_MEX_FILE
void mexErrMsgIdAndTxt(char const *errId, char const *errMsg)
{
    std::cout << errId << std::endl << errMsg << std::endl;
}
#endif


/* Overload a single function for both single and double-precision data
 */
cublasStatus_t trsm( 
        cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
        cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
        const double *alpha, const double *A, int lda, double *B, int ldb )
{   return cublasDtrsm(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb); }
cublasStatus_t trsm( 
        cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
        cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
        const float *alpha, const float *A, int lda, float *B, int ldb )
{   return cublasStrsm(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb); }

cublasStatus_t gemm( 
        cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb,
        int m, int n, int k, const double *alpha, 
        const double *A, int lda, const double *B, int ldb, 
        const double *beta, double *C, int ldc )
{   return cublasDgemm(handle,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc); }
cublasStatus_t gemm( 
        cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb,
        int m, int n, int k, const float *alpha, 
        const float *A, int lda, const float *B, int ldb, 
        const float *beta, float *C, int ldc )
{   return cublasSgemm(handle,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc); }

void trtri(char *uplo, char *diag, ptrdiff_t *n, 
        double *A, ptrdiff_t *lda, ptrdiff_t *info)
{   return dtrtri(uplo,diag,n,A,lda,info); }
void trtri(char *uplo, char *diag, ptrdiff_t *n, 
        float *A, ptrdiff_t *lda, ptrdiff_t *info)
{   return strtri(uplo,diag,n,A,lda,info); }


/* Optimization of the kernel parameters
 * 
 * Inputs:
 *    D         Number of feature space dimensions
 *    N         Number of spikes
 * Outputs:
 *    nWarps    Number of warps (32 threads) per block
 *    nBanks    Number of banks per block
 *    nBlocks   Total number of blocks
 */
template <typename numeric_t>
void optimizeKernelParams(int D, int N, int &nWarps, int &nBanks, int &nBlocks) {
    int M = (D+4-1)/4; // ceil(D/4)
    int L_size = M*(M+1)/2;
    // Get some device info
    int deviceNo;
    cudaGetDevice(&deviceNo);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceNo);
    
    // Consider between 2..8 blocks per MP
    std::vector<int> nBlocksArr(7);
    int nBlkTemp = 2;
    for (auto &elem : nBlocksArr)
        elem = nBlkTemp++;
    // See how many banks per MP this corresponds to
    int maxMem = prop.sharedMemPerMultiprocessor / sizeof(numeric_t);
    std::vector<int> maxBanksTot(7);
    std::transform(nBlocksArr.begin(), nBlocksArr.end(), maxBanksTot.begin(),
            [maxMem,M,L_size](int nBlocks) { 
                return ((maxMem/nBlocks - 4*4*L_size)/(4*8*M)) * nBlocks;
            });

    // Because of the overhead for storing the L matrix (incurred on each block),
    // fewer blocks per MP means more banks total. If even the 2-block-per-MP
    // case supports fewer than 16 banks total, then let's use BLAS.
    // Also, always use BLAS for D >= 64
    int maxWarpsTot = prop.maxThreadsPerMultiProcessor / 32;
    if ((maxBanksTot[0] < 16) || (D >= 64)) {
        nWarps = -1;
        nBanks = -1;
        nBlocks = 2*(maxWarpsTot/sumSqCols_blockDim_y)*prop.multiProcessorCount;
        nBlocks = std::min(nBlocks, N/sumSqCols_blockDim_y + 1);
        return;
    }

    // If any of these hit 75% utilization, run in one-warp-per-bank mode
    int bankThresh = static_cast<int>(0.75 * maxWarpsTot);
    int lastSuccess = -1;
    for (int ii=0; ii<maxBanksTot.size(); ii++) {
        if (maxBanksTot[ii] >= bankThresh)
            lastSuccess = ii;
    }
    int blocksPerMP;
    if (lastSuccess > 0) {
        // One-warp-per-bank mode
        blocksPerMP = nBlocksArr[lastSuccess];
        nBanks = maxBanksTot[lastSuccess] / blocksPerMP;
        nBanks = std::min(nBanks, maxWarpsTot/blocksPerMP);
        nWarps = nBanks;
    }
    else {
        // We have enough banks to run our kernel, but not enough to have one
        // warp per bank. We want to maximize the number of banks, but we also
        // have a preference for having more blocks (with fewer banks each).
        lastSuccess = 0;
        int nBanksTot = maxBanksTot[0];
        for (int ii=1; ii<maxBanksTot.size(); ii++) {
            if (maxBanksTot[ii] >= 0.9 * nBanksTot) {
                lastSuccess = ii;
                nBanksTot = maxBanksTot[ii];
            }
        }
        blocksPerMP = nBlocksArr[lastSuccess];
        nBanks = nBanksTot / blocksPerMP;
        nWarps = maxWarpsTot / blocksPerMP;
    }
    nBlocks = 2 * blocksPerMP * prop.multiProcessorCount;
    nBlocks = std::min(nBlocks, N/(8*nBanks) + 1);
}


/* Main routine for computing the Mahalanobis distance
 * 
 * Inputs:
 *    D         #rows in X and size of L (number of feature space dimensions)
 *    N         #cols in X (number of spikes)
 *    L         [D x D] lower Cholesky-factorized covariance matrix
 *    d_X       [D x N] data matrix (on GPU device)
 % Outputs:
 *    d_delta   [N] squared Mahalanobis distances (on GPU device)
 */
template <typename numeric_t>
void computeMahalDist(int D, int N, 
        numeric_t const *L, numeric_t const *d_X, numeric_t *d_delta)
{
    char const * const cudaErrId = "MoDT:calcMahalDistGpu:cudaError";
    // Determine parameters for our kernel
    int nWarps, nBanks, nBlocks;
    optimizeKernelParams<numeric_t>(D, N, nWarps, nBanks, nBlocks);

    if (nWarps > 0) {
        // Use our kernel
        // Copy L into blocks
        std::vector<numeric_t> L_blocks = copyLToBlocks(D, L);
        // Copy this to the device
        numeric_t *d_L;
        cudaError_t cudaStat;
        int M = (D+4-1)/4;
        int L_size = M*(M+1)/2;
        cudaStat = cudaMalloc((void**) &d_L, 4*4*L_size * sizeof(*d_L));
        std::unique_ptr<numeric_t,CudaDeleter> cleanup_L(d_L);
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, "Failed to allocate CUDA memory for L");
        cudaStat = cudaMemcpy(d_L, L_blocks.data(), 4*4*L_size * sizeof(*d_L), 
                cudaMemcpyHostToDevice);
        // Launch the kernel
        dim3 tPerBlk(32,nWarps);
        int shMemPerBlk = (4*4*L_size + 4*8*M*nBanks)*sizeof(*d_X);
        if (nWarps==nBanks) {
            triSolveNorm_unsynchronized<<<nBlocks, tPerBlk, shMemPerBlk>>>
                    (D, N, d_X, d_L, d_delta);
        }
        else {
            triSolveNorm<<<nBlocks, tPerBlk, shMemPerBlk>>>
                    (D, N, nBanks, d_X, d_L, d_delta);
        }
        // Make sure it completes before we let the temp vars go out of scope
        cudaStat = cudaDeviceSynchronize();
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, cudaGetErrorString(cudaStat));
    }
    else if (D < 2048) {
        // First, copy and invert L
        std::vector<numeric_t> Linv(L, L+D*D);
        char uplo = 'L';    // L is lower triangular
        char diag = 'N';    // L is not unit triangular
        ptrdiff_t D_64 = D;
        ptrdiff_t info = 0;      // Status output for trtri
        trtri(&uplo, &diag, &D_64, Linv.data(), &D_64, &info);
        if (info != 0)
            mexErrMsgIdAndTxt("MoDT:calcMahalDistGpu:LAPACKError", 
                    "LAPACK routine ?trtri() exited with error");
        // Initialize
        cublasStatus_t cublasStat;
        cudaError_t cudaStat;
        cublasHandle_t handle;
        cublasStat = cublasCreate(&handle);
        cublasHandleCleanupWrapper cleanup_handle(handle);
        if (cublasStat != CUBLAS_STATUS_SUCCESS)
            mexErrMsgIdAndTxt(cudaErrId, "Unable to initialize cuBLAS context");
        // Move Linv to the device
        numeric_t *d_Linv;
        cudaStat = cudaMalloc((void**)&d_Linv, D*D*sizeof(*d_Linv));
        std::unique_ptr<numeric_t,CudaDeleter> cleanup_Linv(d_Linv);
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, "Failed to allocate CUDA memory for Linv");
        cublasStat = cublasSetMatrix(D,D,sizeof(*d_Linv), Linv.data(),D, d_Linv,D);
        if (cublasStat != CUBLAS_STATUS_SUCCESS)
            mexErrMsgIdAndTxt(cudaErrId, "cuBLAS error copying Linv to GPU");
        // Allocate space for Y = Linv * X
        numeric_t *d_Y;
        cudaStat = cudaMalloc((void**)&d_Y, D*N*sizeof(*d_X));
        std::unique_ptr<numeric_t,CudaDeleter> cleanup_Y(d_Y);
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, "Failed to allocate CUDA memory for Y");
        // Call GEMM
        numeric_t alpha = 1;
        numeric_t beta = 0;
        cublasStat = gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                D, N, D, &alpha, d_Linv, D, d_X, D, &beta, d_Y, D);
        if (cublasStat != CUBLAS_STATUS_SUCCESS)
            mexErrMsgIdAndTxt(cudaErrId, "cuBLAS error during TRSM");
        // Call our kernel for computing the squared norm
        dim3 threadsPerBlock(32, sumSqCols_blockDim_y);
        sumSqCols<<<nBlocks,threadsPerBlock>>>(D, N, d_Y, d_delta);
        // Make sure it completes before we let the temp vars go out of scope
        cudaStat = cudaDeviceSynchronize();
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, cudaGetErrorString(cudaStat));
    }
    else {
        // For very large D, use TRSM
        cublasStatus_t cublasStat;
        cudaError_t cudaStat;
        cublasHandle_t handle;
        // Initialize
        cublasStat = cublasCreate(&handle);
        cublasHandleCleanupWrapper cleanup_handle(handle);
        if (cublasStat != CUBLAS_STATUS_SUCCESS)
            mexErrMsgIdAndTxt(cudaErrId, "Unable to initialize cuBLAS context");
        // Move L to the device
        numeric_t *d_L;
        cudaStat = cudaMalloc((void**)&d_L, D*D*sizeof(*L));
        std::unique_ptr<numeric_t,CudaDeleter> cleanup_L(d_L);
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, "Failed to allocate CUDA memory for L");
        cublasStat = cublasSetMatrix(D, D, sizeof(*L), L, D, d_L, D);
        if (cublasStat != CUBLAS_STATUS_SUCCESS)
            mexErrMsgIdAndTxt(cudaErrId, "cuBLAS error copying L to GPU");
        // Copy X because it'll be overwritten by TRSM
        numeric_t *d_X_copy;
        cudaStat = cudaMalloc((void**)&d_X_copy, D*N*sizeof(*d_X));
        std::unique_ptr<numeric_t,CudaDeleter> cleanup_X_copy(d_X_copy);
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, "Failed to allocate CUDA memory for X");
        cudaStat = cudaMemcpy(d_X_copy, d_X, D*N*sizeof(*d_X), 
                cudaMemcpyDeviceToDevice);
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, "Failed to make copy of X");
        // Call TRSM
        numeric_t const alpha = 1;     // Scaling applied to X
        cublasStat = trsm( handle, CUBLAS_SIDE_LEFT, 
                CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
                D, N, &alpha, d_L, D, d_X_copy, D );
        if (cublasStat != CUBLAS_STATUS_SUCCESS)
            mexErrMsgIdAndTxt(cudaErrId, "cuBLAS error during TRSM");
        // Call our kernel for computing the squared norm
        dim3 threadsPerBlock(32, sumSqCols_blockDim_y);
        sumSqCols<<<nBlocks,threadsPerBlock>>>(D, N, d_X_copy, d_delta);
        // Make sure it completes before we let the temp vars go out of scope
        cudaStat = cudaDeviceSynchronize();
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, cudaGetErrorString(cudaStat));
    }
}


#ifdef MATLAB_MEX_FILE
/* Some MATLAB helpers because these function names are ridiculous
 */
template <typename numeric_t>
numeric_t const * gpuPtr(mxGPUArray const *mgpu_X)
{   return static_cast<numeric_t const*>(mxGPUGetDataReadOnly(mgpu_X)); }
template <typename numeric_t>
numeric_t * gpuPtr(mxGPUArray *mgpu_X)
{   return static_cast<numeric_t*>(mxGPUGetData(mgpu_X)); }
template <typename numeric_t>
numeric_t * mxPtr(mxArray *mx_X)
{   return static_cast<numeric_t*>(mxGetData(mx_X)); }
template <typename numeric_t>
numeric_t const * mxPtr(mxArray const *mx_X)
{   return static_cast<numeric_t const*>(mxGetData(mx_X)); }

/* Main entry point into this mex file
 * Inputs and outputs are arrays of mxArray pointers
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
    // Check the inputs
    char const * const errId = "MoDT:calcMahalDistGpu:InvalidInput";
    if (nrhs != 2)
        mexErrMsgIdAndTxt(errId, "This function requires 2 inputs");
    // L = input 0
    mxArray const *mx_L = prhs[0];
    if (mxIsGPUArray(mx_L))
        mexErrMsgIdAndTxt(errId, "L should not be a gpuArray");
    int D = mxGetM(mx_L);
    mxClassID numericType = mxGetClassID(mx_L);
    if ((D==0) || (D!=mxGetN(mx_L)) || 
            !((numericType==mxDOUBLE_CLASS) || (numericType==mxSINGLE_CLASS)) )
        mexErrMsgIdAndTxt(errId, "L must be a square real matrix");
    // X = input 1
    mxArray const *mx_X = prhs[1];
    if (!mxIsGPUArray(mx_X))
        mexErrMsgIdAndTxt(errId, "X must a gpuArray");
    mxGPUArray const *mgpu_X = mxGPUCreateFromMxArray( mx_X );
    mxGPUArrayCleanupWrapper cleanup_X(mgpu_X);
    if ((mxGPUGetNumberOfElements(mgpu_X)==0) || 
            (mxGPUGetNumberOfDimensions(mgpu_X)!=2) ||
            (mxGPUGetClassID(mgpu_X)!=numericType))
        mexErrMsgIdAndTxt(errId, "X must a 2-D array of the same type as L");
    size_t const *dims = mxGPUGetDimensions( mgpu_X );
    if (dims[0] != D)
        mexErrMsgIdAndTxt(errId, "X must be a [D x N] gpuArray");
    int N = dims[1];
    
    // Allocate memory for the output
    std::vector<size_t> dims_delta = {(size_t) N, 1};
    mxGPUArray *mgpu_delta = mxGPUCreateGPUArray(2, dims_delta.data(),
            numericType, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArrayCleanupWrapper cleanup_delta(mgpu_delta);
    // Compute delta = sum((L\X).^2,1)'
    switch (numericType) {
        case mxDOUBLE_CLASS:
            computeMahalDist(D, N, mxPtr<double>(mx_L), gpuPtr<double>(mgpu_X),
                             gpuPtr<double>(mgpu_delta) );
            break;
        case mxSINGLE_CLASS:
            computeMahalDist(D, N, mxPtr<float>(mx_L), gpuPtr<float>(mgpu_X),
                             gpuPtr<float>(mgpu_delta) );
            break;
    }
    
    // Output 0 = delta
    if (nlhs >= 1) {
        // Wrap it in a mxArray
        mxArray *mx_delta = mxGPUCreateMxArrayOnGPU( mgpu_delta );
        plhs[0] = mx_delta;
    }
    // Cleanup done by our wrapper classes
}

#else

template <typename numeric_t>
void demo_mahalDist(ptrdiff_t D, ptrdiff_t N)
{
    // Create the data
    thrust::device_vector<numeric_t> X(D*N,1);
    thrust::device_vector<numeric_t> delta(N,1);
    std::vector<numeric_t> L(D*D,0);
    for (int i=0; i<D; i++)
        L[i+D*i] = 1;    
    // Extract the raw pointers
    numeric_t *d_X = thrust::raw_pointer_cast(X.data());
    numeric_t *d_delta = thrust::raw_pointer_cast(delta.data());
    numeric_t *h_L = L.data();
    // Compute delta = sum((L\X).^2,1)'
    computeMahalDist(D, N, h_L, d_X, d_delta);
}    

/* Main entry point if this is compiled externally (i.e. not as a MEX file)
 * This sets up and runs a simple example program and is suitable for benchmarking
 */
int main(int argc, char* argv[])
{
    // Define the sizes
    ptrdiff_t D=12, N=500000;
    bool use_single = false;
    int c;
    while ( (c = getopt(argc,argv,"D:N:s")) != -1 ) {
        switch (c) {
            case 'D': D = std::atoi(optarg); break; 
            case 'N': N = std::atoi(optarg); break; 
            case 's': use_single = true;    break;
        }
    }
    // Call the appropriate demo type
    if (use_single)
        demo_mahalDist<float>(D, N);
    else
        demo_mahalDist<double>(D, N);
}

#endif
