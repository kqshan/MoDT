/*==============================================================================
 * Compute C = A*diag(w)*A' where w=weights(k,:) (A, weights both gpuArrays)
 *   C = calcCov(A, weights, k)
 * 
 * Returns:
 *   C       [D x D] Symmetric positive definite gpuArray
 * Required arguments:
 *   A       [D x N] gpuArray
 *   weights [N x K] gpuArray
 *   k       Cluster index (1..K)
 * 
 * This function requires all gpuArrays to be double-precision. 
 * 
 * When D < 32, this optimizes the data access by breaking A into chunks, using
 * a custom CUDA kernel to compute the weighted covariance independently for
 * each chunk, then reducing the result across chunks.
 * When D >= 32, this custom kernel doesn't work (needs > 1024 threads/block),
 * so we produce a scaled copy of A (scaled by sqrt(w)) and then call dgemm.
 *
 * Kevin Shan, 2016-06-14
 *============================================================================*/

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include <stdio.h>
#include <algorithm>
#include <utility>

/* CUDA kernel for computing A*diag(w)*A'
 *
 * This expects:
 *  - 1-D thread block of size [32*loadWarps + D*(D+1)]
 *  - Shared memory to hold (D+1)*64 doubles
 *  - 1-D block grid of any size
 *
 * This kernel uses (32*loadWarps) non-computing threads to load data into
 * shared memory and D*(D+1) compute threads to compute the weighted covariance.
 *
 * Inputs:
 *    D             # rows in A, and dimension of C
 *    N             # columns in A, and entries in w
 *    loadWarps     # warps (32 threads per warp) to dedicate to loading data
 *    A             [D x N] data matrix
 *    w             [N] weight vector
 * Outputs:
 *    C             [D x D x #blocks] matrix to store the output
 */
__global__ void calcWeightedCov(
        int const D, int const N, int const loadWarps,
        double const * __restrict__ A, double const * __restrict__ w, 
        double *C)
{
    /* Dynamically-allocated shared memory, length = (D+1)*64
     * This is split into 2 pages of size (32*D+32), laid out as:
     *  [ A_page0 ; w_page0; A_page1 ; w_page1 ]
     * In our main loop, we alternate pages so that the next load can occur
     * while we are still computing on the previously-loaded page. */
    extern __shared__ double S[];
    double * A_buff = S;
    int const A_buffSize = 32*D;
    double * w_buff = S + A_buffSize;
    int const w_buffSize = 32;
    int const pageSize = A_buffSize + w_buffSize;
    
    // Figure out what this thread is supposed to be doing
    int const loadThreads = 32 * loadWarps;
    int const tIsLoad = (threadIdx.x < loadThreads);
    int const tIdx = (tIsLoad) ? (threadIdx.x) : (threadIdx.x-loadThreads);
    int ii, jj;
    if (tIsLoad) {
        /* Since we are loading 32 columns at a time, exactly one warp
         * will be tasked with loading from w.
         * ii = wIdx if this thread belongs to that warp, ii = -1 otherwise */
        ii = ((tIdx/32)==((D+1)%loadWarps)) ? (tIdx%32) : (-1);
    } else {
        /* Compute threads are organized into a [D x (D+1)] array.
         *          [ 00a  00b  01b  02b ]
         * For D=3: [ 10a  11a  11b  12b ]
         *          [ 20a  21a  22a  22b ]
         * The entries marked __a will compute the covariance over
         * even-numbered columns and __b will compute the odd columns. */
        jj = tIdx/D;
        ii = tIdx - D*jj;
        if (jj > ii) {
            jj--;
            // Shift the pointers up by one column so it does the odd columns
            A_buff += D;
            w_buff++;
        }
    }
    
    // Main loop: load and compute the covariance
    int currPage = 0;       // Start on buffer page 0
    double result_acc = 0;  // Local accumulator for C(ii, jj)
    for (int colOffset=blockIdx.x*32; colOffset<N; colOffset+=32*gridDim.x) {
        // Point to the appropriate buffer page
        double * A_page = A_buff + currPage*pageSize;
        double * w_page = w_buff + currPage*pageSize;
        // Load A and w from global memory
        if (tIsLoad) {
            if (colOffset+32 > N) {
                // This page is incomplete; pad it with zeroes
                int nLeftover = N - colOffset;
                for (int idx=tIdx; idx<A_buffSize; idx+=loadThreads)
                    A_page[idx] = (idx<D*nLeftover) ? A[idx+D*colOffset] : 0;
                if (ii >= 0)
                    w_page[ii] = (ii<nLeftover) ? w[ii+colOffset] : 0;
            } else {
                // No need to worry about indices going out of bounds
                for (int idx=tIdx; idx<A_buffSize; idx+=loadThreads)
                    A_page[idx] = A[idx + D*colOffset];
                if (ii >= 0)
                    w_page[ii] = w[ii + colOffset];
            }
        }
        // Wait for load to complete before computing on this page
        __syncthreads();
        // Compute
        if (!tIsLoad) {
            for (int kk=0; kk<32; kk+=2)
                result_acc += A_page[ii+D*kk] * A_page[jj+D*kk] * w_page[kk];
        }
        // Switch buffer pages
        currPage = (currPage+1) % 2;
    }
    
    // Sum the compute halves and write output to global memory
    if (!tIsLoad) {
        // Write to shared memory
        __syncthreads();
        S[tIdx] = result_acc;
        __syncthreads();
        // Add the result from the other compute half
        int transposeIdx = jj + ii*D;
        if (tIdx==ii+D*jj) transposeIdx += D; // See compute thread layout
        result_acc += S[transposeIdx];
        // Write to global memory
        double * C_block = C + blockIdx.x*D*D;
        C_block[ii+D*jj] = result_acc;
    }
}


/* CUDA kernel for summing across the columns of X
 *
 * This assumes a 1-D thread block and a 1-D block grid, and that there are
 * enough threads*blocks that each row of X gets its own thread.
 *
 * Inputs
 *    nRows     # rows in X
 *    nCols     # columns in X
 *    X         [rows x cols] data matrix
 * Outputs
 *    Y         [rows x 1] data vector
 */
__global__ void sumColumns( int const nRows, int const nCols, 
        double const * __restrict__ X, double *Y )
{
    // Check if this thread falls in range
    int const row = threadIdx.x + blockIdx.x*blockDim.x;
    if (row < nRows) {
        // For loop to sum over columns
        double result_acc = 0;
        for (int col=0; col<nCols; col++)
            result_acc += X[row+col*nRows];
        // Write to output
        Y[row] = result_acc;
    }
}


/* CUDA kernel for producing a scaled copy of A, scaled by sqrt(w)
 *
 * This assumes 1-D thread blocks, shared memory of the same size, and a 
 * 1-D block grid. Each [P x 1] thread block will process a [D x P] block of
 * data at a time, and take grid-sized strides across the data.
 *
 * Inputs:
 *    D     Number of rows in A, B
 *    N     Number of columns in A, B
 *    A     [D x N] data matrix
 *    w     [N] weight vector
 * Outputs:
 *    B     [D x N] scaled data matrix
 */
__global__ void sqrtScaledCopy(int D, int N, 
        double const *A, double const *w, double *B)
{
    // Dynamically-allocated shared memory
    extern __shared__ double S[];
    // Notational convenience for some dimensions
    int tIdx = threadIdx.x;
    int P = blockDim.x;
    
    // For loop over the [N] dimension
    int blockStart = blockIdx.x * P;
    int blockStride = P * gridDim.x;
    for (int bColOffset=blockStart; bColOffset<N; bColOffset+=blockStride) {
        // Read and sqrt() the weight vector
        int wIdx = tIdx + bColOffset;
        if (wIdx < N)
            S[tIdx] = sqrt(w[wIdx]);
        __syncthreads();
        // Read, scale, and write the data in these columns
        int bDataOffset = bColOffset * D;
        int dataCount = min(P,N-bColOffset) * D;
        int dataStride = blockDim.x;
        for (int dataIdx=tIdx; dataIdx<dataCount; dataIdx+=dataStride) {
            int dataCol = dataIdx/D;
            B[dataIdx+bDataOffset] = S[dataCol] * A[dataIdx+bDataOffset];
        }
        __syncthreads();
    }
}


/* Simple wrapper classes so we can do RAII-style cleanup 
 */
struct mxGPUArrayCleanupWrapper {
    mxGPUArray const *mxgpu_ptr;
    mxGPUArrayCleanupWrapper(mxGPUArray const *p) {mxgpu_ptr = p;}
    ~mxGPUArrayCleanupWrapper(void) {mxGPUDestroyGPUArray(mxgpu_ptr);}
};
struct cublasHandleCleanupWrapper {
    cublasHandle_t handle;
    cublasHandleCleanupWrapper(cublasHandle_t h) {handle = h;}
    ~cublasHandleCleanupWrapper(void) {cublasDestroy(handle);}
};


/* Main entry point into this mex file
 * Inputs and outputs are arrays of mxArray pointers
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
    // Check the inputs
    char const * const errId = "MoDT:weightCovGpu:InvalidInput";
    if (nrhs != 3)
        mexErrMsgIdAndTxt(errId, "This function requires 3 inputs");
    // A = input 0
    mxArray const *mx_A = prhs[0];
    if (!mxIsGPUArray(mx_A))
        mexErrMsgIdAndTxt(errId, "A must be a gpuArray");
    mxGPUArray const *mgpu_A = mxGPUCreateFromMxArray( mx_A );
    mxGPUArrayCleanupWrapper cleanup_A(mgpu_A);
    if ((mxGPUGetNumberOfElements(mgpu_A)==0) || 
            (mxGPUGetNumberOfDimensions(mgpu_A)!=2) ||
            (mxGPUGetClassID(mgpu_A)!=mxDOUBLE_CLASS))
        mexErrMsgIdAndTxt(errId, "A must a 2-D double-precision gpuArray");
    size_t const *dims = mxGPUGetDimensions( mgpu_A );
    size_t D = dims[0];
    size_t N = dims[1];
    double const *d_A = (double const *) mxGPUGetDataReadOnly( mgpu_A );
    // weights = input 1
    mxArray const *mx_weights = prhs[1];
    if (!mxIsGPUArray(mx_weights))
        mexErrMsgIdAndTxt(errId, "weights must a gpuArray");
    mxGPUArray const *mgpu_weights = mxGPUCreateFromMxArray( mx_weights );
    mxGPUArrayCleanupWrapper cleanup_weights(mgpu_weights);
    if ((mxGPUGetNumberOfElements(mgpu_weights)==0) || 
            (mxGPUGetNumberOfDimensions(mgpu_weights)!=2) ||
            (mxGPUGetClassID(mgpu_weights)!=mxDOUBLE_CLASS))
        mexErrMsgIdAndTxt(errId, "weights must a 2-D double-precision gpuArray");
    dims = mxGPUGetDimensions( mgpu_weights );
    size_t N_weights = dims[0];
    size_t K = dims[1];
    if (N_weights != N)
        mexErrMsgIdAndTxt(errId, "weights must be a [N x K] gpuArray");
    double const *d_weights = 
            (double const *) mxGPUGetDataReadOnly( mgpu_weights );
    // k (weight index) = input 2
    mxArray const *mx_k = prhs[2];
    if (!mxIsScalar(mx_k))
        mexErrMsgIdAndTxt(errId, "k must be a scalar");
    ptrdiff_t weight_index = ((ptrdiff_t) mxGetScalar(mx_k)) - 1;
    if ((weight_index < 0) || (weight_index >= K))
        mexErrMsgIdAndTxt(errId, "k is out of bounds");
    
    // Now we can get a pointer to the selected column of the weights
    double const *d_w = d_weights + weight_index*N;
    
    // Allocate memory for the output
    size_t dims_C[2];
    dims_C[0] = D; dims_C[1] = D;
    mxGPUArray *mgpu_C = mxGPUCreateGPUArray(2, dims_C, 
            mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArrayCleanupWrapper cleanup_C(mgpu_C);
    double *d_C = (double *) mxGPUGetData(mgpu_C);
    
    // Compute C = A*diag(w)*A'
    char const * const cudaErrId = "MoDT:weightCovGpu:cudaError";
    if (D < 32) {
        /* When D is small, we can improve GPU utilization by breaking X into
         * batches, computing the weighted covariance of each batch, and then
         * summing across the batches. */
        int computeThreads = D*(D+1);
        int sharedMemPerBlock = (D+1)*64*sizeof(*d_A);
        int loadWarps = 1;
        int blocksPerMP = 16;
        /* Some hardcoded optimization (specific to Compute Capability 3.0) to
         * maximize the number of load threads per block, but only if it doesn't
         * affect the total # of blocks per multiprocessor. */
        int computeWarps = (computeThreads+31)/32;
        int const sharedMemLimit = 48*1024;
        int const warpLimit = 64;
        blocksPerMP = min(blocksPerMP, sharedMemLimit/sharedMemPerBlock);
        blocksPerMP = min(blocksPerMP, warpLimit/(computeWarps+loadWarps));
        int extraWarps = warpLimit - (computeWarps+loadWarps)*blocksPerMP;
        loadWarps += extraWarps/blocksPerMP;
        /* Figure out how many blocks. More blocks can potentially assist in GPU
         * scheduling, but we get diminishing returns after we are able to
         * saturate all available multiprocessors. Assumes 8 MPs. */
        int numBlocks = min( (int) (N/1024)+1, 8*blocksPerMP );
        // Allocate memory for the outputs
        double *d_CBatched;
        cudaError_t cudaStat = cudaMalloc((void**)&d_CBatched, 
                D*D*numBlocks*sizeof(*d_CBatched));
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, "Failed to allocate CUDA memory");
        // Launch our CUDA kernel to calculate the weighted covariance
        int threadsPerBlock = computeThreads + 32*loadWarps;
        calcWeightedCov<<<numBlocks,threadsPerBlock,sharedMemPerBlock>>>
                (D, N, loadWarps, d_A, d_w, d_CBatched);
        // Sum the batched results
        sumColumns<<<(D*D+31)/32, 32>>>(D*D, numBlocks, d_CBatched, d_C);
        // Free the memory that we'd allocated for the outputs
        cudaFree(d_CBatched);
    }
    else {
        /* For larger D, we'll use cuBLAS.
         * Strangley, it seems that cuBLAS gemm is faster than syrk. */
        // Initialize cuBLAS
        cublasStatus_t stat;
        cublasHandle_t handle;
        stat = cublasCreate(&handle);
        cublasHandleCleanupWrapper cleanup_handle(handle);
        if (stat != CUBLAS_STATUS_SUCCESS)
            mexErrMsgIdAndTxt(cudaErrId, "Unable to initialize cuBLAS context");
        // Allocate memory for the scaled copy
        double *d_X;
        cudaError_t cudaStat = cudaMalloc((void**)&d_X, D*N*sizeof(*d_A));
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, "Failed to allocate CUDA memory");
        // X = bsxfun(@times, A, sqrt(w)')
        int threadsPerBlock = 256;
        int numBlocks = min(128, (int) ((N+threadsPerBlock-1)/threadsPerBlock));
        int sharedMemPerBlock = threadsPerBlock*sizeof(*d_A);
        sqrtScaledCopy<<<numBlocks,threadsPerBlock,sharedMemPerBlock>>>
                (D, N, d_A, d_w, d_X);
        // C = X*X'
        double const alpha = 1; // Scaling applied to X*X'
        double const beta = 0;  // Scaling applied to C
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                D, D, N, &alpha, d_X, D, d_X, D, &beta, d_C, D);
        // Free the memory we'd allocated for the scaled copy
        cudaFree(d_X);
        // Check for a linear algebra error
        if (stat != CUBLAS_STATUS_SUCCESS)
            mexErrMsgIdAndTxt(cudaErrId, "cuBLAS error during GEMM");
    }
    
    // Output 0 = C
    if (nlhs >= 1) {
        // Wrap it in a mxArray
        mxArray *mx_C = mxGPUCreateMxArrayOnGPU( mgpu_C );
        plhs[0] = mx_C;
    }
    
    // Cleanup done by our wrapper classes
}
