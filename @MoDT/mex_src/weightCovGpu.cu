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

#ifdef MATLAB_MEX_FILE
#include "mex.h"
#include "gpu/mxGPUArray.h"
#endif

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <thrust/device_vector.h>

#include <iostream>
#include <cstdlib>
#include <algorithm>


/* CUDA kernel for computing A*diag(w)*A'
 *
 * This expects:
 *  - 1-D thread block with at least D*(D+1)/2 threads
      (it should also have at least 32 threads)
 *  - Shared memory to hold (D+1)*32 doubles
 *  - 1-D block grid of any size
 *
 * Inputs:
 *    D             # rows in A, and dimension of C
 *    N             # columns in A, and entries in w
 *    A             [D x N] data matrix
 *    w             [N] weight vector
 * Outputs:
 *    C             [D x D x #blocks] matrix to store the output
 */
int const READ_BUFF_SZ = 32;
__global__ void calcWeightedCov(
        int const D, int const N, double const *A, double const *w, double *C)
{
    // Dynamically-allocated shared memory, length = (D+1)*32
    extern __shared__ double S[];
    double *w_buff = S;                // [32 x 1] buffer for weights
    double *A_buff = S + READ_BUFF_SZ; // [D x 32] buffer for data matrix
    
    // Figure out what this thread is supposed to be doing
    int const tIdx = threadIdx.x;
    int const nThreads = blockDim.x;
    // Loading phase: the first 32 threads will load w, rest will do A
    int const readOps_w = READ_BUFF_SZ; // Gonna assume nThreads > READ_BUFF_SZ
    int const tIdx_A = (tIdx - readOps_w + nThreads) % nThreads;
    int const readOps_A = D * READ_BUFF_SZ * sizeof(double)/sizeof(double2);
    // Compute phase: this thread will compute C(ii,jj)
    int ii = -1, jj = -1;
    if (tIdx < D*(D+1)/2) {
        ii = tIdx % D; jj = tIdx / D;
        if (jj > ii) {
            ii = D-1 - ii; jj = D - jj;
        }
    }
    // And we will store the result in this local register
    double result_acc = 0;
    
    // Main loop: grid-stride loop over complete buffers
    int readOffset = READ_BUFF_SZ * blockIdx.x;
    for ( ; readOffset <= N-READ_BUFF_SZ; readOffset += READ_BUFF_SZ*gridDim.x) {
        // Load w from memory
        if (tIdx < readOps_w)
            w_buff[tIdx] = w[readOffset + tIdx];
        // Load A from memory, 16 bytes at a time
        double2 const *src = reinterpret_cast<double2 const*>(A + D*readOffset);
        double2 *tgt = reinterpret_cast<double2*>(A_buff);
        for (int idx = tIdx_A; idx < readOps_A; idx += nThreads)
            tgt[idx] = src[idx];
        // Wait for load to complete
        __syncthreads();
        // Compute
        if (ii >= 0) {
            // Compute the result for this buffer
            double local_acc = 0;
            for (int kk=0; kk<READ_BUFF_SZ; kk++)
                local_acc += A_buff[ii+D*kk] * A_buff[jj+D*kk] * w_buff[kk];
            // Add it to the running total
            result_acc += local_acc;
        }
        // Wait for computation to finish
        __syncthreads();
    }
    // Deal with remaining data
    int nLeftover = N - readOffset;
    if (nLeftover > 0) {
        // Load remaining w
        if (tIdx < nLeftover)
            w_buff[tIdx] = w[readOffset + tIdx];
        // Load remaining A
        for (int idx = tIdx_A; idx < D*nLeftover; idx += nThreads)
            A_buff[idx] = A[D*readOffset + idx];
        // Wait for load to complete.
        __syncthreads();
        // Compute
        if (ii >= 0) {
            double local_acc = 0;
            for (int kk=0; kk<nLeftover; kk++)
                local_acc += A_buff[ii+D*kk] * A_buff[jj+D*kk] * w_buff[kk];
            result_acc += local_acc;
        }
    }
    
    // Write output to global memory
    if (ii >= 0) {
        double *C_block = C + D*D*blockIdx.x;
        C_block[ii+D*jj] = result_acc;
        // Write the other half as well
        if (ii != jj)
            C_block[jj+D*ii] = result_acc;
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
#ifdef MATLAB_MEX_FILE
struct mxGPUArrayCleanupWrapper {
    mxGPUArray const *mxgpu_ptr;
    mxGPUArrayCleanupWrapper(mxGPUArray const *p) {mxgpu_ptr = p;}
    ~mxGPUArrayCleanupWrapper(void) {mxGPUDestroyGPUArray(mxgpu_ptr);}
};
#endif
struct cublasHandleCleanupWrapper {
    cublasHandle_t handle;
    cublasHandleCleanupWrapper(cublasHandle_t h) {handle = h;}
    ~cublasHandleCleanupWrapper(void) {cublasDestroy(handle);}
};

/* Define a "mexErrMsgIdAndTxt" function for the non-MATLAB case
 */
#ifndef MATLAB_MEX_FILE
void mexErrMsgIdAndTxt(char const *errId, char const *errMsg)
{
    std::cout << errId << std::endl << errMsg << std::endl;
}
#endif


/* Main routine for computing the weighted covariance
 *
 * Inputs:
 *    D         #rows in A (number of feature space dimensions)
 *    N         #cols in A and entries in w (number of spikes)
 *    d_A       [D x N] data matrix (on GPU device)
 *    d_w       [N] vector of weights (on GPU device)
 * Outputs:
 *    d_C       [D x D] covariance matrix
 */
void computeWeightedCov(size_t D, size_t N, 
        double const * d_A, double const * d_w, double * d_C)
{
    char const * const cudaErrId = "MoDT:weightCovGpu:cudaError";
    if (D < 32) {
        /* When D is small, we can improve GPU utilization by breaking X into
         * batches, computing the weighted covariance of each batch, and then
         * summing across the batches. */
        
        // Figure out how to maximize the kernel residency
        int deviceNo;
        cudaGetDevice(&deviceNo);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceNo);
        int numMPs = prop.multiProcessorCount;
        /* I haven't found a way to programatically determine the maximum number
         * of blocks per multiprocessor or the total shared memory per MP, so 
         * this is going to be hardcoded for Compute Capability 3.0 */
        int maxWarpsPerMP = 64;
        int maxSharedMemPerMP = 48*1024;
        int maxBlocksPerMP = 16;
        // Determine the maximum number of blocks per multiprocessor (MP)
        int minWarpsPerBlock = (D*(D+1)/2 + 31)/32;
        int sharedMemPerBlock = (D+1)*32*sizeof(*d_A);
        int blocksPerMP = std::min( maxWarpsPerMP/minWarpsPerBlock, 
                maxSharedMemPerMP/sharedMemPerBlock );
        blocksPerMP = std::min(blocksPerMP, maxBlocksPerMP);
        // Allocate additional warps if the limit allows. Although these won't
        // participate in computation, they can help load data.
        int warpsPerBlock = std::max(minWarpsPerBlock, maxWarpsPerMP/blocksPerMP);
        // And decide on how many blocks total
        int numBlocks = std::min( (int)(N/1024)+1, blocksPerMP*numMPs );
        
        // Allocate memory for the outputs
        double *d_CBatched;
        cudaError_t cudaStat = cudaMalloc((void**)&d_CBatched, 
                D*D*numBlocks*sizeof(*d_CBatched));
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, "Failed to allocate CUDA memory");
        // Launch our CUDA kernel to calculate the weighted covariance
        int threadsPerBlock = 32*warpsPerBlock;
        calcWeightedCov<<<numBlocks,threadsPerBlock,sharedMemPerBlock>>>
                (D, N, d_A, d_w, d_CBatched);
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
}


#ifdef MATLAB_MEX_FILE
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
    computeWeightedCov(D, N, d_A, d_w, d_C);
    
    // Output 0 = C
    if (nlhs >= 1) {
        // Wrap it in a mxArray
        mxArray *mx_C = mxGPUCreateMxArrayOnGPU( mgpu_C );
        plhs[0] = mx_C;
    }
    
    // Cleanup done by our wrapper classes
}

#else

/* Main entry point if this is compiled externally (i.e. not as a MEX file)
 * This sets up and runs a simple example program and is suitable for benchmarking
 */
int main(int argc, char* argv[])
{
    // Define the sizes
    size_t D = (argc > 1) ? (size_t) std::atoi(argv[1]) : 12;
    size_t N = (argc > 2) ? (size_t) std::atoi(argv[2]) : 500000;
    // Create the data
    thrust::device_vector<double> A(D*N,1);
    thrust::device_vector<double> w(N,1);
    thrust::device_vector<double> C(D*D,0);
    // Extract the raw pointers
    double *d_A = thrust::raw_pointer_cast(A.data());
    double *d_w = thrust::raw_pointer_cast(w.data());
    double *d_C = thrust::raw_pointer_cast(C.data());
    
    // Compute C = A*diag(w)*A'
    computeWeightedCov(D, N, d_A, d_w, d_C);
}

#endif

