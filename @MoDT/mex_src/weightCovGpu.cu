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
 * A and weights can be either single- or double-precision, but they both must
 * be the same type. The output is returned in that type as well.
 * 
 * When D < 32, this optimizes the data access by breaking A into chunks, using
 * a custom CUDA kernel to compute the weighted covariance independently for
 * each chunk, then reducing the result across chunks.
 * When D >= 32, this custom kernel doesn't work (needs > 1024 threads/block),
 * so we produce a scaled copy of A (scaled by sqrt(w)) and then call dgemm.
 *
 * Kevin Shan
 * 2017-04-25  Add single-precision support
 * 2016-06-14  Initial version
 *============================================================================*/

#ifdef MATLAB_MEX_FILE
    #include "mex.h"
    #include "gpu/mxGPUArray.h"
#else
    #include <unistd.h>
#endif

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <thrust/device_vector.h>

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <cmath>


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
int const READ_BUFF_BYTES = 256;
template <typename numeric_t>
__global__ void calcWeightedCov(
        int D, int N, numeric_t const *A, numeric_t const *w, numeric_t *C)
{
    // Dynamically-allocated shared memory, length = (D+1)*32
    extern __shared__ __align__(sizeof(int4)) unsigned char shared_mem[];
    numeric_t *S = reinterpret_cast<numeric_t*>(shared_mem);
    int const READ_BUFF_SZ = READ_BUFF_BYTES/sizeof(numeric_t);
    numeric_t *w_buff = S;                // [32 x 1] buffer for weights
    numeric_t *A_buff = S + READ_BUFF_SZ; // [D x 32] buffer for data matrix
    
    // Figure out what this thread is supposed to be doing
    int const tIdx = threadIdx.x;
    int const nThreads = blockDim.x;
    // Loading phase: the first 32 threads will load w, rest will do A
    int const readOps_w = READ_BUFF_SZ; // Gonna assume nThreads > READ_BUFF_SZ
    int const tIdx_A = (tIdx - readOps_w + nThreads) % nThreads;
    int const readOps_A = D * READ_BUFF_SZ * sizeof(numeric_t)/sizeof(int4);
    // Compute phase: this thread will compute C(ii,jj)
    int ii = -1, jj = -1;
    if (tIdx < D*(D+1)/2) {
        ii = tIdx % D; jj = tIdx / D;
        if (jj > ii) {
            ii = D-1 - ii; jj = D - jj;
        }
    }
    // And we will store the result in this local register
    numeric_t result_acc = 0;
    
    // Realign the inputs with the block offset
    int blockStart = READ_BUFF_SZ * blockIdx.x;
    A += D*blockStart;
    w += blockStart;
    N -= blockStart;
    C += D*D*blockIdx.x;
    // Main loop: grid-stride loop over complete buffers
    int blockStride = READ_BUFF_SZ * gridDim.x;
    while (N >= READ_BUFF_SZ) {
        // Load w from memory
        if (tIdx < readOps_w)
            w_buff[tIdx] = w[tIdx];
        // Load A from memory, 16 bytes at a time
        int4 const *src = reinterpret_cast<int4 const*>(A);
        int4 *tgt = reinterpret_cast<int4*>(A_buff);
        for (int idx = tIdx_A; idx < readOps_A; idx += nThreads)
            tgt[idx] = src[idx];
        // Wait for load to complete
        __syncthreads();
        
        // Compute
        if (ii >= 0) {
            // Compute the result for this buffer
            numeric_t local_acc = 0;
            for (int kk=0; kk<READ_BUFF_SZ; kk++)
                local_acc += A_buff[ii+D*kk] * A_buff[jj+D*kk] * w_buff[kk];
            // Add it to the running total
            result_acc += local_acc;
        }
        // Wait for computation to finish
        __syncthreads();
        
        // Increment the pointers and counters
        A += D*blockStride;
        w += blockStride;
        N -= blockStride;
    }
    // Deal with remaining data
    if (N > 0) {
        // Load remaining w and A
        if (tIdx < N)
            w_buff[tIdx] = w[tIdx];
        for (int idx = tIdx_A; idx < D*N; idx += nThreads)
            A_buff[idx] = A[idx];
        __syncthreads();
        // Compute
        if (ii >= 0) {
            numeric_t local_acc = 0;
            for (int kk=0; kk<N; kk++)
                local_acc += A_buff[ii+D*kk] * A_buff[jj+D*kk] * w_buff[kk];
            result_acc += local_acc;
        }
    }
    
    // Write output to global memory
    if (ii >= 0) {
        C[ii+D*jj] = result_acc;
        // Write the other half as well
        if (ii != jj)
            C[jj+D*ii] = result_acc;
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
template <typename numeric_t>
__global__ void sumColumns( int const nRows, int const nCols, 
        numeric_t const * __restrict__ X, numeric_t *Y )
{
    // Check if this thread falls in range
    int const row = threadIdx.x + blockIdx.x*blockDim.x;
    if (row < nRows) {
        // For loop to sum over columns
        numeric_t result_acc = 0;
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
 * B can have more columns than A, in which case they will be filled with zero.
 *
 * Inputs:
 *    D     Number of rows in A, B
 *    N_A   Number of columns in A
 *    N_B   Number of columns in B
 *    A     [D x N] data matrix
 *    w     [N] weight vector
 * Outputs:
 *    B     [D x N] scaled data matrix
 */
template <typename numeric_t>
__global__ void sqrtScaledCopy(int D, int N_A, int N_B,
        numeric_t const *A, numeric_t const *w, numeric_t *B)
{
    // Dynamically-allocated shared memory
    extern __shared__ __align__(sizeof(int4)) unsigned char shared_mem[];
    numeric_t *S = reinterpret_cast<numeric_t*>(shared_mem);
    // Notational convenience for some dimensions
    int tIdx = threadIdx.x;
    int P = blockDim.x;
    // Realign everything with the block offset
    int blockStart = blockIdx.x * P;
    A += D*blockStart;
    w += blockStart;
    B += D*blockStart;
    N_A -= blockStart; // # of columns in A remaining
    N_B -= blockStart; // # of columns in B remaining
    // Loop over the [N] dimension
    int blockStride = P * gridDim.x;
    while (N_B > 0) {
        // Read and sqrt() the weight vector
        if (tIdx < N_A)
            S[tIdx] = sqrt(w[tIdx]);
        __syncthreads();
        // Read, scale, and write the data in these columns
        int dataCount = D * min(P,N_B);
        int dataStride = blockDim.x;
        for (int dataIdx=tIdx; dataIdx<dataCount; dataIdx+=dataStride) {
            int dataCol = dataIdx/D;
            if (dataCol < N_A)
                B[dataIdx] = S[dataCol] * A[dataIdx];
            else
                B[dataIdx] = 0;
        }
        __syncthreads();
        // Increment the pointers and counters
        A += D*blockStride;
        w += blockStride;
        B += D*blockStride;
        N_A -= blockStride;
        N_B -= blockStride;
    }
}


/* Overload a single function for both single and double-precision data
 */
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
cublasStatus_t gemmStridedBatched( 
        cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb,
        int m, int n, int k, const double *alpha, 
        const double *A, int lda, ptrdiff_t strideA, 
        const double *B, int ldb, ptrdiff_t strideB, const double *beta,
        double *C, int ldc, ptrdiff_t strideC, int batchCount )
{   return cublasDgemmStridedBatched(handle,ta,tb,m,n,k,alpha,
            A,lda,strideA,B,ldb,strideB,beta,C,ldc,strideC,batchCount); }
cublasStatus_t gemmStridedBatched(
        cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb,
        int m, int n, int k, const float *alpha, 
        const float *A, int lda, ptrdiff_t strideA, 
        const float *B, int ldb, ptrdiff_t strideB, const float *beta, 
        float *C, int ldc, ptrdiff_t strideC, int batchCount )
{   return cublasSgemmStridedBatched(handle,ta,tb,m,n,k,alpha,
            A,lda,strideA,B,ldb,strideB,beta,C,ldc,strideC,batchCount); }
cublasStatus_t gemv(
        cublasHandle_t handle, cublasOperation_t trans, int m, int n,
        const double *alpha, const double *A, int lda, 
        const double *x, int incx,
        const double *beta, double *y, int incy)
{   return cublasDgemv(handle,trans,m,n,alpha,A,lda,x,incx,beta,y,incy); }
cublasStatus_t gemv(
        cublasHandle_t handle, cublasOperation_t trans, int m, int n,
        const float *alpha, const float *A, int lda, 
        const float *x, int incx,
        const float *beta, float *y, int incy)
{   return cublasSgemv(handle,trans,m,n,alpha,A,lda,x,incx,beta,y,incy); }


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
struct CudaDeleter {
    void operator() (void *d_ptr) { cudaFree(d_ptr); }
};

/* Define a "mexErrMsgIdAndTxt" function for the non-MATLAB case
 */
#ifndef MATLAB_MEX_FILE
void mexErrMsgIdAndTxt(char const *errId, char const *errMsg)
{
    std::cout << errId << std::endl << errMsg << std::endl;
}
#endif


int nextPow2(double x) { return static_cast<int>( exp2(ceil(log2( x ))) ); }
template <typename T>
int nextPow2(T x) { return nextPow2(static_cast<double>(x)); }

/* Optimize the number of batches to use
 * 
 * Inputs:
 *    D         Number of feature space dimensions
 *    N         Number of spikes
 *    M         Adjustment factor (M=1 for GEMM)
 * Outputs:
 *    nBatches  Number of batches to perform this in
 *    batchSize Size of each batch (#spikes)
 */
void optimizeBatches(int D, int N, int M, int &nBatches, int &batchSize)
{
    // batchSize = nBatches = sqrt(N) is best for numerical accuracy.
    // Round this to the next power of 2
    batchSize = nextPow2( M*sqrt((N+M-1)/M) );
    // Enforce some minimum sizes
    batchSize = std::max(batchSize, 256); // Computational efficiency
    batchSize = std::max(batchSize, M*32); // Roundoff error with adjustment
    batchSize = std::max(batchSize, nextPow2(10*D)); // Memory overhead
    // Determine the number of batches
    nBatches = (N + batchSize-1) / batchSize;
    // Some special cases
    if (nBatches < 4) {
        // If it's a small number of batches, just forget it
        nBatches = 1;
        batchSize = N;
    } else if (nBatches*batchSize > 1.125*N) {
        // Use smaller batches if there would be a lot of inflation
        batchSize /= 2;
        nBatches = (N + batchSize-1) / batchSize;
    }
}


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
template <typename numeric_t>
void computeWeightedCov(int D, int N, 
        numeric_t const * d_A, numeric_t const * d_w, numeric_t * d_C)
{
    char const * const cudaErrId = "MoDT:weightCovGpu:cudaError";
    cudaError_t cudaStat;
    // Get some device info
    int deviceNo;
    cudaGetDevice(&deviceNo);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceNo);
    // Decide on the batching and whether to use cuBLAS or our own kernel
    bool use_blas = (D > 44);
    int nBatches, batchSize;
    if (use_blas)
        optimizeBatches(D, N, 1, nBatches, batchSize);
    else
        optimizeBatches(D, N, READ_BUFF_BYTES/sizeof(*d_A), nBatches, batchSize);
    // Allocate memory for batches, if necessary
    numeric_t *d_CBatched;
    std::unique_ptr<numeric_t,CudaDeleter> cleanup_CBatched;
    if (nBatches > 1) {
        cudaStat = cudaMalloc((void**)&d_CBatched, 
                static_cast<ptrdiff_t>(D)*D*nBatches*sizeof(*d_A) );
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, "Failed to allocate CUDA memory for batch output");
        cleanup_CBatched = std::unique_ptr<numeric_t,CudaDeleter>{d_CBatched};
    } else {
        // No batch necessary
        d_CBatched = d_C;
    }
    // Compute the covariance
    if (use_blas) {
        // Initialize cuBLAS
        cublasStatus_t stat;
        cublasHandle_t handle;
        stat = cublasCreate(&handle);
        cublasHandleCleanupWrapper cleanup_handle(handle);
        if (stat != CUBLAS_STATUS_SUCCESS)
            mexErrMsgIdAndTxt(cudaErrId, "Unable to initialize cuBLAS context");
        // Allocate memory for the scaled copy
        numeric_t *d_X;
        int N_padded = nBatches * batchSize;
        cudaError_t cudaStat = cudaMalloc((void**)&d_X, 
                static_cast<ptrdiff_t>(D)*N_padded*sizeof(*d_A) );
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, "Failed to allocate CUDA memory for scaled copy");
        std::unique_ptr<numeric_t,CudaDeleter> cleanup_X(d_X);
        // Perform the scaled copy
        int tPerBlk = 256;
        int numBlocks = std::min(128, (N_padded+tPerBlk-1)/tPerBlk);
        int sharedMemPerBlock = tPerBlk * sizeof(*d_A);
        sqrtScaledCopy<<<numBlocks,tPerBlk,sharedMemPerBlock>>>
                (D, N, N_padded, d_A, d_w, d_X);
        // C = X*X'
        numeric_t const alpha = 1; // Scaling applied to X*X'
        numeric_t const beta = 0;  // Scaling applied to C    
        if (nBatches > 1) {
            stat = gemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                    D, D, batchSize,
                    &alpha, d_X, D, D*batchSize, d_X, D, D*batchSize,
                    &beta, d_CBatched, D, D*D, nBatches);
        } else {
            stat = gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    D, D, N, &alpha, d_X, D, d_X, D, &beta, d_C, D);
        }
        if (stat != CUBLAS_STATUS_SUCCESS)
            mexErrMsgIdAndTxt(cudaErrId, "cuBLAS error during GEMM");
    } else {
        // Launch our CUDA kernel
        int warpsPerBlock = std::max(2, ((D*(D+1))/2 + 31)/32);
        int tPerBlk = 32 * warpsPerBlock;
        int sharedMemPerBlock = (D+1)*READ_BUFF_BYTES;
        int numBlocks = nBatches;
        calcWeightedCov<<<numBlocks,tPerBlk,sharedMemPerBlock>>>
                (D, N, d_A, d_w, d_CBatched);
    }
    // Sum across batches
    if (nBatches > 1) {
        int tPerBlk = 256;
        int numBlocks = (D*D + tPerBlk-1) / tPerBlk;
        sumColumns<<<numBlocks,tPerBlk>>>(D*D, nBatches, d_CBatched, d_C);
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
    mxClassID numericType = mxGPUGetClassID(mgpu_A);
    if ((mxGPUGetNumberOfElements(mgpu_A)==0) || 
            (mxGPUGetNumberOfDimensions(mgpu_A)!=2) ||
            !((numericType==mxDOUBLE_CLASS) || (numericType==mxSINGLE_CLASS)) )
        mexErrMsgIdAndTxt(errId, "A must be a 2-D real gpuArray");
    size_t const *dims = mxGPUGetDimensions( mgpu_A );
    int D = dims[0];
    int N = dims[1];
    // weights = input 1
    mxArray const *mx_weights = prhs[1];
    if (!mxIsGPUArray(mx_weights))
        mexErrMsgIdAndTxt(errId, "weights must a gpuArray");
    mxGPUArray const *mgpu_weights = mxGPUCreateFromMxArray( mx_weights );
    mxGPUArrayCleanupWrapper cleanup_weights(mgpu_weights);
    if ((mxGPUGetNumberOfElements(mgpu_weights)==0) || 
            (mxGPUGetNumberOfDimensions(mgpu_weights)!=2) ||
            (mxGPUGetClassID(mgpu_weights)!=numericType))
        mexErrMsgIdAndTxt(errId, "weights must be a 2-D gpuArray of the same type as A");
    dims = mxGPUGetDimensions( mgpu_weights );
    int N_weights = dims[0];
    int K = dims[1];
    if (N_weights != N)
        mexErrMsgIdAndTxt(errId, "weights must be a [N x K] gpuArray");
    // k (weight index) = input 2
    mxArray const *mx_k = prhs[2];
    if (!mxIsScalar(mx_k))
        mexErrMsgIdAndTxt(errId, "k must be a scalar");
    ptrdiff_t weight_index = static_cast<ptrdiff_t>(mxGetScalar(mx_k)) - 1;
    if ((weight_index < 0) || (weight_index >= K))
        mexErrMsgIdAndTxt(errId, "k is out of bounds");
    
    // Allocate memory for the output
    std::vector<size_t> dims_C = {(size_t) D, (size_t) D};
    mxGPUArray *mgpu_C = mxGPUCreateGPUArray(2, dims_C.data(), 
            numericType, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArrayCleanupWrapper cleanup_C(mgpu_C);
    
    // Compute C = A*diag(w)*A'
    switch (numericType) {
        case mxDOUBLE_CLASS:
            computeWeightedCov(D, N, gpuPtr<double>(mgpu_A), 
                    gpuPtr<double>(mgpu_weights) + weight_index*N,
                    gpuPtr<double>(mgpu_C) );
            break;
        case mxSINGLE_CLASS:
            computeWeightedCov(D, N, gpuPtr<float>(mgpu_A), 
                    gpuPtr<float>(mgpu_weights) + weight_index*N,
                    gpuPtr<float>(mgpu_C) );
            break;
    }
    
    // Output 0 = C
    if (nlhs >= 1) {
        // Wrap it in a mxArray
        mxArray *mx_C = mxGPUCreateMxArrayOnGPU( mgpu_C );
        plhs[0] = mx_C;
    }
    
    // Cleanup done by our wrapper classes
}

#else

template <typename numeric_t>
void demo_weightCov(ptrdiff_t D, ptrdiff_t N)
{
    // Create the data
    thrust::device_vector<numeric_t> A(D*N,1);
    thrust::device_vector<numeric_t> w(N,1);
    thrust::device_vector<numeric_t> C(D*D,0);
    // Extract the raw pointers
    numeric_t *d_A = thrust::raw_pointer_cast(A.data());
    numeric_t *d_w = thrust::raw_pointer_cast(w.data());
    numeric_t *d_C = thrust::raw_pointer_cast(C.data());
    
    // Compute C = A*diag(w)*A'
    computeWeightedCov(D, N, d_A, d_w, d_C);
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
    if (use_single) {
        demo_weightCov<float>(D, N);
    } else {
        demo_weightCov<double>(D, N);
    }
}

#endif

