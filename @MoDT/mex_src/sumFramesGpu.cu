/*==============================================================================
 * Compute the weighted sum of data points for each time frame
 *   [wzuY, sum_wzu] = sumFramesGpu(Y, wzu, f_spklim)
 * 
 * Returns:
 *   wzuY        [D x K x T] weighted sums of data points in each time frame
 *   sum_wzu     [K x T] sums of weights in each time frame
 * Required arguments:
 *   Y           [D x N] data points (D dimensions x N points) (gpuArray)
 *   wzu         [N x K] weights for each (cluster) x (data point) (gpuArray)
 *   f_spklim    [T x 2] [first,last] data index (1..N) in each time frame
 * 
 * Y and wzu can be single- or double-precision, but they both must be the same
 * type. The output will be returned in that type as well.
 * 
 * This performs the following for loop:
 * for t = 1:T
 *     n1 = f_spklim(t,1); n2 = f_spklim(t,2);
 *     sum_wzu(:,t) = sum(wzu(n1:n2,:),1)';
 *     wzuY(:,:,t) = Y(:,n1:n2) * wzu(n1:n2,:);
 * end
 *
 * For small problems, (D <= 32), this is performed in a custom kernel that
 * reduces the kernel launch overhead. For larger problems, this calls cuBLAS: 
 * GEMM for Y*wzu, and GEMV with a vector of ones for sum(wzu).
 * 
 * Kevin Shan
 * 2017-04-25  Add single-precision support
 * 2016-08-29  Initial version
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
#include <algorithm>
#include <utility>
#include <memory>

/* CUDA kernel to perform matrix multiplication in chunks
 *
 * This must be called with the following:
 *    [grid_x x grid_y] block grid; let K_eff = ceil(K/grid_x)
 *    [32 x nWarps] thread block w/ nWarps >= ceil((D*K_eff + K_eff) / 32)
 *    Shared memory for READ_BUFF_SZ*(D+K_eff) doubles and ceil(T/grid_y) ints
 * 
 * Inputs
 *    D         Spike dimensionality (rows of <spikes>)
 *    N         Number of spikes (columns of <spikes> and rows of <weights>)
 *    K         Number of clusters (columns of <weights>)
 *    T         Number of time frames (rows of <spkLims>)
 *    spikes    [D x N] data matrix
 *    weights   [N x K] weight matrix
 *    spkLims   [T x 2] first and last spike (0..N-1) in each time frame
 * Outputs
 *    spkSum    [D x K x T] matrix product of spikes*weights in each time frame
 *    wSum      [K x T] sum of the weights in each time frame
 */
int const READ_BUFF_SZ = 16;
template <typename numeric_t>
__global__ void sumFrames( int const D, int const N, int const K, int const T, 
        numeric_t const * __restrict__ spikes, numeric_t const * __restrict__ weights,
        int const * __restrict__ spkLims, numeric_t *spkSum, numeric_t *wSum )
{
    // Declare our dynamically-allocated shared memory
    extern __shared__ int S[];
    
    // Determine which clusters this block is responsible for
    int const kPerBlk = (K+gridDim.x-1) / gridDim.x;        // ceil(K/gridDim.x)
    int const kOffset = kPerBlk * blockIdx.x;
    int const kCount = min(kPerBlk, K-kOffset);
    if (kCount <= 0) return;
    // Determine which time frames this block is responsible for
    int const framesPerBlk = (T+gridDim.y-1) / gridDim.y;   // ceil(T/gridDim.y)
    int const frameOffset = framesPerBlk * blockIdx.y;
    int const frameCount = min(framesPerBlk, T-frameOffset);
    if (frameCount <= 0) return;
    // And the spike range
    int const spkOffset = spkLims[frameOffset];
    // Shift the pointers to account for these offsets
    spikes += D*static_cast<ptrdiff_t>(spkOffset);
    weights += spkOffset + static_cast<ptrdiff_t>(N)*kOffset;
    spkLims += frameOffset;
    spkSum += D*(kOffset + K*frameOffset);
    wSum += kOffset + K*frameOffset;
    // Lay out our shared memory
    int offset = 0;
    numeric_t *spikeBuff = reinterpret_cast<numeric_t*> (S + offset);
    offset += D * READ_BUFF_SZ * sizeof(*spikeBuff)/sizeof(*S);
    numeric_t *weightBuff = reinterpret_cast<numeric_t*> (S + offset);
    offset += kPerBlk * READ_BUFF_SZ * sizeof(*weightBuff)/sizeof(*S);
    int *spkCounts = reinterpret_cast<int*> (S + offset);
    // Copy the spike counts into shared memory
    int const tIdx = threadIdx.x + threadIdx.y*blockDim.x;
    int const tPerBlk = blockDim.x * blockDim.y;
    for (int ii=tIdx; ii<frameCount; ii+=tPerBlk)
        spkCounts[ii] = (int) (spkLims[ii+T] - spkLims[ii] + 1);
    
    // Determine what this tread is responsible for computing
    bool threadHasValidCompute = false;
    numeric_t *spkCompBuff, *wCompBuff, *outputTgt;
    int spkCompStride, outputStride;
    numeric_t one = 1;
    if (tIdx < D*kCount) {
        threadHasValidCompute = true;
        // Thread computes spkSum(d,k,t) = spikes(d,:) * weights(:,k)
        int d = tIdx % D;
        int k = tIdx / D;
        spkCompBuff = spikeBuff + d;
        spkCompStride = D;
        wCompBuff = weightBuff + READ_BUFF_SZ*k;
        outputTgt = spkSum + d + D*k;
        outputStride = D*K;
    } else if (tIdx - D*kCount < kCount) {
        threadHasValidCompute = true;
        // Thread computes wSum(k,t) = sum(weights(:,k))
        int k = tIdx - D*kCount;
        spkCompBuff = &one;
        spkCompStride = 0;
        wCompBuff = weightBuff + READ_BUFF_SZ*k;
        outputTgt = wSum + k;
        outputStride = K;
    }
    // Determine what to do when loading
    int const loadOps_spk = D * READ_BUFF_SZ;
    int const loadOps_w = READ_BUFF_SZ * kCount;
    int const tIdx_2 = (tIdx + 32*((loadOps_spk+31)/32)) % tPerBlk;
    
    // Main loop over time frames
    for (int frameIdx=0; frameIdx<frameCount; frameIdx++) {
        __syncthreads();
        int nSpkRemain = spkCounts[frameIdx];
        numeric_t result_acc = 0;
        // About 98% of the time, we can load+compute a whole buffer at a time
        while (nSpkRemain >= READ_BUFF_SZ) {
            // Load from spikes
            for (int ldIdx = tIdx; ldIdx < loadOps_spk; ldIdx += tPerBlk)
                spikeBuff[ldIdx] = spikes[ldIdx];
            spikes += D*READ_BUFF_SZ;
            // Load from weights
            for (int ldIdx = tIdx_2; ldIdx < loadOps_w; ldIdx += tPerBlk) {
                int load_n = ldIdx % READ_BUFF_SZ;
                int load_k = ldIdx / READ_BUFF_SZ;
                weightBuff[ldIdx] = weights[load_n + N*load_k];
            }
            weights += READ_BUFF_SZ;
            // Compute spikes*weights and sum(weights)
            __syncthreads();
            if (threadHasValidCompute) {
                numeric_t local_acc = 0;
                for (int ii=0; ii<READ_BUFF_SZ; ii++)
                    local_acc += spkCompBuff[spkCompStride*ii] * wCompBuff[ii];
                result_acc += local_acc;
            }
            __syncthreads();
            // Advance the buffer
            nSpkRemain -= READ_BUFF_SZ;
        }
        // Load the remaining spikes
        for (int ldIdx = tIdx; ldIdx < D*nSpkRemain; ldIdx += tPerBlk)
            spikeBuff[ldIdx] = spikes[ldIdx];
        spikes += D*nSpkRemain;
        // Load the remaining weights
        for (int ldIdx = tIdx_2; ldIdx < nSpkRemain*kCount; ldIdx += tPerBlk) {
            int load_n = ldIdx % nSpkRemain;
            int load_k = ldIdx / nSpkRemain;
            weightBuff[load_n+READ_BUFF_SZ*load_k] = weights[load_n + N*load_k];
        }
        weights += nSpkRemain;
        // Compute and write
        __syncthreads();
        if (threadHasValidCompute) {
            // Compute on the partial buffer
            numeric_t local_acc = 0;
            for (int ii=0; ii<nSpkRemain; ii++)
                local_acc += spkCompBuff[spkCompStride*ii] * wCompBuff[ii];
            result_acc += local_acc;
            // Write to output
            outputTgt[outputStride*frameIdx] = result_acc;
        }
        // Go on to the next time frame
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
cublasStatus_t gemv(
        cublasHandle_t handle, cublasOperation_t trans, int m, int n,
        const double *alpha, const double *A, int lda, const double *x, int incx,
        const double *beta, double *y, int incy)
{   return cublasDgemv(handle,trans,m,n,alpha,A,lda,x,incx,beta,y,incy); }
cublasStatus_t gemv(
        cublasHandle_t handle, cublasOperation_t trans, int m, int n,
        const float *alpha, const float *A, int lda, const float *x, int incx,
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


/* Main routine for computing the weighted sum over data frames
 *
 * Inputs:
 *    D         Number of feature space dimensions
 *    N         Number of spikes
 *    K         Number of clusters
 *    T         Number of time frames
 *    d_Y       [D x N] data matrix (on GPU device)
 *    d_wzu     [N x K] weights for each cluster x spike (on GPU device)
 *    fsLim0    [T x 2] [first,last] data index (0-indexed) in each frame
 * Outputs:
 *    d_wzuY    [D x K x T] weighted sums for each frame (on GPU device)
 *    d_sumwzu  [K x T] sums of weights in each frame (on GPU device)
 */
template <typename numeric_t>
void computeFrameSums(int D, int N, int K, int T, 
        numeric_t const *d_Y, numeric_t const *d_wzu, std::vector<int> &fsLim0, 
        numeric_t *d_wzuY, numeric_t *d_sumwzu)
{
    // Validate the fsLim indices and get the max # spikes per frame
    char const * const errId = "MoDT:sumFramesGpu:InvalidInput";
    int maxCount = 0;
    int last_n2 = 0;
    for (int t=0; t<T; t++) {
        int n1 = fsLim0[t];
        int n2 = fsLim0[t+T];
        // Check that the indices are valid
        if ((n1<0) || (n2<-1) || (n1>N) || (n2>=N) || (n2-n1 < -1))
            mexErrMsgIdAndTxt(errId, "Invalid frame spike limits");
        if ((t>0) & (n1 != last_n2+1))
            mexErrMsgIdAndTxt(errId, "Non-consecutive frame spike limits");
        last_n2 = n2;
        // Get the difference
        maxCount = std::max(maxCount, n2-n1+1);
    }
    
    // Sum across time frames
    cudaError_t cudaStat;
    char const * const cudaErrId = "MoDT:sumFramesGpu:cudaError";
    if (D <= 32) {
        /* For small problems, we have a custom kernel that will perform the for
         * loop over time frames, reducing the kernel launch overhead. */
        
        // Copy the fsLim indices to the GPU
        int *d_fsLim;
        cudaStat = cudaMalloc((void**)&d_fsLim, T*2*sizeof(*d_fsLim));
        std::unique_ptr<int,CudaDeleter> cleanup_fsLim(d_fsLim);
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, "Failed to allocate CUDA memory");
        cudaStat = cudaMemcpyAsync(d_fsLim, fsLim0.data(), T*2*sizeof(*d_fsLim), 
                cudaMemcpyHostToDevice);
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, "cuBLAS error copying to GPU");
        
        // Optimize the kernel parameters
        int deviceNo;
        cudaGetDevice(&deviceNo);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceNo);
        // Design for the worst-case scenario in terms of # frames per block
        int maxT_eff = 64; // I decided this
        // Figure out how many clusters we can do and still have 2 blocks per MP
        int maxThreads = prop.maxThreadsPerMultiProcessor;
        int maxK_thread = (maxThreads/2) / (D+1);
        int maxMem = prop.sharedMemPerMultiprocessor;
        int maxK_mem = ((maxMem/2) - maxT_eff*sizeof(*d_fsLim))
                       / (READ_BUFF_SZ*sizeof(numeric_t)) - D;
        int maxK = std::min(maxK_thread, maxK_mem);
        // If we can't do all of the clusters at once, try to spread them evenly
        int K_eff, grid_x;
        if (maxK >= K) {
            grid_x = 1;
            K_eff = K;
        } else {
            grid_x = (K + maxK-1)/maxK;
            K_eff = (K + grid_x-1)/grid_x;
        }
        // This determines the threads per block and the memory usage
        int nWarps = (D*K_eff + K_eff + 31)/32;
        dim3 threadsPerBlock(32, nWarps);
        int memPerBlock = (D+K_eff)*READ_BUFF_SZ*sizeof(numeric_t) + 
                maxT_eff*sizeof(*d_fsLim);
        // Figure out how many blocks in the grid
        int blocksPerMP = std::min(maxThreads/(64*nWarps), maxMem/memPerBlock);
        int grid_y = 2 * prop.multiProcessorCount * blocksPerMP;
        grid_y = std::max(grid_y, (T+maxT_eff-1)/maxT_eff);
        grid_y = std::min(grid_y, T);
        dim3 blocksPerGrid(grid_x, grid_y);
        
        // Launch the kernel
        sumFrames<<<blocksPerGrid, threadsPerBlock, memPerBlock>>>
                (D, N, K, T, d_Y, d_wzu, d_fsLim, d_wzuY, d_sumwzu);
        
    } else {
        /* For larger problems, we turn to cuBLAS */
        
        // Initialize the cuBLAS context
        cublasHandle_t handle;
        cublasStatus_t cublasStat;
        cublasStat = cublasCreate(&handle);
        cublasHandleCleanupWrapper cleanup_handle(handle);
        if (cublasStat != CUBLAS_STATUS_SUCCESS)
            mexErrMsgIdAndTxt(cudaErrId, "Unable to initialize cuBLAS context");
        // Create a vector of all ones and copy it to the GPU
        std::vector<numeric_t> ones(maxCount, 1.0);
        numeric_t *d_ones;
        cudaStat = cudaMalloc((void**)&d_ones, maxCount*sizeof(*d_ones));
        std::unique_ptr<numeric_t,CudaDeleter> cleanup_ones(d_ones);
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, "Failed to allocate CUDA memory");
        cublasStat = cublasSetMatrix(maxCount, 1, sizeof(*d_ones), 
                ones.data(), maxCount, d_ones, maxCount);
        if (cublasStat != CUBLAS_STATUS_SUCCESS)
            mexErrMsgIdAndTxt(cudaErrId, "cuBLAS error copying to GPU");
        // Constants passed by reference to the BLAS routine
        numeric_t const alpha = 1.0;
        numeric_t const beta = 0.0;
        
        // For loop over time frames
        for (int t=0; t<T; t++) {
            // Get the first spike index and spike count for this time frame
            int n1 = fsLim0[t];
            int spkCount = (int) (fsLim0[t+T] - fsLim0[t] + 1);
            if (spkCount <= 0) continue;
            // Call cublasDgemm to get wzuY(:,:,t) = Y(:,n1:n2) * wzu(n1:n2,:)
            cublasStat = gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    D, K, spkCount,
                    &alpha, d_Y+(n1*D),D, d_wzu+n1,N, 
                    &beta, d_wzuY+(t*D*K),D );
            if (cublasStat != CUBLAS_STATUS_SUCCESS)
                mexErrMsgIdAndTxt(cudaErrId, "cuBLAS error calling GEMM");
            // Call cublasDgemv to get sum_wzu(:,t) = sum(wzu(n1:n2,:),1)'
            cublasStat = gemv(handle, CUBLAS_OP_T, spkCount, K, 
                    &alpha, d_wzu+n1,N, d_ones,1, &beta, d_sumwzu+(t*K),1 );
            if (cublasStat != CUBLAS_STATUS_SUCCESS)
                mexErrMsgIdAndTxt(cudaErrId, "cuBLAS error calling GEMV");
            // Go on to the next time frame
        }
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
    char const * const errId = "MoDT:sumFramesGpu:InvalidInput";
    if (nrhs != 3)
        mexErrMsgIdAndTxt(errId, "This function requires 3 inputs");
    // Y (data) = input 0
    mxArray const *mx_Y = prhs[0];
    if (!mxIsGPUArray(mx_Y))
        mexErrMsgIdAndTxt(errId, "Y must be a gpuArray");
    mxGPUArray const *mgpu_Y = mxGPUCreateFromMxArray( mx_Y );
    mxGPUArrayCleanupWrapper cleanup_Y(mgpu_Y);
    mxClassID numericType = mxGPUGetClassID(mgpu_Y);
    if ((mxGPUGetNumberOfElements(mgpu_Y)==0) || 
            (mxGPUGetNumberOfDimensions(mgpu_Y)!=2) ||
            !((numericType==mxDOUBLE_CLASS) || (numericType==mxSINGLE_CLASS)))
        mexErrMsgIdAndTxt(errId, "Y must a 2-D real gpuArray");
    size_t const *dims = mxGPUGetDimensions( mgpu_Y );
    int D = dims[0];
    int N = dims[1];
    // wzu (weights) = input 1
    mxArray const *mx_wzu = prhs[1];
    if (!mxIsGPUArray(mx_wzu))
        mexErrMsgIdAndTxt(errId, "wzu must a gpuArray");
    mxGPUArray const *mgpu_wzu = mxGPUCreateFromMxArray( mx_wzu );
    mxGPUArrayCleanupWrapper cleanup_wzu(mgpu_wzu);
    if ((mxGPUGetNumberOfElements(mgpu_wzu)==0) || 
            (mxGPUGetNumberOfDimensions(mgpu_wzu)!=2) ||
            (mxGPUGetClassID(mgpu_wzu)!=numericType))
        mexErrMsgIdAndTxt(errId, "wzu must a 2-D gpuArray of the same type as Y");
    dims = mxGPUGetDimensions( mgpu_wzu );
    int N_wzu = dims[0];
    int K = dims[1];
    if (N_wzu != N)
        mexErrMsgIdAndTxt(errId, "wzu must be a [N x K] gpuArray");
    // fsLim (frame spike limits) = input 2
    mxArray const *mx_fsLim = prhs[2];
    int T = mxGetM(mx_fsLim);
    if (!mxIsDouble(mx_fsLim) || (T==0) || (mxGetN(mx_fsLim)!=2))
        mexErrMsgIdAndTxt(errId, "f_spklim must be a [T x 2] array of doubles");
    double const *fsLim = mxGetPr(mx_fsLim);
    
    // Copy the fsLim indices to a vector of 0-indexed integers
    std::vector<int> fsLim0(T*2);
    std::transform(fsLim, fsLim+2*T, fsLim0.begin(), 
            [](double matlabIdx){ return ((int) matlabIdx)-1; });
    
    // Allocate memory for the outputs
    std::vector<size_t> dims_wzuY = {(size_t)D, (size_t)K, (size_t)T};
    mxGPUArray *mgpu_wzuY = mxGPUCreateGPUArray( 3, dims_wzuY.data(), 
            numericType, mxREAL, MX_GPU_INITIALIZE_VALUES );
    mxGPUArrayCleanupWrapper cleanup_wzuY(mgpu_wzuY);
    std::vector<size_t> dims_sumwzu = {(size_t)K, (size_t)T};
    mxGPUArray *mgpu_sumwzu = mxGPUCreateGPUArray( 2, dims_sumwzu.data(),
            numericType, mxREAL, MX_GPU_INITIALIZE_VALUES );
    mxGPUArrayCleanupWrapper cleanup_sumwzu(mgpu_sumwzu);
    
    // Sum across time frames
    switch (numericType) {
        case mxDOUBLE_CLASS:
            computeFrameSums(D, N, K, T, 
                    gpuPtr<double>(mgpu_Y), gpuPtr<double>(mgpu_wzu), fsLim0,
                    gpuPtr<double>(mgpu_wzuY), gpuPtr<double>(mgpu_sumwzu) );
            break;
        case mxSINGLE_CLASS:
            computeFrameSums(D, N, K, T, 
                    gpuPtr<float>(mgpu_Y), gpuPtr<float>(mgpu_wzu), fsLim0,
                    gpuPtr<float>(mgpu_wzuY), gpuPtr<float>(mgpu_sumwzu) );
            break;
    }
    
    // Output 0 = wzuY
    if (nlhs >= 1)
        plhs[0] = mxGPUCreateMxArrayOnGPU( mgpu_wzuY );
    // Output 1 = sum_wzu
    if (nlhs >= 2)
        plhs[1] = mxGPUCreateMxArrayOnGPU( mgpu_sumwzu );
}

#else

template <typename numeric_t>
void demo_sumFrames(int D, int N, int K, int T)
{
    // Create the data
    thrust::device_vector<numeric_t> Y(D*N,1);
    thrust::device_vector<numeric_t> wzu(N*K,1);
    std::vector<int> fsLim(T*2);
    for (int t=0; t<T; t++) {
        fsLim[t]   = (t*N)/T;         // fsLim[t,0] = first in frame
        fsLim[t+T] = ((t+1)*N)/T - 1; // fsLim[t,1] = last in frame
    }
    thrust::device_vector<numeric_t> wzuY(D*K*T,1);
    thrust::device_vector<numeric_t> sumwzu(K*T,1);
    // Extract the raw pointers
    numeric_t *d_Y = thrust::raw_pointer_cast(Y.data());
    numeric_t *d_wzu = thrust::raw_pointer_cast(wzu.data());
    numeric_t *d_wzuY = thrust::raw_pointer_cast(wzuY.data());
    numeric_t *d_sumwzu = thrust::raw_pointer_cast(sumwzu.data());
    
    // Sum across time frames
    computeFrameSums(D, N, K, T, d_Y, d_wzu, fsLim, d_wzuY, d_sumwzu);
}    

/* Main entry point if this is compiled externally (i.e. not as a MEX file)
 * This sets up and runs a simple example program and is suitable for benchmarking
 */
int main(int argc, char* argv[])
{
    // Define the sizes
    int D=12, N=500000, K=25, T=100;
    bool use_single = false;
    int c;
    while ( (c = getopt(argc,argv,"D:N:K:T:s")) != -1 ) {
        switch (c) {
            case 'D': D = std::atoi(optarg); break; 
            case 'N': N = std::atoi(optarg); break; 
            case 'K': K = std::atoi(optarg); break; 
            case 'T': T = std::atoi(optarg); break; 
            case 's': use_single = true;    break;
        }
    }
    // Call the appropriate demo type
    if (use_single) {
        demo_sumFrames<float>(D, N, K, T);
    } else {
        demo_sumFrames<double>(D, N, K, T);
    }
}

#endif

