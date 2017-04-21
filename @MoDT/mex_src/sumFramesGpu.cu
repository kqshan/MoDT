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
 * This function requires all input arguments to be double-precision. Outputs
 * will be gpuArrays.
 * 
 * This performs the following for loop:
 * for t = 1:T
 *     n1 = f_spklim(t,1); n2 = f_spklim(t,2);
 *     sum_wzu(:,t) = sum(wzu(n1:n2,:),1)';
 *     wzuY(:,:,t) = Y(:,n1:n2) * wzu(n1:n2,:);
 * end
 *
 * For small problems, (D < 32), this is performed in a custom kernel that
 * reduces the kernel launch overhead. For larger problems, this calls cuBLAS: 
 * GEMM for Y*wzu, and GEMV with a vector of ones for sum(wzu).
 * 
 * Kevin Shan, 2016-08-29
 *============================================================================*/

#ifdef MATLAB_MEX_FILE
#include "mex.h"
#include "gpu/mxGPUArray.h"
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
__global__ void sumFrames( int const D, long const N, int const K, int const T, 
        double const * __restrict__ spikes, double const * __restrict__ weights,
        long const * __restrict__ spkLims, double *spkSum, double *wSum )
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
    long const spkOffset = spkLims[frameOffset];
    // Shift the pointers to account for these offsets
    spikes += D*spkOffset;
    weights += spkOffset + N*kOffset;
    spkLims += frameOffset;
    spkSum += D*(kOffset + K*frameOffset);
    wSum += kOffset + K*frameOffset;
    // Lay out our shared memory
    int offset = 0;
    double *spikeBuff = reinterpret_cast<double*> (S + offset);
    offset += D * READ_BUFF_SZ * sizeof(*spikeBuff)/sizeof(*S);
    double *weightBuff = reinterpret_cast<double*> (S + offset);
    offset += kPerBlk * READ_BUFF_SZ * sizeof(*weightBuff)/sizeof(*S);
    int *spkCounts = reinterpret_cast<int*> (S + offset);
    // Copy the spike counts into shared memory
    int const tIdx = threadIdx.x + threadIdx.y*blockDim.x;
    int const tPerBlk = blockDim.x * blockDim.y;
    for (int ii=tIdx; ii<frameCount; ii+=tPerBlk)
        spkCounts[ii] = (int) (spkLims[ii+T] - spkLims[ii] + 1);
    
    // Determine what this tread is responsible for computing
    bool threadHasValidCompute = false;
    double *spkCompBuff, *wCompBuff, *outputTgt;
    int spkCompStride, outputStride;
    double one = 1;
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
        double result_acc = 0;
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
                double local_acc = 0;
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
            double local_acc = 0;
            for (int ii=0; ii<nSpkRemain; ii++)
                local_acc += spkCompBuff[spkCompStride*ii] * wCompBuff[ii];
            result_acc += local_acc;
            // Write to output
            outputTgt[outputStride*frameIdx] = result_acc;
        }
        // Go on to the next time frame
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
void computeFrameSums(ptrdiff_t D, ptrdiff_t N, ptrdiff_t K, ptrdiff_t T, 
        double const *d_Y, double const *d_wzu, std::vector<ptrdiff_t> &fsLim0, 
        double *d_wzuY, double *d_sumwzu)
{
    // Validate the fsLim indices and get the max # spikes per frame
    char const * const errId = "MoDT:sumFramesGpu:InvalidInput";
    int maxCount = 0;
    ptrdiff_t last_n2 = 0;
    for (int t=0; t<T; t++) {
        ptrdiff_t n1 = fsLim0[t];
        ptrdiff_t n2 = fsLim0[t+T];
        // Check that the indices are valid
        if ((n1<0) || (n2<-1) || (n1>N) || (n2>=N) || (n2-n1 < -1))
            mexErrMsgIdAndTxt(errId, "Invalid frame spike limits");
        if ((t>0) & (n1 != last_n2+1))
            mexErrMsgIdAndTxt(errId, "Non-consecutive frame spike limits");
        last_n2 = n2;
        // Get the difference
        maxCount = std::max(maxCount, (int)(n2-n1+1));
    }
    
    // Sum across time frames
    cudaError_t cudaStat;
    char const * const cudaErrId = "MoDT:sumFramesGpu:cudaError";
    if (D < 32) {
        /* For small problems, we have a custom kernel that will perform the for
         * loop over time frames, reducing the kernel launch overhead. */
        
        // Copy the fsLim indices to the GPU
        ptrdiff_t *d_fsLim;
        cudaStat = cudaMalloc((void**)&d_fsLim, T*2*sizeof(d_fsLim));
        std::unique_ptr<ptrdiff_t,CudaDeleter> cleanup_fsLim(d_fsLim);
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, "Failed to allocate CUDA memory");
        cudaStat = cudaMemcpyAsync(d_fsLim, fsLim0.data(), T*2*sizeof(d_fsLim), 
                cudaMemcpyHostToDevice);
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, "cuBLAS error copying to GPU");
        
        // Optimize the kernel parameters
        int deviceNo;
        cudaGetDevice(&deviceNo);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceNo);
        // Figure out how many clusters we can do at once
        int maxK_thread = prop.maxThreadsPerBlock / (D+1);
        int maxK_mem = (prop.sharedMemPerBlock/2 - T*sizeof(d_fsLim)) / 
                (READ_BUFF_SZ*sizeof(d_Y)) - D;
        int maxK = std::min(maxK_thread, maxK_mem);
        int K_eff, grid_x;
        if (maxK < K) {
            // If we can't do them all at once, try to spread them evenly
            grid_x = (K + maxK-1)/maxK;
            K_eff = (K + grid_x-1)/grid_x;
        } else {
            // We can do all the clusters at once
            grid_x = 1;
            K_eff = K;
        }
        // Figure out how many threads per block
        int nWarps = (D*K_eff + K_eff + 31)/32;
        dim3 threadsPerBlock(32, nWarps);
        // Figure out how many blocks in the grid
        int blocksPerDevice = prop.multiProcessorCount * (maxK/K_eff);
        int grid_y = std::min((int) T, 2*blocksPerDevice);
        dim3 blocksPerGrid(grid_x, grid_y);
        // Figure out how much memory per block
        int T_eff = (T + grid_y-1)/grid_y;
        int memPerBlock = (D+K_eff)*READ_BUFF_SZ*sizeof(d_Y) + 
                T_eff*sizeof(d_fsLim);
        
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
        std::vector<double> ones(maxCount);
        std::fill(ones.begin(), ones.end(), 1.0);
        double *d_ones;
        cudaStat = cudaMalloc((void**)&d_ones, maxCount*sizeof(*d_ones));
        std::unique_ptr<double,CudaDeleter> cleanup_ones(d_ones);
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, "Failed to allocate CUDA memory");
        cublasStat = cublasSetMatrix(maxCount, 1, sizeof(*d_ones), 
                ones.data(), maxCount, d_ones, maxCount);
        if (cublasStat != CUBLAS_STATUS_SUCCESS)
            mexErrMsgIdAndTxt(cudaErrId, "cuBLAS error copying to GPU");
        // Constants passed by reference to the BLAS routine
        double const alpha = 1;
        double const beta = 0;
        
        // For loop over time frames
        for (int t=0; t<T; t++) {
            // Get the first spike index and spike count for this time frame
            ptrdiff_t n1 = fsLim0[t];
            int spkCount = (int) (fsLim0[t+T] - fsLim0[t] + 1);
            if (spkCount <= 0) continue;
            // Call cublasDgemm to get wzuY(:,:,t) = Y(:,n1:n2) * wzu(n1:n2,:)
            cublasStat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    D, K, spkCount,
                    &alpha, d_Y+(n1*D),D, d_wzu+n1,N, 
                    &beta, d_wzuY+(t*D*K),D );
            if (cublasStat != CUBLAS_STATUS_SUCCESS)
                mexErrMsgIdAndTxt(cudaErrId, "cuBLAS error calling GEMM");
            // Call cublasDgemv to get sum_wzu(:,t) = sum(wzu(n1:n2,:),1)'
            cublasStat = cublasDgemv(handle, CUBLAS_OP_T, spkCount, K, 
                    &alpha, d_wzu+n1,N, d_ones,1, &beta, d_sumwzu+(t*K),1 );
            if (cublasStat != CUBLAS_STATUS_SUCCESS)
                mexErrMsgIdAndTxt(cudaErrId, "cuBLAS error calling GEMM");
            // Go on to the next time frame
        }
    }
}


#ifdef MATLAB_MEX_FILE
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
    if ((mxGPUGetNumberOfElements(mgpu_Y)==0) || 
            (mxGPUGetNumberOfDimensions(mgpu_Y)!=2) ||
            (mxGPUGetClassID(mgpu_Y)!=mxDOUBLE_CLASS))
        mexErrMsgIdAndTxt(errId, "Y must a 2-D double-precision gpuArray");
    size_t const *dims = mxGPUGetDimensions( mgpu_Y );
    ptrdiff_t D = (ptrdiff_t) dims[0];
    ptrdiff_t N = (ptrdiff_t) dims[1];
    double const *d_Y = (double const *) mxGPUGetDataReadOnly( mgpu_Y );
    // wzu (weights) = input 1
    mxArray const *mx_wzu = prhs[1];
    if (!mxIsGPUArray(mx_wzu))
        mexErrMsgIdAndTxt(errId, "wzu must a gpuArray");
    mxGPUArray const *mgpu_wzu = mxGPUCreateFromMxArray( mx_wzu );
    mxGPUArrayCleanupWrapper cleanup_wzu(mgpu_wzu);
    if ((mxGPUGetNumberOfElements(mgpu_wzu)==0) || 
            (mxGPUGetNumberOfDimensions(mgpu_wzu)!=2) ||
            (mxGPUGetClassID(mgpu_wzu)!=mxDOUBLE_CLASS))
        mexErrMsgIdAndTxt(errId, "wzu must a 2-D double-precision gpuArray");
    dims = mxGPUGetDimensions( mgpu_wzu );
    ptrdiff_t N_wzu = (ptrdiff_t) dims[0];
    ptrdiff_t K = (ptrdiff_t) dims[1];
    if (N_wzu != N)
        mexErrMsgIdAndTxt(errId, "wzu must be a [N x K] gpuArray");
    double const *d_wzu = (double const *) mxGPUGetDataReadOnly( mgpu_wzu );
    // fsLim (frame spike limits) = input 2
    mxArray const *mx_fsLim = prhs[2];
    ptrdiff_t T = mxGetM(mx_fsLim);
    if (!mxIsDouble(mx_fsLim) || (T==0) || (mxGetN(mx_fsLim)!=2))
        mexErrMsgIdAndTxt(errId, "f_spklim must be a [T x 2] array of doubles");
    double const *fsLim = mxGetPr(mx_fsLim);
    
    // Copy the fsLim indices to a vector of 0-indexed integers
    std::vector<ptrdiff_t> fsLim0(T*2);
    std::transform(fsLim, fsLim+2*T, fsLim0.begin(), 
            [](double matlabIdx){ return ((ptrdiff_t) matlabIdx)-1; });
    
    // Allocate memory for the outputs
    std::vector<size_t> dims_wzuY = {(size_t)D, (size_t)K, (size_t)T};
    mxGPUArray *mgpu_wzuY = mxGPUCreateGPUArray( 3, dims_wzuY.data(), 
            mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES );
    mxGPUArrayCleanupWrapper cleanup_wzuY(mgpu_wzuY);
    double *d_wzuY = (double *) mxGPUGetData(mgpu_wzuY);
    std::vector<size_t> dims_sumwzu = {(size_t)K, (size_t)T};
    mxGPUArray *mgpu_sumwzu = mxGPUCreateGPUArray( 2, dims_sumwzu.data(),
            mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES );
    mxGPUArrayCleanupWrapper cleanup_sumwzu(mgpu_sumwzu);
    double *d_sumwzu = (double *) mxGPUGetData(mgpu_sumwzu);
    
    // Sum across time frames
    computeFrameSums(D, N, K, T, d_Y, d_wzu, fsLim0, d_wzuY, d_sumwzu);
    
    // Output 0 = wzuY
    if (nlhs >= 1)
        plhs[0] = mxGPUCreateMxArrayOnGPU( mgpu_wzuY );
    // Output 1 = sum_wzu
    if (nlhs >= 2)
        plhs[1] = mxGPUCreateMxArrayOnGPU( mgpu_sumwzu );
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
    size_t K = (argc > 3) ? (size_t) std::atoi(argv[3]) : 25;
    size_t T = (argc > 4) ? (size_t) std::atoi(argv[4]) : 100;
    // Create the data
    thrust::device_vector<double> Y(D*N,1);
    thrust::device_vector<double> wzu(N*K,1);
    std::vector<ptrdiff_t> fsLim(T*2);
    for (int t=0; t<T; t++) {
        fsLim[t]   = (t*N)/T;         // fsLim[t,0] = first in frame
        fsLim[t+T] = ((t+1)*N)/T - 1; // fsLim[t,1] = last in frame
    }
    thrust::device_vector<double> wzuY(D*K*T,1);
    thrust::device_vector<double> sumwzu(K*T,1);
    // Extract the raw pointers
    double *d_Y = thrust::raw_pointer_cast(Y.data());
    double *d_wzu = thrust::raw_pointer_cast(wzu.data());
    double *d_wzuY = thrust::raw_pointer_cast(wzuY.data());
    double *d_sumwzu = thrust::raw_pointer_cast(sumwzu.data());
    
    // Sum across time frames
    computeFrameSums(D, N, K, T, d_Y, d_wzu, fsLim, d_wzuY, d_sumwzu);
}

#endif

