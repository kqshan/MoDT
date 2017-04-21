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
 * This function requires all matrices to be double-precision. There may be some
 * issues if X contains more than 2 billion (2^31) elements.
 * 
 * For D <= 28, this uses a custom CUDA kernel that multiplies the matrices and
 * computes the squared norm, without requiring intermediate storage.
 * For D > 28, we use cublasDtrsm and have a custom kernel for the squared norm.
 *
 * Kevin Shan, 2016-06-16
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
#include <utility>
#include <limits>
#include <memory>
#include <vector>

/* Overload the __shfl() intrinsics for double-precision data
 * Damnit, there's undocumented versions of __shfl for double-precision
 * arguments, so I had to name these something else. Also, note that the __shfl
 * intrinsics were only introduced in compute capability 3.0 and higher.
 */
#if __CUDA_ARCH__ >= 300
__device__ double dblShfl(double var, int srcLane, int width=32)
{
    int hi = __shfl( __double2hiint(var), srcLane, width );
    int lo = __shfl( __double2loint(var), srcLane, width );
    return __hiloint2double( hi, lo );
}

__device__ double dblShfl_down(double var, unsigned int delta, int width=32)
{
    int hi = __shfl_down( __double2hiint(var), delta, width );
    int lo = __shfl_down( __double2loint(var), delta, width );
    return __hiloint2double( hi, lo );
}
#endif


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
 * For a D=12 triangular solve, there are 6 [4 x 4] sub-blocks of L, and on each
 * iteration we have 3 loads from global memory (LD), 3 triangular solves (TS), 
 * and 3 matrix multiplications (MM). We also need 6 no-ops for proper
 * synchronization and broadcasting of intermediate results, which brings the
 * memory requirement up to 15 [4 x 8] workspcaes. Here is a diagram (no-ops are
 * labeled with the node #, which is numbered consecutively across rows):
 *
 * LD_0 -> TS_00 -> [ 2 ]
 *                    |
 * LD_1 -> [ 4 ] -> MM_10 -> TS_11 -> [ 7 ]
 *                    |                 |
 * LD_2 -> [ 9 ] -> MM_20 -> [ 11] -> MM_21 -> TS_22 -> [ 14]
 *
 * On each loop iteration, each node performs its specified operation, then
 * shifts its workspace one node to the right. The last no-op node on each row 
 * contains the solution to L\X. Vertical lines indicate dependencies; e.g.
 * MM_10 and MM_20 both read from node 2.
 *
 * Note that each row has a different latency: row 0 has a latency of 2 (i.e.
 * node 2 contains L\X for the sub-block of X loaded 2 cycles ago), whereas
 * row 1 has a latency of 4, etc. When squaring and summing across rows, we need
 * to make sure that we account for this latency:
 *
 * -> TS_00 -> [ 2 ]
 *               |   -> TS_11 -> [ 7 ]
 *               |                 |   -> TS_22 -> [ 14]
 *               |                 |                 |
 *             ACC_1 -> [ 16] -> ACC_2 -> [ 18] -> ACC_3
 * 
 * The accumulators (ACC) compute the squared sum over the 4 rows of the input
 * node's [4 x 8] memory block. Hence this last row of ACC nodes requires only
 * [8 x 1] memory per node. The final ACC node also writes to global memory.
 */
__device__ void tsn_Load(int x, int y, int tIdx, double * W, 
        int D, int N, int xOffset, int yOffset, 
        double const * __restrict__ X) {
    // Load a [4 x 8] block of X into W, padding with zeroes
    int xRead = x + xOffset;
    int yRead = y + yOffset;
    W[tIdx] = ((xRead<D)&&(yRead<N)) ? X[xRead+D*yRead] : 0;
}
__device__ void tsn_TriSolve(int x, int y, int tIdx, double volatile *W, 
        double const *L) {
#if __CUDA_ARCH__ >= 300
    // Triangular solve with unit-diagonal [4 x 4] L and [4 x 8] W
    double val = W[tIdx];
    double coeff = L[tIdx%16];
    for (int ii=0; ii<4-1; ii++) {
        // if (x>ii) W[x,y] -= L[x,ii] * W[ii,y];
        double adjustment = dblShfl(coeff,x+ii*4,16) * dblShfl(val,ii,4);
        if (x>ii) val -= adjustment;
    }
    // Apply scaling (produced when we unit-diagonalized L)
    W[tIdx] = val * dblShfl(coeff,5*x,16); // W[x,y] *= L[x,x];
#else
    for (int ii=0; ii<4-1; ii++)
        if (x>ii) W[tIdx] -= L[x+ii*4] * W[ii+y*4];
    W[tIdx] *= L[5*x];
#endif
}
__device__ void tsn_MatMult(int x, int y, int tIdx, double *W, 
        double const *L, double const *Y) {
    // Matrix multiplication: W -= L * Y, where L is [4x4] and W,Y are [4x8]
    double w_val = W[tIdx];
#if __CUDA_ARCH__ >= 300
    double y_val = Y[tIdx];
    double coeff = L[tIdx%16];
    for (int ii=0; ii<4; ii++) {
        // W[x,y] -= L[x,ii] * Y[ii,y];
        w_val -= dblShfl(coeff,x+ii*4,16) * dblShfl(y_val,ii,4);
    }
#else
    for (int ii=0; ii<4; ii++)
        w_val -= L[x+ii*4] * Y[ii+y*4];
#endif
    W[tIdx] = w_val;
}
__device__ void tsn_Acc(int x, int y, int tIdx, double const *W, 
        double *acc) {
    // Compute the squared norm for each column of the [4x8] matrix W
    double val = W[tIdx];
    val *= val;
#if __CUDA_ARCH__ >= 300
    val += dblShfl_down(val,2,4);
    val += dblShfl_down(val,1,4);
    // Write to output
    if (x==0) acc[y] += val;
#else
    if (x==0) acc[y] += val;
    if (x==1) acc[y] += val;
    if (x==2) acc[y] += val;
    if (x==3) acc[y] += val;
#endif
}

/* There is a pretty intricate scheduling of what each warp is supposed to do in
 * each loop iteration. Rather than have each thread block figure this out on
 * its own, we'll plan it out on the host (CPU) in this KernelPlan class.
 */
struct __align__(16) WarpTask {
    enum Op : int {Load,TriSolve,MatMult,Acc} operation; // Operation to perform
    int wkspIdx;        // Workspace index
    int opArg1;         // (optional) second argument for this operation
    int opArg2;         // (optional) third argument for this operation
};
class KernelPlan {
public:
    int R;                          // ceil(D/4)
    int M;                          // R*(R+1)/2 = # of [4 x 4] sub-blocks of L
    std::vector<double> L_blocks;   // [4 x 4 x M] with the sub-blocks of L
    int nWksp;                      // # of [4 x 8] workspaces
    int nAcc;                       // # of [8 x 1] norm accumulator vectors
    std::vector<int> nextWksp;      // [nWksp] circularly-linked lists
    int nTask;                      // # of tasks to perform every iteration
    int latency;                    // Overall computation latency
    int sharedMemPerBlock;          // Shared memory required (Bytes)
    std::vector<WarpTask> taskList; // [nTask] list of WarpTasks to perform
    KernelPlan(int, double const *);// Constructor
};
/* KernelPlan constructor */
KernelPlan::KernelPlan(int D, double const *L)
{
    // Basic dimensions
    R = (D+3)/4; // ceil(D/4)
    M = (R*(R+1))/2;
    nWksp = 2*M+R;
    nAcc = 2*R-1;
    nTask = M+2*R;
    latency = 2*R;
    // Copy L into blocks, ordered [00, 10, 11, 20, 21, 22, ... ]
    L_blocks.resize(4*4*M);
    std::fill(L_blocks.begin(), L_blocks.end(), 0);
    int blockIdx = 0;
    for (int block_i=0; block_i<R; block_i++) {
        // Sub-diagonal blocks are a simple copy
        for (int block_j=0; block_j<block_i; block_j++) {
            for (int idx=0; idx<16; idx++) {
                int tgt_idx = idx + blockIdx*16;
                int src_i = (idx%4) + block_i*4;
                int src_j = (idx/4) + block_j*4;
                if ((src_i<D)&&(src_j<D)) L_blocks[tgt_idx] = L[src_i+src_j*D];
            }
            blockIdx++;
        }
        // For blocks on the diagonal, scale so that it is unit-diagonal
        for (int jj=0; jj<4; jj++) {
            // Set the diagonal equal to the scaling
            int src_j = jj + block_i*4;
            if (src_j>=D) continue;
            double scaling = 1 / L[src_j+src_j*D];
            L_blocks[jj+jj*4+blockIdx*16] = scaling;
            // Scale the rest of that column
            for (int ii=jj+1; ii<4; ii++) {
                int src_i = ii + block_i*4;
                int tgt_idx = ii + jj*4 + blockIdx*16;
                if (src_i<D) L_blocks[tgt_idx] = scaling*L[src_i+src_j*D];
            }
        }
        blockIdx++;
    }
    // Set up the circular buffers for the [4 x 8] workspaces
    /* If a node is operating on workspace ii this loop iteration, it should
     * operate on workspace nextWksp[ii] the next loop iteration. */
    nextWksp.resize(nWksp);
    for (int ii=0; ii<R; ii++) {
        int rowStart = ii*(ii+2);
        int rowEnd = rowStart + 2*ii + 2;
        nextWksp[rowStart] = rowEnd;
        for (int idx=rowStart; idx < rowEnd; idx++)
            nextWksp[idx+1] = idx;
    }
    // Initialize the WarpTasks
    taskList.resize(nTask);
    int taskIdx = 0;
    for (int ii=0; ii<R; ii++) {                // Loads from global memory
        taskList[taskIdx].operation = WarpTask::Load;
        taskList[taskIdx].wkspIdx = ii*(ii+2);      // First wksp in the row
        taskList[taskIdx].opArg1 = 4*ii;            // xOffset argument
        taskIdx++;
    }
    for (int ii=0; ii<R; ii++) {                // Squared-norm accumulators
        taskList[taskIdx].operation = WarpTask::Acc;
        taskList[taskIdx].wkspIdx = ii*(ii+4)+2;    // Last wksp in the row
        taskList[taskIdx].opArg1 = 2*ii;            // normAccum index
        taskList[taskIdx].opArg2 = ii;              // row # this operates on
        taskIdx++;
    }
    for (int ii=0; ii<R; ii++) {                // Triangular solves
        taskList[taskIdx].operation = WarpTask::TriSolve;
        taskList[taskIdx].wkspIdx = ii*(ii+4)+1;    // 2nd-to-last wksp in row
        taskList[taskIdx].opArg1 = (ii*(ii+3))/2;   // L_block index for (ii,ii)
        taskIdx++;
    }
    for (int jj=0; jj<R; jj++) {                // Matrix multiplication
        int yIdx = jj*(jj+4)+2;                     // Last wksp in row jj
        for (int ii=jj+1; ii<R; ii++) {
            taskList[taskIdx].operation = WarpTask::MatMult;
            taskList[taskIdx].wkspIdx 
                    = ii*(ii+2) + 2*jj + 2;         // wkspIdx for W argument
            taskList[taskIdx].opArg1 
                    = (ii*(ii+1))/2 + jj;           // L_block index for (ii,jj)
            taskList[taskIdx].opArg2 = yIdx;        // wkspIdx for Y argument
            taskIdx++;
        }
    }
    // Shared memory required
    sharedMemPerBlock = 
            nWksp*4*8*sizeof(double)        // Workspaces
            + nAcc*8*sizeof(double)         // normAccums
            + M*4*4*sizeof(double)          // L_blocks
            + nTask*sizeof(WarpTask)        // taskList
            + nWksp*sizeof(int);            // nextWksp
}
/* Ultimately, the kernel will need to access this KernelPlan object. So here
 * are some wrappers etc to move the KernelPlan onto the GPU device.
 */
struct KernelPlanOnDevice {
    int R;
    int M;
    double *L_blocks;   // Pointer to device memory
    int nWksp;
    int nAcc;
    int *nextWksp;      // Pointer to device memory
    int nTask;
    int latency;
    WarpTask *taskList; // Pointer to device memory
};
class KernelPlanDeviceWrapper {
public:
    KernelPlanDeviceWrapper(KernelPlan const&);
    KernelPlanOnDevice *get_ptr(void) { return d_kpod.get(); }
private:
    std::unique_ptr<KernelPlanOnDevice,CudaDeleter> d_kpod;
    std::unique_ptr<double,CudaDeleter> L_blocks;
    std::unique_ptr<int,CudaDeleter> nextWksp;
    std::unique_ptr<WarpTask,CudaDeleter> taskList;
};

/* Constructor/destructor for wrapper class that copies KernelPlan to device */
KernelPlanDeviceWrapper::KernelPlanDeviceWrapper(KernelPlan const& src)
{
    // Start by making a KernelPlanOnDevice struct in host memory
    KernelPlanOnDevice kpod;
    // Copy fields
    kpod.R = src.R;
    kpod.M = src.M;
    kpod.nWksp = src.nWksp;
    kpod.nAcc = src.nAcc;
    kpod.nTask = src.nTask;
    kpod.latency = src.latency;
    // Deep copy of arrays
    size_t nBytes;
    nBytes = 4*4*src.M*sizeof(*kpod.L_blocks);
    cudaMalloc((void**)&kpod.L_blocks, nBytes);
    L_blocks.reset(kpod.L_blocks);
    cudaMemcpy(kpod.L_blocks,src.L_blocks.data(),nBytes,cudaMemcpyHostToDevice);
    nBytes = src.nWksp*sizeof(*kpod.nextWksp);
    cudaMalloc((void**)&kpod.nextWksp, nBytes);
    nextWksp.reset(kpod.nextWksp);
    cudaMemcpy(kpod.nextWksp,src.nextWksp.data(),nBytes,cudaMemcpyHostToDevice);
    nBytes = src.nTask*sizeof(*kpod.taskList);
    cudaMalloc((void**)&kpod.taskList, nBytes);
    taskList.reset(kpod.taskList);
    cudaMemcpy(kpod.taskList,src.taskList.data(),nBytes,cudaMemcpyHostToDevice);
    // Make a device copy of the KernelPlanOnDevice struct
    nBytes = sizeof(kpod);
    KernelPlanOnDevice *d_kpod_copy;
    cudaMalloc((void**)&d_kpod_copy, nBytes);
    d_kpod.reset(d_kpod_copy);
    cudaMemcpy(d_kpod.get(), &kpod, nBytes, cudaMemcpyHostToDevice);
}
/* Finally, the kernel will need to copy this KernelPlan into its shared memory.
 * This is not super trivial, so here is a separate function to do that.
 */
__device__ void tsn_LoadKernelPlan(KernelPlanOnDevice const *plan, int *S, 
        dim3 const &threadIdx, dim3 const &blockDim, 
        double * &workspaces, double * &normAccums, double * &L_blocks, 
        int * &nextWksp, WarpTask * &taskList)
{
    int const M = plan->M;
    int const nWksp = plan->nWksp;
    int const nAcc = plan->nAcc;
    int const nTask = plan->nTask;
    // Lay out memory in order of descending alignment
    int offset = 0;
    workspaces = (double*)&S[offset];   // 256 bytes
    offset += nWksp * 4*8*sizeof(*workspaces)/sizeof(*S);
    L_blocks = (double*)&S[offset];     // 128 bytes
    offset += M * 4*4*sizeof(*L_blocks)/sizeof(*S);
    normAccums = (double*)&S[offset];   // 64 bytes
    offset += nAcc * 8*sizeof(*normAccums)/sizeof(*S);
    taskList = (WarpTask*)&S[offset];   // 16 bytes
    offset += nTask * sizeof(WarpTask)/sizeof(*S);
    nextWksp = (int*)&S[offset];        // 4 bytes
    // Initialize L_blocks, nextWksp, and taskList
    int const tIdx = threadIdx.x + 32*threadIdx.y;
    int const stride = 32*blockDim.y;
    for (int ii=tIdx; ii<4*4*M; ii+=stride)
        L_blocks[ii] = plan->L_blocks[ii];
    for (int ii=tIdx; ii<nWksp; ii+=stride)
        nextWksp[ii] = plan->nextWksp[ii];
    for (int ii=tIdx; ii<nTask; ii+=stride)
        taskList[ii] = plan->taskList[ii];
}


/* triSolveNorm - CUDA kernel to perform triangular matrix solve and the compute
 *                the squared 2-norm over the columns of the result.
 *
 * This expects:
 *   - 1-D block grid of any size
 *   - Thread block of size [32 x P]
 *   - Shared memory 
 *
 * This kernel uses a serial approach to processing the data:
 *   1. Data is loaded into shared memory in [D x 8] chunks
 *   2. A triangular solve is performed in-place, operating on one 4x4 sub-block
 *      of L at a time. This takes 2*R loop iterations due to the dependencies.
 *   3. The squared 2-norm is computed for each column.
 * This is pipelined so that we are loading new data on each loop iteration in
 * the kernel. Within a loop iteration, there are M+2*R independent tasks that
 * are distributed to the P warps of the thread block.
 *
 * Inputs:
 *   D          # of rows in X, and dimension of L
 *   N          # of columns in X
 *   X          [D x N] data matrix
 *   plan       Pointer to KernelPlanOnDevice struct
 * Outputs:
 *   sqNorm     [N] output vector; y = sum((L\X).^2,1)
 */
__global__ void triSolveNorm( int const D, int const N, 
        double const * __restrict__ X, KernelPlanOnDevice const * plan, 
        double * sqNorm )
{
    extern __shared__ int S[];
    // Arrange the shared memory and load the kernel plan into shared memory
    double *workspaces, *normAccums, *L_blocks;
    int *nextWksp;
    WarpTask *taskList;
    tsn_LoadKernelPlan(plan, S, threadIdx, blockDim, 
            workspaces, normAccums, L_blocks, nextWksp, taskList);
    __syncthreads();
    // Get some other dimensions etc
    int const R = plan->R;
    int const nAcc = plan->nAcc;
    int const nTask = plan->nTask;
    int const latency = plan->latency;
    int const nWarp = blockDim.y;
    // Get the identity of this thread
    int const thread_idx = threadIdx.x;
    int const thread_x = thread_idx % 4;
    int const thread_y = thread_idx / 4;
    int const warp_idx = threadIdx.y;
    // Determine the number of iterations we will need to take
    int readOffset = blockIdx.x * 8;
    int readStride = gridDim.x * 8;
    int nIter = (N-readOffset+readStride-1) / readStride;
    nIter += latency;
    // Grid stride through the data
    for (int iter=0; iter<nIter; iter++) {
        // For loop over the operations per iteration
        for (int taskIdx=warp_idx; taskIdx<nTask; taskIdx+=nWarp) {
            // Perform the specified operation
            WarpTask *task = taskList + taskIdx;
            int wkspIdx = task->wkspIdx;
            double *W = workspaces + wkspIdx*32;
            switch (task->operation) {
                case WarpTask::Load : {
                    int xOffset = task->opArg1;
                    int yOffset = readOffset + iter*readStride;
                    // Load from global memory
                    tsn_Load(thread_x, thread_y, thread_idx, W, 
                            D, N, xOffset, yOffset, X);
                    break;
                }
                case WarpTask::TriSolve : {
                    double const *L = L_blocks + task->opArg1*16;
                    // Triangular solve
                    tsn_TriSolve(thread_x, thread_y, thread_idx, W, L);
                    break;
                }
                case WarpTask::MatMult : {
                    double const *L = L_blocks + task->opArg1*16;
                    int wkspIdx_Y = task->opArg2;
                    double const *Y = workspaces + wkspIdx_Y*32;
                    // Matrix multiplication
                    tsn_MatMult(thread_x, thread_y, thread_idx, W, L, Y);
                    // Increment the Y workspace pointer
                    if (thread_idx==0) task->opArg2 = nextWksp[wkspIdx_Y];
                    break;
                }
                case WarpTask::Acc : {
                    int accIdx = task->opArg1;
                    int rowIdx = task->opArg2;
                    double *acc = normAccums + accIdx*8;
                    // Reset the accumulator
                    if ((rowIdx==0) && (thread_idx<8))
                        acc[thread_idx] = 0;
                    // Accumulate squared norm
                    tsn_Acc(thread_x, thread_y, thread_idx, W, acc);
                    // Write to global memory if applicable
                    if ((rowIdx==R-1) && (iter>=latency)) {
                        int yOffset = readOffset + (iter-latency)*readStride;
                        if ((thread_idx<N-yOffset) && (thread_idx<8))
                            sqNorm[thread_idx+yOffset] = acc[thread_idx];
                    }
                    // Increment the norm accumulator pointer
                    if (thread_idx==0)
                        task->opArg1 = (accIdx==0) ? (nAcc-1) : (accIdx-1);
                    break;
                }
            }
            // Increment the workspace pointer
            if (thread_idx==0) task->wkspIdx = nextWksp[wkspIdx];
        }
        // Wait for all warps to finish before going to the next iteration
        __syncthreads();
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
__global__ void sumSqCols( int const D, int const N, 
        double const * __restrict__ A, double * b )
{
    // Shared memory helps us coalesce the writes
    #if __CUDA_ARCH__ >= 300
    __shared__ double S[sumSqCols_blockDim_y];
    #else
    __shared__ double S[sumSqCols_blockDim_y*33];
    double volatile * S_sum = S + (threadIdx.y+1)*32;
    #endif
    // Some dimension-related constants
    int const linearIdx = threadIdx.x + threadIdx.y*blockDim.x;
    int const P = sumSqCols_blockDim_y;
    int const D_eff = ((D+31)/32) * 32; // Round up to next multiple of 32
    // Grid-stride loop over the columns
    int const yStart = blockIdx.x * P;
    int const yStride = gridDim.x * P;
    for (int readOffset=yStart; readOffset<N; readOffset+=yStride) {
        // Compute the sum over this column
        double running_sum = 0;
        int readIdx_y = threadIdx.y + readOffset;
        if (readIdx_y < N) {
            // For loop over the rows, 32 at a time
            /* No need to synchronize because each y belongs to a single warp */
            for (int readIdx_x=threadIdx.x; readIdx_x<D_eff; readIdx_x+=32) {
                // Read and square the data
                double value;
                value = (readIdx_x<D) ? A[readIdx_x+readIdx_y*D] : 0;
                value *= value;
                // Reduce across rows, noting that they belong to the same warp
                #if __CUDA_ARCH__ >= 300
                for (int shflOffset=16; shflOffset>0; shflOffset/=2)
                    value += dblShfl_down(value, shflOffset);
                #else
                S_sum[threadIdx.x] = value;
                for (int shflOffset=16; shflOffset>0; shflOffset/=2)
                    if (threadIdx.x<shflOffset)
                        S_sum[threadIdx.x] += S_sum[threadIdx.x+shflOffset];
                #endif
                // Note that this value is only valid for threadIdx.x==0
                running_sum += value;
            }
        }
        // Synchronize so that we have a coalesced write
        __syncthreads();
        if (threadIdx.x==0)
            S[threadIdx.y] = running_sum;
        __syncthreads();
        if ((linearIdx < P) && (readOffset+linearIdx<N))
            b[readOffset+linearIdx] = S[linearIdx];
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
void computeMahalDist(size_t D, size_t N, 
        double const *L, double const *d_X, double *d_delta)
{
    char const * const cudaErrId = "MoDT:calcMahalDistGpu:cudaError";
    if (D <= 28) {
        /* We have a kernel that performs a triangular solve and computes the
         * squared norm of each column, and it's optimized for small matrices */
        // I should have thought more carefully about where to use ints...
        if (D*N > std::numeric_limits<int>::max())
            mexErrMsgIdAndTxt("MoDT:calcMahalDistGpu:InvalidInput",
                    "Large arrays (> 2^31 elements) not currently supported");
        // Create an execution plan for the kernel and transfer it to the device
        KernelPlan plan(D, L);
        KernelPlanDeviceWrapper kpdWrapper(plan);
        // Determine how many warps we can allocate per block
        int sharedMemPerBlock = plan.sharedMemPerBlock;
        int blocksPerMP = std::min(16, (48*1024)/sharedMemPerBlock);
        int warpsPerBlock = std::min(32, 51/blocksPerMP);
        dim3 threadsPerBlock(32, warpsPerBlock);
        // Determine how many blocks to launch
        int colsPerBlock = 64 * plan.latency;
        int numBlocks = std::min(8*blocksPerMP, (int)(N/colsPerBlock)+1);
        // Launch our kernel
        triSolveNorm<<<numBlocks,threadsPerBlock,sharedMemPerBlock>>>
                (D, N, d_X, kpdWrapper.get_ptr(), d_delta);
        // Wait for it to finish before we let kpdWrapper go out of scope
        cudaDeviceSynchronize();
    } 
    else {
        /* For larger D, our kernel runs out of shared memory, so use cuBLAS */
        cublasStatus_t cublasStat;
        cudaError_t cudaStat;
        cublasHandle_t handle;
        // Initialize
        cublasStat = cublasCreate(&handle);
        cublasHandleCleanupWrapper cleanup_handle(handle);
        if (cublasStat != CUBLAS_STATUS_SUCCESS)
            mexErrMsgIdAndTxt(cudaErrId, "Unable to initialize cuBLAS context");
        // Move L to the device
        double *d_L;
        cudaStat = cudaMalloc((void**)&d_L, D*D*sizeof(*L));
        std::unique_ptr<double,CudaDeleter> cleanup_L(d_L);
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, "Failed to allocate CUDA memory");
        cublasStat = cublasSetMatrix(D, D, sizeof(*L), L, D, d_L, D);
        if (cublasStat != CUBLAS_STATUS_SUCCESS)
            mexErrMsgIdAndTxt(cudaErrId, "cuBLAS error copying L to GPU");
        // Copy X because it'll be overwritten by TRSM
        double *d_X_copy;
        cudaStat = cudaMalloc((void**)&d_X_copy, D*N*sizeof(*d_X));
        std::unique_ptr<double,CudaDeleter> cleanup_X_copy(d_X_copy);
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, "Failed to allocate CUDA memory");
        cudaStat = cudaMemcpy(d_X_copy, d_X, D*N*sizeof(*d_X), 
                cudaMemcpyDeviceToDevice);
        if (cudaStat != cudaSuccess)
            mexErrMsgIdAndTxt(cudaErrId, "Failed to make copy of X");
        // Call TRSM
        double const alpha = 1;     // Scaling applied to X
        cublasStat = cublasDtrsm( handle, CUBLAS_SIDE_LEFT, 
                CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
                D, N, &alpha, d_L, D, d_X_copy, D );
        if (cublasStat != CUBLAS_STATUS_SUCCESS)
            mexErrMsgIdAndTxt(cudaErrId, "cuBLAS error during TRSM");
        // Call our kernel for computing the squared norm
        dim3 threadsPerBlock(32, sumSqCols_blockDim_y);
        int numBlocks = std::min(64, (int)(N/sumSqCols_blockDim_y)+1);
        sumSqCols<<<numBlocks,threadsPerBlock>>>(D, N, d_X_copy, d_delta);
    }
}


#ifdef MATLAB_MEX_FILE
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
    size_t D = mxGetM(mx_L);
    if ((D==0) || (D!=mxGetN(mx_L)) || !mxIsDouble(mx_L))
        mexErrMsgIdAndTxt(errId, "L must be a square double-precision matrix");
    double const *L = mxGetPr(mx_L);
    // X = input 1
    mxArray const *mx_X = prhs[1];
    if (!mxIsGPUArray(mx_X))
        mexErrMsgIdAndTxt(errId, "X must a gpuArray");
    mxGPUArray const *mgpu_X = mxGPUCreateFromMxArray( mx_X );
    mxGPUArrayCleanupWrapper cleanup_X(mgpu_X);
    if ((mxGPUGetNumberOfElements(mgpu_X)==0) || 
            (mxGPUGetNumberOfDimensions(mgpu_X)!=2) ||
            (mxGPUGetClassID(mgpu_X)!=mxDOUBLE_CLASS))
        mexErrMsgIdAndTxt(errId, "X must a 2-D double-precision array");
    size_t const *dims = mxGPUGetDimensions( mgpu_X );
    if (dims[0] != D)
        mexErrMsgIdAndTxt(errId, "X must be a [D x N] gpuArray");
    size_t N = dims[1];
    double const *d_X = (double const *) mxGPUGetDataReadOnly( mgpu_X );
    
    // Allocate memory for the output
    size_t dims_delta[2] = {N, 1};
    mxGPUArray *mgpu_delta = mxGPUCreateGPUArray(2, dims_delta, 
            mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArrayCleanupWrapper cleanup_delta(mgpu_delta);
    double *d_delta = (double *) mxGPUGetData(mgpu_delta);
    
    // Compute delta = sum((L\X).^2,1)'
    computeMahalDist(D, N, L, d_X, d_delta);
    
    // Output 0 = delta
    if (nlhs >= 1) {
        // Wrap it in a mxArray
        mxArray *mx_delta = mxGPUCreateMxArrayOnGPU( mgpu_delta );
        plhs[0] = mx_delta;
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
    thrust::device_vector<double> X(D*N,1);
    thrust::device_vector<double> delta(N,1);
    std::vector<double> L(D*D,0);
    for (int i=0; i<D; i++)
        L[i+D*i] = 1;    
    // Extract the raw pointers
    double *d_X = thrust::raw_pointer_cast(X.data());
    double *d_delta = thrust::raw_pointer_cast(delta.data());
    double *h_L = L.data();
    
    // Compute delta = sum((L\X).^2,1)'
    computeMahalDist(D, N, h_L, d_X, d_delta);
}

#endif

