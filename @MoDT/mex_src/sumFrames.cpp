/*==============================================================================
 * Compute the weighted sum of data points for each time frame
 *   [wzuY, sum_wzu] = sumFrames(Y, wzu, f_spklim)
 * 
 * Returns:
 *   wzuY        [D x K x T] weighted sums of data points in each time frame
 *   sum_wzu     [K x T] sums of weights in each time frame
 * Required arguments:
 *   Y           [D x N] data points (D dimensions x N points)
 *   wzu         [N x K] weights for each (cluster) x (data point)
 *   f_spklim    [T x 2] [first,last] data index (1..N) in each time frame
 * 
 * This performs the following for loop:
 * for t = 1:T
 *     n1 = f_spklim(t,1); n2 = f_spklim(t,2);
 *     sum_wzu(:,t) = sum(wzu(n1:n2,:),1)';
 *     wzuY(:,:,t) = Y(:,n1:n2) * wzu(n1:n2,:);
 * end
 * 
 * Y and wzu must both be the same datatype (either single- or double-precision)
 *
 * Kevin Shan, 2016-06-06
 *============================================================================*/

#include <algorithm>
#include <vector>
#include "mex.h"
#include "blas.h"

/* Overload a single function for both single and double-precision data
 */
void gemv(char *trans, ptrdiff_t *m, ptrdiff_t *n, double *alpha, 
          double const *a, ptrdiff_t *lda, double const *x, ptrdiff_t *incx,
          double *beta, double *y, ptrdiff_t *incy)
{   dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);  }
void gemv(char *trans, ptrdiff_t *m, ptrdiff_t *n, float *alpha, 
          float const *a, ptrdiff_t *lda, float const *x, ptrdiff_t *incx,
          float *beta, float *y, ptrdiff_t *incy)
{   sgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);  }
void gemm(char *transa, char *transb, ptrdiff_t *m, ptrdiff_t *n, ptrdiff_t *k,
          double *alpha, double const *a, ptrdiff_t *lda, 
          double const *b, ptrdiff_t *ldb,
          double *beta, double *c, ptrdiff_t *ldc)
{   dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
void gemm(char *transa, char *transb, ptrdiff_t *m, ptrdiff_t *n, ptrdiff_t *k,
          float *alpha, float const *a, ptrdiff_t *lda, 
          float const *b, ptrdiff_t *ldb,
          float *beta, float *c, ptrdiff_t *ldc)
{   sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }

/* A single-precision analogue to mxGetPr
 */
float* mxGetFloatPr( mxArray *mx_A )
{   return static_cast<float*>(mxGetData(mx_A));    }
float const* mxGetFloatPr( mxArray const *mx_A )
{   return static_cast<float const*>(mxGetData(mx_A));    }


/* Main routine for computing the weighted sum over data frames
 *
 * Inputs:
 *    D         Number of feature space dimensions
 *    N         Number of spikes
 *    K         Number of clusters
 *    T         Number of time frames
 *    Y         [D x N] data matrix
 *    wzu       [N x K] weights for each cluster x spike
 *    fsLim0    [T x 2] [first,last] data index (0-indexed) in each frame
 * Outputs:
 *    wzuY      [D x K x T] weighted sums for each frame (on GPU device)
 *    sumwzu    [K x T] sums of weights in each frame (on GPU device)
 */
template <typename numeric_t>
void computeFrameSums(ptrdiff_t D, ptrdiff_t N, ptrdiff_t K, ptrdiff_t T, 
        numeric_t const *Y, numeric_t const *wzu, 
        std::vector<ptrdiff_t> const &fsLim0, 
        numeric_t *wzuY, numeric_t *sumwzu)
{
    // Validate the fsLim indices and get the max # spikes per frame
    char const * const errId = "MoDT:sumFramesGpu:InvalidInput";
    ptrdiff_t maxCount = 0;
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
        maxCount = std::max(maxCount, n2-n1+1);
    }
    
    // Vector of ones so we can sum the weights using dgemv
    std::vector<numeric_t> ones(maxCount, 1.0);
    
    // Variables used by the BLAS routines
    char trans_N = 'N';     // Do no transpose matrix
    char trans_T = 'T';     // Transpose matrix
    numeric_t alpha = 1;  // Scaling on Y*wzu' and wzu*ones
    numeric_t beta = 0;   // Scaling on wzuY and sum_wzu
    ptrdiff_t incr = 1;     // Vector increment of one
    
    // For loop over time frames
    for (int t=0; t<T; t++) {
        // Get the first spike index and spike count for this time frame
        ptrdiff_t n1 = ((ptrdiff_t) fsLim0[t]);
        ptrdiff_t M = ((ptrdiff_t) fsLim0[t+T]) - n1 + 1;
        if (M <= 0) continue;
        // Call GEMM for wzuY = alpha*Y*wzu + beta*wzuY
        gemm(&trans_N, &trans_N, &D, &K, &M, 
                &alpha, Y+(n1*D), &D, wzu+n1, &N, 
                &beta, wzuY+(t*D*K), &D);
        // Call GEMV for sum_wzu = alpha*wzu'*ones + beta*sum_wzu
        gemv(&trans_T, &M, &K, &alpha, wzu+n1, &N, 
                ones.data(), &incr, &beta, sumwzu+(t*K), &incr);
    }
}



/* Main entry point into this mex file
 * Inputs and outputs are arrays of mxArray pointers
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Check the inputs
    char const * const errId = "MoDT:sumFrames:InvalidInput";
    if (nrhs != 3)
        mexErrMsgIdAndTxt(errId, "sumFrames() expects 3 inputs");
    // Y = input 0
    mxArray const *mx_Y = prhs[0];
    ptrdiff_t D = mxGetM(mx_Y);
    ptrdiff_t N = mxGetN(mx_Y);
    mxClassID numericType = mxGetClassID(mx_Y);
    if (!((numericType==mxDOUBLE_CLASS) || (numericType==mxSINGLE_CLASS)) || (D==0) || (N==0))
        mexErrMsgIdAndTxt(errId, "Y must be a [D x N] array of real numbers");
    // wzu = input 1
    mxArray const *mx_wzu = prhs[1];
    ptrdiff_t K = mxGetN(mx_wzu);
    if ((mxGetClassID(mx_wzu)!=numericType) || (K==0) || (mxGetM(mx_wzu)!=N))
        mexErrMsgIdAndTxt(errId, "wzu must be an [N x K] array of the same type as Y");
    // f_spklim = input 2
    mxArray const *mx_fsLim = prhs[2];
    ptrdiff_t T = mxGetM(mx_fsLim);
    if (!mxIsDouble(mx_fsLim) || (T==0) || (mxGetN(mx_fsLim)!=2))
        mexErrMsgIdAndTxt(errId, "f_spklim must be a [T x 2] array of doubles");
    double const *fsLim = mxGetPr(mx_fsLim);
    
    // Copy the fsLim indices to a vector of 0-indexed integers
    std::vector<ptrdiff_t> fsLim0(T*2);
    std::transform(fsLim, fsLim+2*T, fsLim0.begin(), 
            [](double matlabIdx){ return static_cast<ptrdiff_t>(matlabIdx)-1; });

    // Allocate memory for the outputs (initally filled with zeroes)
    std::vector<size_t> dims_wzuY = {(size_t) D, (size_t) K, (size_t) T};
    mxArray *mx_wzuY = mxCreateNumericArray(3, dims_wzuY.data(), numericType, mxREAL);
    mxArray *mx_sumwzu = mxCreateNumericMatrix(K, T, numericType, mxREAL);
    
    // Sum across time frames
    switch (numericType) {
        case mxDOUBLE_CLASS:
            computeFrameSums(D, N, K, T, mxGetPr(mx_Y), mxGetPr(mx_wzu), 
                             fsLim0, mxGetPr(mx_wzuY), mxGetPr(mx_sumwzu) );
            break;
        case mxSINGLE_CLASS:
            computeFrameSums(D, N, K, T, mxGetFloatPr(mx_Y), mxGetFloatPr(mx_wzu), 
                             fsLim0, mxGetFloatPr(mx_wzuY), mxGetFloatPr(mx_sumwzu) );
            break;
    }
    
    // Output 0 = wzuY
    if (nlhs >= 1)
        plhs[0] = mx_wzuY;
    else
        mxDestroyArray(mx_wzuY);
    // Output 1 = sum_wzu
    if (nlhs >= 2)
        plhs[1] = mx_sumwzu;
    else
        mxDestroyArray(mx_sumwzu);
}
