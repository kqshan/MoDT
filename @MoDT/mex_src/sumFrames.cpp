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
 * This requires all the input arguments to be double-precision.
 *
 * Kevin Shan, 2016-06-06
 *============================================================================*/

#include <algorithm>
#include <vector>
#include "mex.h"
#include "blas.h"

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
    if (!mxIsDouble(mx_Y) || (D==0) || (N==0))
        mexErrMsgIdAndTxt(errId, "Y must be a [D x N] array of doubles");
    double *Y = mxGetPr(mx_Y);
    // wzu = input 1
    mxArray const *mx_wzu = prhs[1];
    ptrdiff_t K = mxGetN(mx_wzu);
    if (!mxIsDouble(mx_wzu) || (K==0) || (mxGetM(mx_wzu)!=N))
        mexErrMsgIdAndTxt(errId, "wzu must be an [N x K] array of doubles");
    double *wzu = mxGetPr(mx_wzu);
    // f_spklim = input 2
    mxArray const *mx_fsLim = prhs[2];
    ptrdiff_t T = mxGetM(mx_fsLim);
    if (!mxIsDouble(mx_fsLim) || (T==0) || (mxGetN(mx_fsLim)!=2))
        mexErrMsgIdAndTxt(errId, "f_spklim must be a [T x 2] array of doubles");
    double const *fsLim = mxGetPr(mx_fsLim);
    
    // Validate the indices and determine the max spike count per frame
    ptrdiff_t maxCount = 0;
    for (ptrdiff_t t=0; t<T; t++) {
        // Get the first and last index of the frame
        ptrdiff_t n1 = ((ptrdiff_t) fsLim[t]) - 1;
        ptrdiff_t n2 = ((ptrdiff_t) fsLim[t+T]) - 1;
        // Check that the indices are valid
        if ((n1<0) || (n2<-1) || (n1>N) || (n2>=N) || (n2-n1<-1))
            mexErrMsgIdAndTxt(errId, "Invalid frame spike limits");
        if ((t>0) & (n1 != fsLim[t-1+T]))
            mexErrMsgIdAndTxt(errId, "Non-consecutive frame spike limits");
        // Get the difference
        ptrdiff_t count = n2 - n1 + 1;
        if (count > maxCount)
            maxCount = count;
    }
    
    // Allocate memory for the outputs (initally filled with zeroes)
    size_t dims[] = {(size_t)D, (size_t)K, (size_t)T};
    mxArray *mx_wzuY = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
    double *wzuY = mxGetPr(mx_wzuY);
    mxArray *mx_sumwzu = mxCreateDoubleMatrix(K, T, mxREAL);
    double *sumwzu = mxGetPr(mx_sumwzu);
    
    // Vector of ones so we can sum the weights using dgemv
    std::vector<double> ones(maxCount);
    std::fill(ones.begin(), ones.end(), 1.0);
    
    // Variables used by the BLAS routines
    char trans_N = 'N';     // Do no transpose matrix
    char trans_T = 'T';     // Transpose matrix
    double alpha = 1;       // Scaling on Y*wzu'  and wzu*ones
    double beta = 0;        // Scaling on wzuY    and sum_wzu
    ptrdiff_t incr = 1;     // Vector increment of one
    
    // For loop over time frames
    for (ptrdiff_t t=0; t<T; t++) {
        // Get the first spike index and spike count for this time frame
        ptrdiff_t n1 = ((ptrdiff_t) fsLim[t]) - 1;
        ptrdiff_t M = ((ptrdiff_t) fsLim[t+T]) - n1;
        if (M <= 0) continue;
        // Call dgemm for wzuY = alpha*Y*wzu + beta*wzuY
        dgemm(&trans_N, &trans_N, &D, &K, &M, 
                &alpha, Y+(n1*D), &D, wzu+n1, &N, 
                &beta, wzuY+(t*D*K), &D);
        // Call dgemv for sum_wzu = alpha*wzu'*ones + beta*sum_wzu
        dgemv(&trans_T, &M, &K, &alpha, wzu+n1, &N, 
                ones.data(), &incr, &beta, sumwzu+(t*K), &incr);
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
