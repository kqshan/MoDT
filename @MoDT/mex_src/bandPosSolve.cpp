/*==============================================================================
 * 
 * Solve A*x = b, where A is a positive definite matrix in banded storage
 *   x = bandPosSolve(A_bands, b)
 * 
 * Returns:
 *   x           [N x m] solution vector
 * Required arguments:
 *   A_bands     [p x N] banded storage of a [N x N] positive definite matrix
 *   b           [N x m] input vector
 * 
 * This calls LAPACK dpbsv, and requires all inputs to be double-precision.
 * This assumes that A is symmetric, and therefore uses only the first 
 * (p+1)/2 rows of A_bands.
 *
 * Kevin Shan, 2016-06-06
 *============================================================================*/

#include "mex.h"
#include "lapack.h"

/* Main entry point into this mex file
 * Inputs and outputs are arrays of mxArray pointers
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Check the inputs
    char const * const errId = "MoDT:bandPosSolve:InvalidInput";
    if (nrhs != 2)
        mexErrMsgIdAndTxt(errId, "bandPosSolve() expects 2 inputs");
    // A_bands = input 0
    mxArray const *mx_AB = prhs[0];
    ptrdiff_t p = mxGetM(mx_AB);
    ptrdiff_t N = mxGetN(mx_AB);
    if (!mxIsDouble(mx_AB) || (N==0))
        mexErrMsgIdAndTxt(errId, "A_bands must be a [p x N] array of doubles");
    if ((p%2==0) || (p>=2*N))
        mexErrMsgIdAndTxt(errId, "A_bands is an invalid # of rows");
    // b = input 1
    mxArray const *mx_b = prhs[1];
    ptrdiff_t m = mxGetN(mx_b);
    if (!mxIsDouble(mx_b) || (mxGetM(mx_b)!=N) || (m==0))
        mexErrMsgIdAndTxt(errId, "b must be an [N x m] array of doubles");
    
    // dpbsv overwrites A_bands and B, so we need to duplicate them
    mxArray *mx_AB_copy = mxDuplicateArray(mx_AB);
    mxArray *mx_b_copy = mxDuplicateArray(mx_b);
    
    // Variables used by the LAPACK routine
    char uplo = 'U';                // Use the upper triangle of A
    ptrdiff_t nSupDiag = (p-1)/2;   // Number of superdiagonals in [bands]
    ptrdiff_t info = 0;             // Status output for dpbsv
    
    // Call dpbsv to compute x = A \ b
    double *AB = mxGetPr(mx_AB_copy);
    double *b = mxGetPr(mx_b_copy);
    dpbsv(&uplo, &N, &nSupDiag, &m, AB, &p, b, &N, &info);
    // Check that it succeeded
    if (info != 0)
        mexErrMsgIdAndTxt(errId, "LAPACK routine dpbsv() exited with error");
    
    // Cleanup
    mxDestroyArray(mx_AB_copy);
    
    // Return mx_b_copy, which now contains x
    if (nlhs >= 1)
        plhs[0] = mx_b_copy;
    else
        mxDestroyArray(mx_b_copy);
}
