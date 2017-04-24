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
 * This calls LAPACK dpbsv or spbsv, depending on the input datatype. If either
 * A_bands or b is single-precision, then the result will be single-precision.
 * This assumes that A is symmetric, and therefore uses only the first 
 * (p+1)/2 rows of A_bands.
 *
 * Kevin Shan, 2016-06-06
 *============================================================================*/

#include "mex.h"
#include "lapack.h"
#include <algorithm>

/* Overload a single function for both single and double-precision data
 */
void pbsv(char *uplo, ptrdiff_t *n, ptrdiff_t *kd, ptrdiff_t *nrhs, 
          double *ab, ptrdiff_t *ldab, double *b, ptrdiff_t *ldb, ptrdiff_t *info)
{   dpbsv(uplo, n, kd, nrhs, ab, ldab, b, ldb, info);   }
void pbsv(char *uplo, ptrdiff_t *n, ptrdiff_t *kd, ptrdiff_t *nrhs, 
          float *ab, ptrdiff_t *ldab, float *b, ptrdiff_t *ldb, ptrdiff_t *info)
{   spbsv(uplo, n, kd, nrhs, ab, ldab, b, ldb, info);   }

/* Templated PBSV for both single and double
 * 
 * Performs X = A \ X; A = chol(A,'lower');
 *
 * Inputs:
 *    N     Size of A and #rows of X
 *    p     #diagonals in A, i.e. 2*#superdiagonals + 1
 *    m     #cols of X
 * Inputs that are modified by this function:
 *    mx_A  mxArray containing [N x N] symm. pos. def. matrix in banded storage
 *    mx_X  mxArray containing [N x m] matrix
 */
template <typename T>
void posBandSolve(ptrdiff_t N, ptrdiff_t p, ptrdiff_t m, mxArray *mx_A, mxArray *mx_X)
{
    // Variables used by the LAPACK routine
    char uplo = 'U';                // Use the upper triangle of A
    ptrdiff_t nSupDiag = (p-1)/2;   // Number of superdiagonals in A
    ptrdiff_t info = 0;             // Status output for dpbsv
    // Extract data pointers
    T *A = static_cast<T*>(mxGetData(mx_A));
    T *X = static_cast<T*>(mxGetData(mx_X));
    // Call DPBSV or SPBSV as necessary
    pbsv(&uplo, &N, &nSupDiag, &m, A, &p, X, &N, &info);
    // Check that it succeeded
    if (info != 0)
        mexErrMsgIdAndTxt("MoDT:bandPosSolve:LAPACKError", 
                "LAPACK routine ?pbsv() exited with error");
}

/* Create a copy of an mxArray that is converted to single precision
 */
mxArray* copyToSingle(mxArray const *mx_X)
{
    mxArray *mx_X_copy;
    if (mxIsDouble(mx_X)) {
        // Cast double to single
        double *X = mxGetPr(mx_X);
        ptrdiff_t M = mxGetM(mx_X);
        ptrdiff_t N = mxGetN(mx_X);
        mx_X_copy = mxCreateUninitNumericMatrix( M, N, mxSINGLE_CLASS, mxREAL );
        std::copy( X, X+(M*N), static_cast<float*>(mxGetData(mx_X_copy)) );
    } else if (mxIsSingle(mx_X)) {
        // Source is already single-precision
        mx_X_copy = mxDuplicateArray(mx_X);
    } else {
        // Throw an error because this is weird
        mexErrMsgIdAndTxt("MoDT:bandPosSolve:TypeError","Unsupported datatype");
    }
}

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
    bool AB_is_dbl = mxIsDouble(mx_AB);
    if (!(AB_is_dbl || mxIsSingle(mx_AB)) || (N==0))
        mexErrMsgIdAndTxt(errId, "A_bands must be a [p x N] array of real numbers");
    if ((p%2==0) || (p>=2*N))
        mexErrMsgIdAndTxt(errId, "A_bands is an invalid # of rows");
    // b = input 1
    mxArray const *mx_b = prhs[1];
    ptrdiff_t m = mxGetN(mx_b);
    bool b_is_dbl = mxIsDouble(mx_b);
    if (!(b_is_dbl || mxIsSingle(mx_b)) || (mxGetM(mx_b)!=N) || (m==0))
        mexErrMsgIdAndTxt(errId, "b must be an [N x m] array of real numbers");
    
    // dpbsv overwrites A_bands and B, so we need to duplicate them
    mxArray *mx_AB_copy;
    mxArray *mx_b_copy;
    if (AB_is_dbl && b_is_dbl) {
        // Both are double, perform the operation in double-precision
        mx_AB_copy = mxDuplicateArray(mx_AB);
        mx_b_copy = mxDuplicateArray(mx_b);
        posBandSolve<double>(N, p, m, mx_AB_copy, mx_b_copy);
    } else {
        // Perform the operation in single-precision
        mx_AB_copy = copyToSingle(mx_AB);
        mx_b_copy = copyToSingle(mx_b);
        posBandSolve<float>(N, p, m, mx_AB_copy, mx_b_copy);
    }
    
    // Cleanup
    mxDestroyArray(mx_AB_copy);
    
    // Return mx_b_copy, which now contains x
    if (nlhs >= 1)
        plhs[0] = mx_b_copy;
    else
        mxDestroyArray(mx_b_copy);
}
