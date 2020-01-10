function x = bandPosSolve(A_bands, b)
% Solve A*x = b, where A is a positive definite matrix in banded storage
%   x = bandPosSolve(A_bands, b)
%
% Returns:
%   x           [N x m] solution vector
% Required arguments:
%   A_bands     [p x N] banded storage of a [N x N] positive definite matrix
%   b           [N x m] input vector
%
% The MEX version directly calls the LAPACK routine for solving a banded
% positive definite matrix, whereas the MATLAB code involves an inefficient step
% of converting this to a compressed sparse column (CSC) representation.
%
% The MEX version also supports single-precision arithmetic, whereas the MATLAB
% code performs everything in double-precision, though it will cast x as single
% if A_bands or b is single-precision.

[p,N] = size(A_bands);

% MATLAB doesn't support single-precision sparse yet (as of R2017a)
was_single = isa(A_bands,'single') || isa(b,'single');
A_bands = double(A_bands); b = double(b);

% Call sparse() to construct A
q = (p - 1) / 2; % Number of superdiagonals
i = bsxfun(@plus, (-q:q)', 1:N);
j = repmat(1:N, 2*q+1, 1);
mask = (A_bands ~= 0);
A = sparse(i(mask), j(mask), A_bands(mask), N, N);

% Solve A*x = b
x = A \ b;

% Cast to single if A or b was single
if was_single
    x = single(x);
end

end
