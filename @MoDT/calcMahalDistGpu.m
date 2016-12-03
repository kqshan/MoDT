function delta = calcMahalDistGpu(L, X)
% Compute the squared Mahalanobis distance, delta = sum((L\X).^2,1)', on GPU
%   delta = calcMahalDistGpu(L, X)
%
% Returns:
%   delta   [N x 1] squared Mahalanobis distance (gpuArray)
% Required arguments:
%   L       [D x D] lower triangular Cholesky-factorized covariance matrix
%   X       [D x N] data matrix (gpuArray)
%
% The MEX version uses a custom CUDA kernel that is optimized for small D.

% For some reason, the triangular solve is incredibly slow on gpuArrays.
% In contrast, general matrix multiplication seems to be pretty well-optimized,
% so we are going to assume well-conditioned matrices and explicitly invert L.
delta = sum((inv(L)*X).^2, 1)'; %#ok<MINV>

end
