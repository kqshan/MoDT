function C = weightCovGpu(A, weights, k)
% Compute C = A*diag(w)*A' where w=weights(k,:)
%   C = weightCovGpu(A, weights, k);
%
% Returns:
%   C       [D x D] Symmetric positive definite gpuArray
% Required arguments:
%   A       [D x N] gpuArray
%   weights [N x K] gpuArray
%   k       Cluster index (1..K)
%
% The MEX version uses a custom CUDA kernel to efficiently compute this weighted
% covariance, even when D is small (i.e. D < 32)

% Scale A by sqrt(w)
A_scaled = bsxfun(@times, A, sqrt(weights(:,k))');

% Compute the covariance
C = A_scaled * A_scaled';

end
