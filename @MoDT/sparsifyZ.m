function Z = sparsifyZ( Z, thresh )
% Return a sparser version of Z for the M-step
%   Z = sparsifyZ( Z, thresh )
%
% This method sparsifies Z by setting any values < thresh to zero. If it can
% achieve a high level of sparsity (80% or more elements are zero), it returns a
% sparse matrix. Otherwise, it leaves Z unchanged.

% Find the indices that correspond to Z >= thresh
[i,j] = find(Z >= thresh);
% Exit if the sparsity is not good enough
if numel(i) > 0.2*numel(Z)
    return
end

% Get the corresponding values of Z
[N,K] = size(Z);
v = Z(i+(j-1)*N);

% Since we set a bunch of terms to zero, the rows no longer sum to 1.
% Renormalize based on the new row sum
row_sum = accumarray(i, v, [N 1]);
v = v ./ row_sum(i);

% Create the sparse matrix
Z = sparse(i,j,v,N,K);

end
