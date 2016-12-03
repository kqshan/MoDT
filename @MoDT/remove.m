function exitcode = remove(self, clustIds)
% Remove the specified cluster(s) from the model
%   exitcode = remove(self, clustIds)
%
% Returns:
%   exitcode    Numeric code indicating the outcome of this procedure:
%                 0: Completed successfully (only possible outcome)
% Required arguments:
%   clustIds    List of cluster indices (1..K) to remove. This may also be
%               specified as a [K x 1] logical vector.
%
% This adjusts the other clusters' priors, but does not change their location or
% scale parameters.

% Determine which clusters we're keeping
cl_keep = true(self.K,1);
cl_keep(clustIds) = false;
assert(any(cl_keep), 'MoDT:remove:NoClustersLeft', ...
    'Cannot remove selected clusters; no clusters would remain');

% Clear the cache
delta = self.mahal_dist;
self.clearCache();

% Update the model parameters
alpha_ = self.alpha(cl_keep);
self.alpha = alpha_ / sum(alpha_);
self.mu = self.mu(:,:,cl_keep);
self.C = self.C(:,:,cl_keep);
self.K = sum(cl_keep);

% The Mahalanobis distance is actually preserved by this operation
if ~isempty(delta)
    self.mahal_dist = delta(:,cl_keep);
end

% Done!
exitcode = 0;

end
