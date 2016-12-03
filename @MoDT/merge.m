function exitcode = merge(self, clustIds, varargin)
% Merge the specified clusters into a single cluster
%   exitcode = merge(self, clustIds, ...)
% 
% Returns:
%   exitcode        Numeric code indicating the outcome of this procedure:
%                     0: Completed successfully (only possible outcome)
% Required arguments:
%   clustIds        List of cluster indices (1..K) to merge. This may also be
%                   specified as a [K x 1] logical vector.
% Optional parameters (key/value pairs) [default]:
%   maxIter         Maximum # of EM operations on subset model      [ 10 ]
%   ...             Additional arguments are passed to EM()
%
% This replaces the specified clusters with a single cluster by fitting a single
% cluster to the subset of data currently assigned to the specified clusters.
% Other clusters in the model are not affected by this operation.

K = self.K;

% (1) Deal with input ----------------------------------------------------------

% Determine which clusters will be merged
cl_merge = false(K,1);
cl_merge(clustIds) = true;
assert(any(cl_merge), 'MoDT:merge:NoClustersSelected', ...
    'No valid clusters were specified for merging');
% Determine which cluster will be replaced
if isnumeric(clustIds)
    merged_idx = clustIds(1);
else
    merged_idx = find(cl_merge,1,'first');
end
% And which cluster(s) will be removed
cl_keep = ~cl_merge;
cl_keep(merged_idx) = true;

% Parse optional inputs
ip = inputParser();
ip.KeepUnmatched = true;
ip.addParameter('maxIter', 10, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% (2) Fit the subset model -----------------------------------------------------

% Create the subset model
subModel = self.getSubset(cl_merge);

% Update the parameters so there is only one cluster
subModel.setParams('alpha',1, 'mu',mean(subModel.mu,3), 'C',mean(subModel.C,3));

% Fit it
subModel.EM('maxIter',prm.maxIter, 'starveAction','error', ip.Unmatched);

% (3) Update the current model -------------------------------------------------

% Replace the specified cluster
alpha_ = self.alpha; mu_ = self.mu; C_ = self.C;
merged_alpha = sum(alpha_(cl_merge));
alpha_(merged_idx) = merged_alpha;
mu_(:,:,merged_idx) = subModel.mu;
C_(:,:,merged_idx) = subModel.C;
% Remove the other clusters
alpha_ = alpha_(cl_keep);
mu_ = mu_(:,:,cl_keep);
C_ = C_(:,:,cl_keep);
% Update self
self.setParams('alpha',alpha_, 'mu',mu_, 'C',C_);

% Done!
exitcode = 0;

end
