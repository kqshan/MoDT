function subModel = getSubset(self, clustIds, varargin)
% Construct an MoDT model with a subset of clusters/data from the parent model
%   subModel = getSubset(self, clustIds, ...)
%
% Returns:
%   subModel    MoDT object with a subset of the clusters/data from self
% Required arguments:
%   clustIds    List of cluster indices (1..K) or [K x 1] logical vector
%               indicating which clusters to include in subset
% Optional parameters (key/value pairs) [default]:
%   dataMask    [N x 1] logical specifying data for subset  [ use dataThresh]
%   dataThresh  Threshold for inclusion in data subset      [ 0.2 ]
%
% This method constructs an MoDT model containing a subset of the clusters
% present in this model. If data is also attached, then the data is subsetted
% according to dataMask, and reweighted according to the posterior likelihood
% that each spike belongs to the selected subset of clusters. If dataMask is 
% not specified by the user, it is set as (posterior likelihood > dataThresh).

% Determine which clusters we're keeping
cl_subset = false(self.K,1);
cl_subset(clustIds) = true;
assert(any(cl_subset), 'MoDT:getSubset:NoClustersSelected', ...
    'No clusters selected for subsetting');

% Parse optional parameters
ip = inputParser();
ip.addParameter('dataMask', [], @(x) isempty(x) || ...
    (islogical(x) && numel(x)==self.N));
ip.addParameter('dataThresh', 0.2, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Construct the new model
subModel = self.copy();

% Set the parameters based on the selected subset
% If this is slow we could bypass setParams() and set the values directly
subModel.setParams( ...
    'alpha', self.alpha(cl_subset) / sum(self.alpha(cl_subset)), ...
    'mu'   , self.mu(:,:,cl_subset), ...
    'C'    , self.C(:,:,cl_subset) );

% Attach the subset of data
if ~isempty(self.spk_Y)
    % Get the posterior likelihood of belonging to the selected subset
    post = self.getValue('posterior');
    post = sum(post(:,cl_subset), 2);
    % Determine the data to attach
    data_mask = prm.dataMask(:);
    if isempty(data_mask)
        data_mask = (post > prm.dataThresh);
    end
    % Special case for scalar w
    if isscalar(self.spk_w) && (self.N > 1)
        w_masked = self.spk_w;
    else
        w_masked = self.spk_w(data_mask);
    end
    % Attach the selected subset of data
    subModel.attachData(self.spk_Y(:,data_mask), self.spk_t(data_mask), ...
        w_masked .* post(data_mask));
end

end
