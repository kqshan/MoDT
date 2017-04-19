function exitcode = split(self, clustId, varargin)
% Split a single cluster into multiple clusters
%   exitcode = split(self, clustId, [S,] ...)
%
% Returns:
%   exitcode        Numeric code indicating the outcome of this procedure:
%                     0: Completed successfully
%                    -1: EM failed (probably cluster starvation)
%                    -2: Initialization failed (cluster starvation)
% Required arguments:
%   clustId         Cluster index (1..K) to split
% Optional arguments [default]:
%   S               Number of clusters to split this into           [ 2 ]
% Optional parameters (key/value pairs) [default]:
%   splitInit       Initialization of the split (see below)         ['kmeans']
%   maxIter         Maximum # of EM iterations on split model       [ 10 ]
%   starveThresh    Minimum # of spikes per cluster                 [ 2*D ]
%   ...             Additional arguments are passed to EM()
%
% This method splits a cluster by fitting a MoDT model with S clusters to the
% subset of data assigned to the selected cluster. The splitInit parameter
% controls how this model is initalized:
%   'kmeans' - The data residual (Y-mu) is whitened and clustered using k-means
%       (with k=S). The initial M-step uses these cluster assigments and U=1.
%   [N x S] matrix of posteriors or [N x 1] vector of cluster IDs (1..S), to use
%       as cluster assignments for the initial M-step. This also specifies which
%       spikes to include in the data subset: any spikes assigned to cluster 0
%       or have sum(posterior,2)==0 will not be included in the data subset.
% After initialization, EM iterations are run on the subset model only, then the
% cluster parameters (alpha,mu,C) are substituted back into the full model. The
% other clusters of the full model are not affected.

D = self.D; K = self.K; N = self.N;

% (1) Deal with inputs ---------------------------------------------------------

% Check the clustId
assert(isscalar(clustId) && ismember(clustId,1:K), ...
    'MoDT:split:NoClusterSelected', 'Invalid cluster ID for splitting');

% Parse optional inputs
ip = inputParser();
ip.KeepUnmatched = true;
ip.addOptional('S', 2, @(x) isscalar(x) && isnumeric(x));
ip.addParameter('splitInit', 'kmeans');
ip.addParameter('maxIter', 10, @isscalar);
ip.addParameter('starveThresh', 2*D, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% (2) Initialize (data_mask and Z) ---------------------------------------------

S = prm.S;
badInitErrId = 'MoDT:split:BadInit';
minSpkPerCluster = prm.starveThresh;

% Check what we got as our splitInit
if ischar(prm.splitInit)
    % String specifying an initialization method
    init_method = prm.splitInit;
    switch (init_method)
        case 'kmeans'
            % Get the data points associated with this cluster
            post = self.getValue('posterior');
            data_mask = post(:,clustId) > 0.2;
            % Get the whitened residual
            L = chol(self.C(:,:,clustId));
            frames = self.spike_frameid(data_mask);
            X = L \ (self.spk_Y(:,data_mask) - self.mu(:,frames,clustId));
            % Run k-means until we get a valid initialization
            is_valid_init = false;
            nTries = 0;
            while ~is_valid_init && nTries <= 10
                nTries = nTries + 1;
                % Run k-means
                split_assign = kmeans(X', S, ...
                    'EmptyAction','singleton', 'OnlinePhase','off');
                % See if the initialization is valid
                spkCount = accumarray(split_assign, 1, [S 1]);
                is_valid_init = all(spkCount >= minSpkPerCluster);
            end
            % Convert the assignment to posterior likelihoods
            N_sub = numel(split_assign);
            Z = sparse((1:N_sub)',split_assign,1,N_sub,S);
        otherwise
            error(badInitErrId, 'Unknown splitInit method "%s', init_method);
    end
    
elseif numel(prm.splitInit)==N
    % [N x 1] vector of cluster IDs (1..S)
    split_assign = prm.splitInit(:);
    % Well, actually some can be 0 to indicate not to use this spike
    data_mask = (split_assign > 0);
    split_assign = split_assign(data_mask);
    % Convert to posterior likelihoods
    S = max(split_assign);
    N_sub = numel(split_assign);
    Z = sparse((1:N_sub)',split_assign,1,N_sub,S);
    % No sparse gpuArray support yet
    if self.use_gpu, Z = full(Z); end
    
elseif size(prm.splitInit,2)>1 && size(prm.splitInit,1)==N
    % [N x S] matrix of posterior likelihoods
    Z = prm.splitInit;
    % Some columns can be 0 to indicate not to use the spike
    S = size(Z,2);
    data_mask = any(Z,2);
    Z = Z(:,data_mask);
    
else
    error(badInitErrId, 'Invalid splitInit specification');
end
% Check if we ended up changing S
assert(S==prm.S || ismember('S',ip.UsingDefaults), 'MoDT:split:BadInit', ...
    'You specified S=%d, but splitInit had %d clusters', prm.S, S);
% Check that the initialization is valid
is_valid_init = all(sum(Z,1) >= minSpkPerCluster);
if ~is_valid_init
    exitcode = -2;
    return
end

% (3) Create and fit the subset model ------------------------------------------

% Call getSubset()
subModel = self.getSubset(clustId, 'dataMask',data_mask);

% Initialize the parameters of the subset model using these assignments
subModel.initFromAssign(Z, 'starveThresh',minSpkPerCluster, ...
    'minIter',1, 'maxIter',1);

% Run EM
sub_exitcode = subModel.EM('maxIter',prm.maxIter, 'starveAction','stop', ...
    'starveThresh',minSpkPerCluster, ip.Unmatched);

% Abort if EM failed
if sub_exitcode < 0
    exitcode = -1;
    return
end

% (4) Copy the subset model parameters back into the main model ----------------

% Append the new clusters at the end
old_alpha_k = self.alpha(clustId);
alpha_ = [self.alpha; old_alpha_k * subModel.alpha(2:end)];
mu_ = cat(3, self.mu, subModel.mu(:,:,2:end));
C_ = cat(3, self.C, subModel.C(:,:,2:end));
% Replace the old cluster with the first of the split clusters
alpha_(clustId) = old_alpha_k * subModel.alpha(1);
alpha_ = alpha_ / sum(alpha_);
mu_(:,:,clustId) = subModel.mu(:,:,1);
C_(:,:,clustId) = subModel.C(:,:,1);
% Assign these values to ourself
self.setParams('alpha',alpha_, 'mu',mu_, 'C',C_);

% Done!
exitcode = 0;

end
