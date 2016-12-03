function varargout = getValue(self, varargin)
% Return computed values from this model
%   [...] = getValue(self, ...)
%
% Specify values by name, e.g.:
%   >> [delta, post] = modt_obj.getValue('mahalDist','posterior');
%
% Values that you can request from this function:
%   mahalDist     [N x K] squared Mahalanobis distance for each spike x cluster
%   posterior     [N x K] posterior cluster likelihood for each spike x cluster
%   assignment    [N x 1] most likely cluster ID (1..K) for each spike
%   clusters      {K x 1} of [n_k x 1] spike IDs (1..N) assigned to each cluster
%   confusionMat  [K x K] confusion matrix (# of spikes from j assigned to i)
%   logLike       Overall log-likelihood = dataLogLike + priorLogLike
%   dataLogLike   Data log-likelihood = log P(spk_Y|alpha,mu,C)
%   priorLogLike  Param prior log-likelihood = log P(alpha,mu,C) = log P(mu)
%   spkFrame      [N x 1] time frame ID (1..T) that each spike belongs to
%   frameSpkLim   [T x 2] first and last spike ID (1..N) in each time frame
%
% Detailed explanations:
%   mahalDist(n,k) = squared Mahalanobis distance from cluster k to spike n.
%   posterior(n,k) = posterior probability that spike n came from cluster k. The
%       rows of this matrix sum to 1.
%   assignment(n) = most likely cluster (1..K) that spike n belongs to.
%   clusters{k} = vector of spikes (1..N) that have been assigned to cluster k.
%   confusionMat(i,j) = expected number of spikes assigned to cluster i that
%       were actually generated by cluster j. Specifically, this is computed as
%       confusionMat(i,j) = sum(posterior(assignment==i, j)). Also note that 
%       sum(confusionMat(i,:)) = sum(assignment==i).
%   logLike = dataLogLike + priorLogLike. This is the value being maximized by
%       the EM algorithm. Note that the EM() convTol is based on the change in
%       logLike/sum(spk_w), i.e. the log-likelihood per spike
%   dataLogLike = log P(spk_Y|alpha,mu,C) = log of the probability density of
%       observing the attached spike data given the current model parameters. 
%   priorLogLike = log P(alpha,mu,C) = log of the probability density of
%       observing the current model parameters given the prior distribution over
%       model parameters. Actually, this only depends on mu, and it is not a
%       true distribution (i.e. its integral over parameter space is not 1).
%   spkFrame and frameSpkLim describe how spikes have been assigned to time
%       frames. Note that frameSpkLim(t,1)=frameSpkLim(t-1,2)+1, and if a given
%       frame is empty then frameSpkLim(t,2)=frameSpkLim(t,1)-1.

% (1) Input validation ---------------------------------------------------------

% Requested values should be string
requested_values = varargin;
assert(iscellstr(requested_values), self.badValueErrId, ...
    'Requested values must be given as strings');
% Check the number of outputs
nValues = numel(requested_values);
nargoutchk(0, nValues);
% Make sure the caches are populated
self.fillCache();

% (2) Obtain the requested values ----------------------------------------------

values = cell(nValues,1);
for ii = 1:nValues
    name = requested_values{ii};
    switch (name)
        case 'mahalDist'
            val = self.mahal_dist;
        case 'posterior'
            val = self.posterior;
        case 'assignment'
            [~,spk_clustId] = max(self.posterior,[],2);
            val = spk_clustId;
        case 'clusters'
            [~,spk_clustId] = max(self.posterior,[],2);
            if self.use_gpu, spk_clustId = gather(spk_clustId); end
            val = MoDT.reverseLookup(spk_clustId);
        case 'confusionMat'
            Z = self.posterior;
            if self.use_gpu, Z = gather(Z); end
            [~,spk_clustId] = max(Z,[],2);
            val = sparse(spk_clustId,1:self.N,1,self.K,self.N) * Z;
        case 'logLike'
            val = self.data_ll + self.prior_ll;
        case 'dataLogLike'
            val = self.data_ll;
        case 'priorLogLike'
            val = self.prior_ll;
        case 'spkFrame'
            val = self.spike_frameid;
            assert(~isempty(val), self.badValueErrId, ...
                'spkFrame is not available; please attach data first');
        case 'frameSpkLim'
            val = self.frame_spklim;
            assert(~isempty(val), self.badValueErrId, ...
                'frameSpkLim is not available; please attach data first');
        otherwise
            error(self.badValueErrId, '"%s" is not a valid value name', name);
    end
    if isa(val,'gpuArray'), val = gather(val); end
    values{ii} = val;
end

% (3) Output -------------------------------------------------------------------

varargout = values;

end
