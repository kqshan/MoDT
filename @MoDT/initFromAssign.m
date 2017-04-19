function initFromAssign(self, assign, varargin)
% Initialize the model parameters based on spike assignments
%   initFromAssign(self, assign, ...)
%
% Required arguments:
%   assign          [N x K] soft assignments or [N x 1] cluster IDs (1..K)
%
% Optional parameters (key/value pairs) [default]:
%   starveThresh    Minimum # of spikes per cluster             [ 2*D ]
%   ...             Additional parameters are passed to EM()
% 
% The "assign" argument can be specified either as an [N x K] matrix of soft
% assignments, where assign(n,k) is the probability that spike n belongs to
% cluster k, or an [N x 1] vector of hard-assigned cluster IDs, where assign(n)
% specifies the cluster (1..K) that spike n belongs to. For best results, use
% soft assignments if possible; hard assignments can substantially underestimate
% the covariance of overlapping clusters.
errid_pfx = 'MoDT:initFromAssign';

% 1. Check the inputs ----------------------------------------------------------

% Make sure there is data attached
assert(~isempty(self.spk_Y), [errid_pfx ':MissingData'], ...
    'Data must be attached before calling initFromAssign()');
D = self.D; N = self.N; T = self.T;
% Convert the assignment to an [N x K] matrix of posteriors
errid = [errid_pfx ':BadAssign'];
errmsg = ['"assign" must be an [N x K] matrix of soft assignments or ' ...
    'an [N x 1] vector of cluster IDs'];
if isvector(assign)
    % Convert to an [N x K] matrix of posteriors
    assert(length(assign)==N, errid, errmsg);
    assert(all((assign==round(assign)) & (assign > 0)), errid, ...
        'assign: [N x 1] Cluster IDs must be positive integers');
    K_ = max(assign);
    assign = sparse(1:N, assign, 1, N, K_);
    % No sparse gpuArray support yet
    if self.use_gpu
        assign = full(assign);
    end
else
    % Make sure these are legitimate posteriors
    assert(size(assign,1)==N, errid, errmsg);
    all_nonneg = ~any(assign(:) < 0);
    within_tol = abs(sum(assign,2) - 1) < 1e-6;
    assert(all_nonneg && all(within_tol), errid, ...
        'assign: [N x K] Soft-assignments must be nonnegative and sum to 1');
    K_ = size(assign,2);
end

% Parse additional arguments
ip = inputParser();
ip.KeepUnmatched = true;
ip.addParameter('starveThresh', 2*D, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;
em_prm = ip.Unmatched;

% Check that the assignments satisfy the starvation threshold
assert(all(sum(assign,1) >= prm.starveThresh), errid, ...
    'Assignment does not satisfy starvation threshold for all clusters');

% 2. Initialize the model ------------------------------------------------------

% Get model parameters assuming a static model
% alpha
wz = bsxfun(@times, assign, self.spk_w);
sum_wz = sum(wz,1)';
alpha = sum_wz / sum(sum_wz);
% Mean
mu_static = bsxfun(@rdivide, self.spk_Y * wz, sum(wz,1)); % [D x K]
mu_static = repmat(reshape(mu_static, [D 1 K_]), [1 T 1]);
% Covariance
C_static = self.optimizeC(wz, mu_static, sum_wz);

% Perform a single M-step with the dynamic model
% Update the model with these parameters
self.setParams('alpha',alpha, 'mu',mu_static, 'C',C_static);
% Perform the M-step using U=1
self.Mstep( assign, 1 );

% 3. Run EM iterations ---------------------------------------------------------

% This is to converge on mu, C, and U
self.EM('starveThresh',prm.starveThresh, 'override_Z',assign, em_prm);

end
