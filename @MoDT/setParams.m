function varargout = setParams(self, varargin)
% Set parameters of this MoDT object
%   setParams(self, ...)
%
% Unless otherwise specified, all of the publically-accessible properties of an
% MoDT object can be set using this function.
%
% New values can be specified as key/value pairs:
%   >> modt_obj.setParams('mu',new_mu, 'C',new_C)
% or as a struct:
%   >> s = struct('mu',new_mu, 'C',new_C);
%   >> modt_obj.setParams(s)
%
% If called without any arguments, this returns a list of field names that may
% be set by this function.
%   >> param_fields = modt_obj.setParams()
%
% See also: MoDT/attachData

% (1) Parse inputs -------------------------------------------------------------

% Create an inputParser to read the arguments.
ip = inputParser();
ip.addParameter('alpha', []);
ip.addParameter('mu'   , []);
ip.addParameter('mu_t' , []);
ip.addParameter('C'    , []);
ip.addParameter('nu'   , [], @isscalar);
ip.addParameter('Q'    , []);
ip.addParameter('C_reg', [], @isscalar);
ip.addParameter('max_cond',[], @isscalar);
ip.addParameter('use_gpu', [], @isscalar);
ip.addParameter('use_mex', [], @isscalar);
param_names = ip.Parameters(:);
% Special case when called with no arguments
if nargin == 1
    varargout = {param_names};
    return
end
% Parse the provided arguments
ip.parse( varargin{:} );
p = ip.Results;
user_spec = setdiff(param_names, ip.UsingDefaults);

% (2) Determine dimensions (D,K,T) ---------------------------------------------

% Use mu to determine the dimensions
dimChangeErrId = 'MoDT:setParams:CannotChangeDim';
if ismember('mu',user_spec)
    % User might be trying to change dimensions
    [D_, T_, K_] = size(p.mu);
    if D_~=self.D
        assert(ismember('C',user_spec), dimChangeErrId, ...
            'mu and C must both be specified to change the dimension D');
        assert(isempty(self.spk_Y), dimChangeErrId, ...
            'Data must be detached before changing the dimension D');
        assert(isscalar(self.Q) || ismember('Q',user_spec), dimChangeErrId, ...
            'Cannot change the dimension D because Q is [D x D]');
    end
    if T_ ~= self.T
        assert(ismember('mu_t',user_spec), dimChangeErrId, ...
            'mu and mu_t must both be specified to change # of time frames T');
    end
    if K_ ~= self.K
        assert(all(ismember({'alpha','C'},user_spec)), dimChangeErrId, ...
            'alpha, mu, and C must all be specified to change # of clusters K');
    end
else
    % User cannot change dimensions without setting mu, so use the old ones
    D_ = self.D; T_ = self.T; K_ = self.K;
    if isempty(self.mu)
        % Ensure that alpha or C are not set without simultaneously setting mu
        assert(~any(ismember({'alpha','C'},user_spec)), dimChangeErrId, ...
            'alpha, mu, and C must be initialized simultaneously');
        % Sorry, the user can change T by setting mu_t when self.mu is empty
        if ismember('mu_t',user_spec), T_ = max(0,numel(p.mu_t)-1); end
    end
end

% (3) Data validation ----------------------------------------------------------

if ismember('alpha',user_spec)
    assert(numel(p.alpha)==K_, self.badDimErrId, ...
        'alpha must be a vector with K=%d elements', K_);
    p.alpha = p.alpha(:);
    assert(isempty(p.alpha) || (all(p.alpha>=0) && abs(sum(p.alpha)-1)<1e-6), ...
        self.badValueErrId, 'alpha must be nonnegative and sum to 1');
end
reset_frame_assignment = false;
if ismember('mu_t',user_spec)
    p.mu_t = p.mu_t(:);
    assert(all(diff(p.mu_t)>0), self.badValueErrId, ...
        'mu_t must be strictly increasing');
    if ~isempty(self.spk_t)
        assert(p.mu_t(1)<=self.spk_t(1) && p.mu_t(end)>self.spk_t(end), ...
            self.badValueErrId, 'The range of mu_t must include all the data');
        reset_frame_assignment = true;
    end
end
max_cond_ = self.max_cond;
if ismember('max_cond',user_spec)
    assert(isscalar(p.max_cond) && p.max_cond > 1, self.badValueErrId, ...
        'max_cond must be a scalar > 1');
    max_cond_ = p.max_cond;
end
if ismember('C',user_spec)
    assert(size(p.C,1)==D_ && size(p.C,2)==D_ && size(p.C,3)==K_, ...
        self.badDimErrId, 'C must be [%d x %d x %d]', D_,D_,K_);
    p.C = double(p.C);
    is_ok = false(K_,1);
    for k = 1:K_
        is_ok(k) = is_pos_def(p.C(:,:,k), max_cond_);
    end
    assert(all(is_ok), self.badValueErrId, ...
        'C must be a well-conditioned, symmetric positive definite matrix');
end
if ismember('nu',user_spec)
    assert(isscalar(p.nu) && p.nu >= 1, self.badValueErrId, ...
        'nu must be a scalar and >= 1');
end
if ismember('Q',user_spec)
    assert(isscalar(p.Q) || isnan(D_) || isequal(size(p.Q),[D_ D_]), ...
        self.badDimErrId, 'Q must be a scalar or [%d x %d]', D_,D_);
    if isscalar(p.Q)
        assert(p.Q > 0, self.badValueErrId, 'Q must be positive');
    else
        assert(is_pos_def(p.Q, max_cond_), self.badValueErrId, ...
            'Q must be a well-conditioned, symmetric positive definite matrix');
    end
end
if ismember('C_reg',user_spec)
    assert(p.C_reg >= 0, self.badValueErrId, 'C_reg must be nonnegative');
end
use_gpu_ = self.use_gpu;
if ismember('use_gpu',user_spec)
    use_gpu_ = p.use_gpu;
    % Check that the GPU is accessible to MATLAB and supports doubles
    if p.use_gpu
        gpu_device = gpuDevice(); % Throws an error if we don't have a GPU
        assert(gpu_device.SupportsDouble, self.badValueErrId, ...
            'GPU device does not support double-precision arithmetic');
    end
    % Convert to/from gpuArrays
    if (p.use_gpu ~= self.use_gpu)
        % Convert the attached data
        if ~isempty(self.spk_Y)
            if p.use_gpu, self.spk_Y = gpuArray(self.spk_Y);
            else          self.spk_Y = gather(self.spk_Y);
            end
        end
        if ~isempty(self.spk_w) && ~isscalar(self.spk_w)
            if p.use_gpu, self.spk_w = gpuArray(self.spk_w);
            else          self.spk_w = gather(self.spk_w);
            end
        end
        % Let's just clear the caches and we'll reconstruct them later
        self.clearCache();
        reset_frame_assignment = true;
    end
end
use_mex_ = self.use_mex;
if ismember('use_mex',user_spec)
    use_mex_ = p.use_mex;
    % Check that MEX files exist for this platform
    [mex_exists, ~] = self.checkMexFiles();
    if p.use_mex && ~mex_exists
        warning('MoDT:setParams:MissingMex', ['You set use_mex=true, but ' ...
            'we could not find compiled routines (MEX files)\nfor your ' ...
            'platform, so you might not see the full speed-up.']);
    end
end
if use_gpu_ && any(ismember({'use_mex','use_gpu'},user_spec))
    % There is a somewhat complicated interaction between use_mex and use_gpu
    if use_mex_
        % GPU + MEX: check that MEX files exist for this platform
        [~,gpu_mex_exists] = self.checkMexFiles();
        if ~gpu_mex_exists
            warning('MoDT:setParams:MissingMex', ['You set use_mex=true, but'...
                ' we could not find compiled GPU routines (MEX files)\n' ...
                'for your platform. You may experience reduced performance.']);
        end
    else
        % GPU no MEX: 
        warning('MoDT:setParams:NonMexGPU', ['Certain MATLAB GPU routines ' ...
            'are unnecessarily slow. It is recommended\nthat you set '...
            'use_mex=true to use compiled routines (MEX files) instead.']);
    end
end

% (4) Clear caches if necessary ------------------------------------------------

if any(ismember({'alpha','mu','C','nu'},user_spec))
    self.clearCache();
end
if reset_frame_assignment
    self.clearFrames();
end

% (5) Update values ------------------------------------------------------------

% Set the values
for fn = user_spec(:)'
    self.(fn{1}) = p.(fn{1});
end

% Update D, K, T - a size of 0 is invalid so replace it with NaN
if D_==0, D_ = NaN; end
if K_==0, K_ = NaN; end
if T_==0, T_ = NaN; end
self.D = D_; self.K = K_; self.T = T_;

% Re-assign spikes to time frames if desired (and possible)
if reset_frame_assignment && ~isempty(self.mu_t) && ~isempty(self.spk_t)
    self.assignFrames();
end

end


% ========================     Helper functions     ============================


function tf = is_pos_def( X, max_cond )
% Return whether X is a well-conditioned positive definite matrix
%   tf = is_pos_def( X, max_cond )

% First, test for symmetry (and squareness)
if isequal(X, X')
    % If symmetric, also test for positive definiteness and condition number
    [~,p] = chol(X);
    tf = (p==0) && cond(X)<max_cond;
else
    % Otherwise, we fail
    tf = false;
end
end
