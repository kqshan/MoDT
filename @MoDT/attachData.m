function attachData(self, Y_, t_, varargin)
% Attach training data to this MoDT object
%   attachData(self, Y, t, [w], ...)
%
% Required arguments:
%   Y           [D x N] spike data
%   t           [N x 1] spike times (ms)
% Optional arguments [default]:
%   w           [N x 1] spike weights; scalar = uniform weighting  [ 1 ]
% Optional parameters (key/value pairs) [default]:
%   tlim        Time limits for auto-defining mu_t                 [t(1) t(end)]
%   frameDur    Frame duration (ms) for auto-definiting mu_t       [ 60e3 ]
%
% If mu_t has not been initialized, this will initalize mu_t as evenly-spaced
% time frames (with duration prm.frameDur) that starts at tlim(1) and may extend
% past tlim(end).

% Parse optional inputs
ip = inputParser();
ip.addOptional('w', 1, @(x) isnumeric(x) && (isscalar(x) || isvector(x)));
ip.addParameter('tlim', [], @(x) isempty(x) || numel(x)==2);
ip.addParameter('frameDur', 60e3, @(x) isscalar(x) && (x>0));
ip.parse( varargin{:} );
prm = ip.Results;

% (1) Data validation ----------------------------------------------------------

% Check dimensions etc
assert(~isempty(t_) && ~isempty(Y_), self.badValueErrId, ...
    'Cannot attach empty data; call detachData() instead');
% t (defines N)
t_ = t_(:);
N_ = numel(t_);
assert(issorted(t_), self.badValueErrId, 't must be sorted');
% Y
if isempty(self.mu) && isscalar(self.Q)
    D_ = size(Y_,1);
else
    D_ = self.D;
end
assert(isequal(size(Y_), [D_ N_]), self.badDimErrId, ...
    'Y must be [%d x %d]', D_,N_);
% w 
w_ = prm.w(:);
assert(isscalar(w_) || numel(w_)==N_, self.badDimErrId, ...
    'w must be a scalar or a vector with N=%d elements', N_);

% Define default tlim
if isempty(prm.tlim)
    prm.tlim = t_([1 end]);
end

% (2) Assign values ------------------------------------------------------------

% Clear caches
self.clearCache();
self.clearFrames();

% Assign Y,t,w
if self.use_gpu
    self.spk_Y = gpuArray(Y_);
    self.spk_t = t_;
    if isscalar(w_)
        self.spk_w = w_; % no need for gpuArray for scalar w
    else
        self.spk_w = gpuArray(w_);
    end
else
    self.spk_Y = Y_;
    self.spk_t = t_;
    self.spk_w = w_;
end
% Update dimensions
self.D = D_;
self.N = N_;

% Auto-assign mu_t if desired
if isempty(self.mu_t)
    t1 = prm.tlim(1);
    dt = prm.frameDur;
    if isfinite(dt)
        T_ = floor((prm.tlim(2)-t1)/dt) + 1;
        self.mu_t = t1 + (0:T_)'*dt;
    else
        T_ = 1;
        self.mu_t = [t1; Inf];
    end
    self.T = T_;
end

% Assign spikes to data frames
self.assignFrames();

end
