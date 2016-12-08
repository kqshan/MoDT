% MoDT              Mixture of drifting t-distributions (MoDT) model
%
% MoDT properties (unless otherwise specified, set using setParams):
% Dimensions (read-only)
%   D             - Number of dimensions of feature space
%   K             - Number of clusters in mixture model
%   T             - Number of time frames in model
%   N             - Number of spikes in training set
% Fitted model parameters
%   alpha         - [K x 1] mixing proportions
%   mu            - [D x T x K] cluster locations
%   C             - [D x D x K] cluster scale matrices
% User-defined model parameters
%   mu_t          - [T+1 x 1] time frame boundaries (ms)
%   nu            - t-distribution degrees of freedom
%   Q             - [D x D] drift regularization matrix, or scalar for diagonal
%   C_reg         - Diagonal ridge added to C to ensure well-conditioning
% Training data (set using attachData)
%   spk_Y         - [D x N] spike data
%   spk_t         - [N x 1] spike times (ms)
%   spk_w         - [N x 1] spike weights, or scalar for uniform weighting
% Other parameters
%   max_cond      - Maximum allowable condition number for C
%   use_gpu       - Use GPU acceleration (boolean).
%   use_mex       - Use precompiled subroutines, a.k.a. MEX files (boolean)
%
% MoDT methods:
% Basic object operations
%   MoDT          - Constructor
%   copy          - Create a deep copy of this handle class
%   saveobj       - Serialize this object to a MATLAB struct
%   loadobj       - Deserialize this object given a struct [Static]
% Set/get model parameters or data
%   setParams     - Assign values to model parameters
%   attachData    - Attach training data to model
%   detachData    - Detach training data, to save space
%   getValue      - Return computed values (see help for options)
%   initFromAssign - Initialize model based on spike assignments
% Model operations
%   EM            - Perform EM iterations
%   split         - Split a single cluster
%   merge         - Merge specified clusters
%   remove        - Remove specified cluster(s)
% Advanced model operations
%   Estep         - Perform the E step and return latent variables Z,U
%   Mstep         - Perform the M step given latent variables Z,U
%   getSubset     - Return an MoDT model with a subset of components

% Protected properties and methods
%
% Cache of intermediate results
% Properties:
%   mahal_dist    - [N x K] squared Mahalanobis distances
%   posterior     - [N x K] posterior cluster likelihoods
%   data_ll       - Data log-likelihood, i.e. log P(spk_Y|alpha,mu,C)
%   prior_ll      - Parameter prior log-likelihood, i.e. log P(alpha,mu,C)
% Methods:
%   fillCache     - Ensure that all cached intermediate results are available
%   clearCache    - Clear all cached intermediate results
%
% Assignment of spikes to frames and vice-versa
% Properties:
%   spike_frameid - [N x 1] time frame # (1..T) for each spike
%   frame_spklim  - [T x 2] [first, last] spike # (1..N) for each time frame
% Methods:
%   assignFrames  - Assign spikes to time frames
%   clearFrames   - Clear assignment of spikes to time frames
%
% Other protected methods [Static]:
%   reverseLookup - Given clust ID of each spike, return spk IDs for each clust
%   sparsifyZ     - Return a sparsified version of Z to speed up the M-step
% Other protected methods:
%   optimizeMu    - Solve the quadratic optimization for mu
%   optimizeC     - Solve the optimization for C
%   calcPriorLL   - Calculate the log-likelihood of the parameter prior
%
% MEX files for accelerating specific operations [Static]:
%   weightCovGpu  - Compute A*diag(w)*A' (on GPU)
%   calcMahalDistGpu - Calculate the Mahalanobis distance (on GPU)
%   sumFrames     - Sum the weighted data over each time frame
%   sumFramesGpu  - Sum the weighted data over each time frame (on GPU)
%   bandPosSolve  - Compute A\b where A is a banded positive definite matrix
%   
% MEX file management [Static]:
%   checkMexFiles - Check whether MEX files exist for this platform
%   buildMexFiles - Compile MEX files, overwriting them if they already exist
%   validateMex   - Run unit tests on MEX files



classdef MoDT < matlab.mixin.Copyable


% ------------------------------------------------------------------------------
% =========================     Class properties     ===========================
% ------------------------------------------------------------------------------


% Dimensions (read-only)
properties (SetAccess=protected)
    D = NaN;    % Number of dimensions of the feature space
    K = NaN;    % Number of clusters (a.k.a. components) in the mixture
    T = NaN;    % Number of time frames in the model
    N = NaN;    % Number of spikes in training set
end

% Fitted parameters
properties (SetAccess=protected)
    % alpha: [K x 1] mixing proportions, i.e. relative size of each cluster.
    alpha = zeros(0,1)
    
    % mu: [D x T x K] drifting cluster locations in feature space
    mu = zeros(0,0,0);
    
    % C: [D x D x K] cluster scale matrix for each cluster.
    C = zeros(0,0,0);
end

% User-defined model parameters
properties (SetAccess=protected)
    % mu_t: [T+1 x 1] time frame boundaries for drifting cluster locations. Time
    %       frame t is is defined as half-closed interval [ mu_t(t), mu_t(t+1) )
    mu_t = zeros(0,1);
    
    % nu: Degrees-of-freedom parameter for the t distribution. This may range
    %     from 1 to Inf, with smaller values corresponding to a heavier tails.
    %     nu=1 corresponds to a Cauchy distribution, and nu=Inf corresponds to a
    %     Gaussian distribution. Default = 7
    nu = 7;
    
    % Q: [D x D] cluster drift regularization matrix. If given as a scalar, it
    %    is interpreted as a diagonal matrix with the given scalar along the
    %    diagonal. Smaller values correspond to more regularization (producing
    %    an estimated mu that is smoother over time). The units of this quantity
    %    are (feature space units)^2/(time frame). Default = 0.033
    Q = 0.033;
    
    % C_reg: Scalar that will be added to the diagonal of the scale matrix C
    %        during the M-step. Setting C_reg > 0 can help ensure that C is 
    %        well-conditioned, but it also means that the M-step is no longer
    %        maximizing the expected likelihood. Hence you may encounter a case
    %        where running EM() actually reduces the overall log-likelihood. 
    %        Default = 0
    C_reg = 0;
end

% Training data (set using attachData)
properties (SetAccess=protected)
    % spk_Y: [D x N] spike data in feature space. If use_gpu==true, this will 
    %        be a gpuArray.
    spk_Y = zeros(0,0);
    
    % spk_t: [N x 1] spike times (ms)
    spk_t = zeros(0,1);
    
    % spk_w: [N x 1] spike weights, or scalar for uniform weighting. A single 
    %        spike with spk_w=2 is equivalent to two spikes at that location. 
    %        Default = 1. If use_gpu==true, this will be a gpuArray
    spk_w = 1;
end

% Other parameters
properties (SetAccess=protected)
    % max_cond: Maximum allowable condition number for the scale matrix C. If
    %           you call setParams() with an ill-conditioned C matrix, it will
    %           throw an error. If Mstep() produces an ill-conditioned C matrix,
    %           it will add max(svd(C))/max_cond to the diagonal of C, which
    %           should reduce cond(C) below max_cond. Default = 1e6
    max_cond = 1e6;
    
    % use_gpu: Use GPU acceleration (boolean). If enabled, this will store the
    %          training data (spk_Y and spk_w) and various intermediate values 
    %          (which are not publicly-accessible) as gpuArray objects. 
    %          Default = false
    use_gpu = false;
    
    % use_mex: Use precompiled dynamically-linked subroutines, a.k.a. MEX files.
    %          If enabled, this will call those MEX files instead of built-in
    %          MATLAB routines for a subset of the operations. Default = false
    %          See also: MoDT.buildMexFiles, MoDT.validateMex
    use_mex = false;
end

% Error IDs
properties (Constant, Hidden)
    badValueErrId = 'MoDT:BadValue';
    badDimErrId = 'MoDT:InconsistentDimensions';
end


% ------------------------------------------------------------------------------
% =====================     Basic object operations     ========================
% ------------------------------------------------------------------------------


methods
    function obj = MoDT(varargin)
        % MoDT constructor
        %   obj = MoDT(...)
        %
        % Any arguments are passed on to setParams()
        obj.setParams( varargin{:} );
    end
    
    function s = saveobj(self)
        % Convert an MoDT object to a struct
        %   s = saveobj(self)
        %
        % This method is called by MATLAB when saving an object to a .mat file,
        % but you can also use the resulting struct however you like (e.g. to
        % store it in a database as a blob attribute using DataJoint).
        % See also: MoDT.loadobj
        
        % It just so happens that all non-transient properties are also public,
        % so properties() returns exactly what we want
        fieldnames = properties(self);
        s = struct();
        for fn = fieldnames(:)'
            s.(fn{1}) = self.(fn{1});
        end
        % Convert the potential gpuArrays
        for fn = {'spk_Y','spk_w'}
            if isa(s.(fn{1}),'gpuArray'), s.(fn{1}) = gather(s.(fn{1})); end
        end
    end
end

methods (Static)
    % Create an MoDT object based on a struct created by saveobj()
    obj = loadobj(s); % Defined externally
end


% ------------------------------------------------------------------------------
% ================     Setting/getting parameters/data     =====================
% ------------------------------------------------------------------------------


methods
    % Set parameters of this MoDT object
    varargout = setParams(varargin); % Defined externally
    
    % Attach training data to this MoDT object
    attachData(self, Y, t, varargin); % Defined externally
    
    function detachData(self)
        % Clear training data to save space
        %   detachData(self)
        %
        % This may be a useful thing to do before saving your model
        self.spk_Y = zeros(0,0); self.spk_t = zeros(0,1); self.spk_w = 1;
        self.clearCache();
        self.clearFrames();
        % Also update the dimensions
        self.N = NaN;
        if isempty(self.mu) && isscalar(self.Q), self.D = NaN; end
    end
    
    % Return computed values from this model
    varargout = getValue(self, varargin);
    
    % Initialize the model based on spike assignments
    initFromAssign(self, assign, varargin);
end


% ------------------------------------------------------------------------------
% =======================     Model operations     =============================
% ------------------------------------------------------------------------------


methods
    % Perform EM iterations on this model
    exitcode = EM(self, varargin);
    
    % Split a single cluster into multiple clusters
    exitcode = split(self, clustId, varargin);
    
    % Merge the specified clusters into a single cluster
    exitcode = merge(self, clustIds, varargin);
    
    % Remove the specified cluster(s) from the model
    exitcode = remove(self, clustIds);
    
    % Perform the E-step and return latent variables Z,U
    [Z, U] = Estep(self);
    
    % Perform the M-step given latent variables Z,U
    Mstep(self, Z, U);
    
    % Return an MoDT model with a subset of components
    subModel = getSubset(self, clustIds, varargin);
end


% ------------------------------------------------------------------------------
% =================     Cache of intermediate results     ======================
% ------------------------------------------------------------------------------


properties (Access=protected, Transient)
    % mahal_dist: [N x K] squared Mahalanobis distances for each cluster x spike
    %             If use_gpu==true, this will be a gpuArray.
    mahal_dist = zeros(0,0);
    
    % posterior: [N x K] posterior likelihood, i.e. the probablity that spike n
    %            belongs to cluster k, given the observation y_n and the current
    %            model parameters. Columns of this matrix sum to one. 
    %            If use_gpu==true, this will be a gpuArray.
    posterior = zeros(0,0);
    
    % data_ll: Data log-likelihood, i.e. log P(spk_Y|alpha,mu,C)
    data_ll = NaN;
    
    % prior_ll: Parameter prior log-likelihood, i.e. log P(alpha,mu,C)
    %           In this model, this only depends on mu, and it isn't a proper
    %           probability because it doesn't integrate to one.
    prior_ll = NaN;
end

methods (Access=protected)
    function fillCache(self)
        % Ensure that all intermediate results are available in the cache
        %   updateCache(self)
        if isempty(self.posterior)
            self.Estep();
        end
    end
    
    function clearCache(self)
        % Clear all cached intermediate values
        %   clearCache(self)
        self.mahal_dist = zeros(0,0);
        self.posterior = zeros(0,0);
        self.data_ll = NaN;
        self.prior_ll = NaN;
    end
end


% ------------------------------------------------------------------------------
% ==============     Assignment of spikes to time frames     ===================
% ------------------------------------------------------------------------------


properties (Access=protected, Transient)
    % spike_frameid: [N x 1] time frame ID (1..T) for each spike. 
    %                If use_gpu==true, this will be a gpuArray.
    spike_frameid = zeros(0,1);
    
    % frame_spklim: [T x 2] where each row is the [first, last] spike ID (1..N)
    %               that falls within each time frame
    frame_spklim = zeros(0,2);
end

methods (Access=protected)
    function assignFrames(self)
        % Assign spikes to time frames. Call only after spk_t,mu_t have been set
        %   assignFrames(self)
        [nSpk_in_frame, self.spike_frameid] = histc(self.spk_t, self.mu_t);
        assert(nSpk_in_frame(end)==0);
        last_spk = cumsum(nSpk_in_frame(1:end-1));
        first_spk = [1; last_spk(1:end-1)+1];
        self.frame_spklim = [first_spk, last_spk];
        % Convert to gpuArray if desired
        if self.use_gpu, self.spike_frameid = gpuArray(self.spike_frameid); end
    end
    
    function clearFrames(self)
        % Reset assignments of spikes to time frames
        %   clearFrames(self)
        self.spike_frameid = zeros(0,1);
        self.frame_spklim = zeros(0,2);
    end
end


% ------------------------------------------------------------------------------
% ====================     Other protected methods     =========================
% ------------------------------------------------------------------------------


methods (Static, Access=protected)
    % Given cluster ID of each spike, return spike IDs for each cluster
    clust_spkId = reverseLookup( spk_clustId );
    
    % Return a sparser version of Z for the M-step
    Z = sparsifyZ( Z, thresh );
end

methods (Access=protected)
    % Solve the quadratic optimization to update mu
    mu = optimizeMu( self, wzu, C );
    
    % Solve the optimization to update C (and compute the Mahalanobis distance)
    [C, delta] = optimizeC( self, wzu, mu, sum_wz );

    % Calculate the log-likelihood of the parameter prior
    prior_ll = calcPriorLL( self );
end


% ------------------------------------------------------------------------------
% ===================     MEX files for performance     ========================
% ------------------------------------------------------------------------------


methods (Static)
    % Compile MEX files from C++/CUDA source
    buildMexFiles();
    
    % Run unit tests on the MEX files for this platform
    validateMex();
end

methods (Static, Access=protected)
    % Check whether MEX files exist for this platform
    [mex_exists, gpu_mex_exists] = checkMexFiles();
    
    % Compute C = A*diag(w)*A' (on GPU)
    C = weightCovGpu(A, weights, k);
    
    % Compute squared Mahalanobis distance, delta = sum((L\X).^2,1)', on GPU
    delta = calcMahalDistGpu(L, X);
    
    % Sum the weighted data over each time frame (dense weights)
    [wzuY, sum_wzu] = sumFrames(Y, wzu, frame_spklim);
    
    % Sum the weighted data over each time frame (dense weights, on GPU)
    [wzuY, sum_wzu] = sumFramesGpu(Y, wzu, frame_spklim);
    
    % Compute A\b where A is a positive definite matrix in banded storage
    x = bandPosSolve(A_bands, b);
end

end
