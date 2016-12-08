function exitcode = EM(self, varargin)
% Perform EM iterations on this model
%   exitcode = EM(self, ...)
%
% Returns:
%   exitcode        Numeric code indicating the outcome of this procedure:
%                     1: Completed successfully (terminated by maxIter)
%                     0: Completed successfully (converged)
%                    -1: Stopped early (cluster starvation)
% Optional parameters (key/value pairs) [default]:
%   verbose         Print status messages to stdout                 [ false ]
%   minIter         Minimum # of iterations to run                  [ 3 ]
%   maxIter         Maximum # of iterations to run                  [ 100 ]
%   convTol         Convergence tolerance (change in log-like/spk)  [ 1e-4 ]
%   starveThresh    Cluster starvation threshold (minimum # spk)    [ 2*D ]
%   starveAction    Action to take upon cluster starvation          ['error']
%                     error  - Stop fitting and throw an error
%                     stop   - Stop fitting and return quietly
%                     remove - Remove cluster and continue fitting
%   sparseThresh    Threshold for sparsifying Z (0 to disable)      [ 0 ]
%   override_Z      [N x K] cluster posterior likelihoods           [Use E-step]
%
% Detailed explanations for some parameters:
%   convTol: EM is considered to have converged when the absolute change in the
%       overall (data+prior) log-likelihood, divided by the number of spikes
%       (sum(self.spk_w)), falls below this threshold.
%   starveThresh: This is a minimum threshold on the number of spikes (before
%       weighting) assigned to a cluster. Cluster k is considered "starved" if
%       sum(Z(:,k)) < starveThresh. Its primary role is to ensure that
%       the scale matrix can be fitted reliably.
%   sparseThresh: When there are many clusters (K > 20), most entries of Z will
%       be very close to zero. We can create a sparse approximation of Z by
%       setting any values < sparseThresh to zero, thus reducing the complexity
%       of the M-step. Setting sparseThresh=0 will disable this optimization.
%       Empirically, I've found that with sparseThresh=0.01, the resulting
%       density of Z is about 2/K, and you need a density < 10% before it
%       becomes faster to use sparse matrices.
%   override_Z: This can be specified to override the values returned from the
%       E-step, which can be useful if you are fitting a MoDT model to already-
%       clustered data. If your classifier can return soft assignments, i.e. the
%       [N x K] posterior likelihood matrix, use that for best results. If you 
%       only have hard assignments, i.e. an [N x 1] vector of cluster IDs, you 
%       can pass that vector as override_Z and we will convert it for you.

% Parse optional arguments
ip = inputParser();
ip.addParameter('verbose', false, @isscalar);
ip.addParameter('minIter', 3, @isscalar);
ip.addParameter('maxIter', 100, @isscalar);
ip.addParameter('convTol', 0.0001, @isscalar);
ip.addParameter('starveThresh', 2*self.D, @isscalar);
ip.addParameter('starveAction', 'error', @ischar);
ip.addParameter('sparseThresh', 0, @isscalar);
ip.addParameter('override_Z', []);
ip.parse( varargin{:} );
prm = ip.Results;

prmErrId = 'MoDT:EM:BadParam';
% Check the starveAction
assert(ismember(prm.starveAction,{'error','stop','remove'}), ...
    prmErrId, 'Unsupported starveAction "%s"', prm.starveAction);
% Check override values
override_Z = prm.override_Z;
if ~isempty(override_Z)
    if isequal(size(override_Z), [self.N, self.K])
        % override_Z was given as an [N x K] matrix; we're all good
    elseif isvector(override_Z) && (numel(override_Z)==self.N)
        % override_z was given a [N x 1] cluster assignments (1..K)
        % Convert to a sparse [N x K] matrix
        override_Z = sparse(1:self.N, override_Z, 1, self.N, self.K);
    else
        error(prmErrId, ['override_Z must be an [N x K] matrix of posterior '...
            'likelihoods or an [N x 1] vector of cluster assignments']);
    end
end
% Some special considerations for gpuArrays
if self.use_gpu
    % Sparse operations are not currently supported for gpuArrays
    assert(prm.sparseThresh <= 0, ...
        'MoDT:EM:GPUError', ...
        'Sparse matrix operations are not currently supported for GPU arrays');
    % If Z is overridden, make sure it is a gpuArray
    if ~isempty(override_Z), override_Z = gpuArray(full(override_Z)); end
end

% Initial E step to get a base log-likelihood
[Z,U] = self.Estep();
last_loglike = self.data_ll + self.prior_ll;
% We will be normalizing the log-likelihood by the effective number of spikes
if isscalar(self.spk_w)
    nSpk = self.spk_w * self.N;
else
    nSpk = sum(self.spk_w);
end

% Report what's going on
verbose = prm.verbose;
if verbose
    fprintf(' Iter | Time (H:M:S) | Log-like/spike | Improvement \n');
    fprintf('------+--------------+----------------+-------------\n');
    disp_data = @(iter,ll,last_ll) fprintf('%5d | %12s |%15.6f |%12.6f\n', ...
        iter, datestr(now(),'HH:MM:SS.FFF'), ll/nSpk, (ll-last_ll)/nSpk);
end

% Run EM
currIter = 1;
keep_running = true;
while keep_running
    % Check for starvation
    clust_N = sum(Z,1)';
    if self.use_gpu, clust_N = gather(clust_N); end
    is_starved = (clust_N < prm.starveThresh);
    if any(is_starved) && isempty(override_Z)
        if verbose
            for k = find(is_starved)'
                fprintf('%5d | %12s | Cluster %d starved (%4.1f spikes)\n', ...
                    currIter, datestr(now(),'HH:MM:SS.FFF'), k, clust_N(k));
            end
        end 
        switch prm.starveAction
            case 'error'
                error('MoDT:EM:ClusterStarvation', ...
                    'Cluster starvation occured');
            case 'stop'
                exitcode = -1;
                if verbose, fprintf('Stopped due to cluster starvation\n'); end
                break;
            case 'remove'
                % Remove the starved cluster(s)
                self.remove(is_starved);
                % Run a new E-step
                [Z,U] = self.Estep();
                last_loglike = self.data_ll + self.prior_ll;
        end
    end
    
    % Sparsify Z if desired
    if (prm.sparseThresh > 0) && isempty(override_Z)
        Z = self.sparsifyZ( Z, prm.sparseThresh );
    end
    
    % Override Z if desired
    if ~isempty(override_Z)
        Z = override_Z;
    end
    
    % Perform the M step
    self.Mstep(Z,U);
    
    % Perform the E step
    [Z,U] = self.Estep();
    loglike = self.data_ll + self.prior_ll;
    if verbose, disp_data(currIter, loglike, last_loglike); end
    
    % Check for convergence
    improvement = loglike - last_loglike;
    if currIter < prm.minIter
        % Still below minIter, so keep going no matter what
    elseif abs(improvement/nSpk) < prm.convTol
        % Converged
        exitcode = 0;
        if verbose, fprintf('Converged\n'); end
        keep_running = false;
        % NB: Sometimes the improvement may be negative. This is because we find
        % the maximum-likelihood estimate for C in the M-step, but then add
        % C_reg to it, so the output of the M-step is not actually the MLE.
    elseif currIter >= prm.maxIter
        % Hit iteration limit
        exitcode = 1;
        if verbose, fprintf('Stopped due to iteration limit\n'); end
        keep_running = false;
    end
    
    % Iterate again
    last_loglike = loglike;
    currIter = currIter + 1;
end

% Done! We should have set exitcode in the EM loop.

end
