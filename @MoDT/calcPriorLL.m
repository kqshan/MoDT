function param_ll = calcPriorLL( self )
% Calculate the log-likelihood of the parameter prior
%   param_ll = calcPriorLL( self )
%
% Returns the log-likelihood of the model parameters given the drift model. This
% is not a proper prior over the parameters because it doesn't integrate to one.
% However, for a given model class, it is proportional to a uniform prior over
% the remaining parameters, so it is still useful for evaluating EM convergence.

T = self.T; D = self.D;

% Calculate the log-likelihood of mu given the drift model
if T > 1
    drift = diff(self.mu, 1, 2);
    % It's the same Q for each cluster, so we can flatten out that dimension
    %   delta = drift(:,:)' * inv(Q) * drift(:,:)
    if isscalar(self.Q)
        % Q = eye(D)*self.Q;  inv(Q) = eye(D)*1/self.Q
        delta = sum(drift(:,:).^2, 1) / self.Q;
        logSqrtDetQ = -D/2*log(self.Q);
    else
        % Q = L*L'; inv(Q) = inv(L)'*inv(L)
        L = chol(self.Q,'lower');
        delta = sum((L \ drift(:,:)).^2,1);
        logSqrtDetQ = sum(log(diag(L)));
    end
    % Evaluate the multivariate Gaussian
    %   p(k,t) = 1/((2*pi)^(D/2) * sqrt(det(Q))) * exp(-1/2*delta(k,t))
    %   drift_ll = sum(log(p))
    logNormConst = -D/2*log(2*pi) - logSqrtDetQ;
    drift_ll = numel(delta)*logNormConst - sum(delta)/2;
else
    drift_ll = 0;
end

% No other priors on the parameters
param_ll = drift_ll;

end
