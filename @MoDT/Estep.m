function [Z, U] = Estep(self)
% Perform the E-step and return latent variables Z,U.
%   [Z, U] = Estep(self)
%
% Returns:
%   Z       [N x K] expected value of Z, i.e. the posterior cluster likelihoods
%   U       [N x K] expected value of U, the t-distribution scaling variable
%           Will be a scalar 1 if nu is infinity.
%
% This method also updates the cached intermediate results.
D = self.D; K = self.K; N = self.N;

% Calculate the Mahalanobis distances and determinant of the covariance
logSqrtDetC = zeros(K,1);
if isempty(self.mahal_dist)
    % Need to compute the Mahalanobis distance
    % Allocate memory
    if self.use_gpu
        delta = zeros(N,K,'gpuArray');
        mu = gpuArray(self.mu);
    else
        delta = zeros(N,K);
        mu = self.mu;
    end
    % Compute for each 
    for k = 1:K
        mu_k = mu(:,:,k);
        L = chol(self.C(:,:,k),'lower');
        logSqrtDetC(k) = sum(log(diag(L)));
        X = self.spk_Y - mu_k(:,self.spike_frameid);
        if self.use_mex && self.use_gpu
            delta(:,k) = self.calcMahalDistGpu(L,X);
        else
            delta(:,k) = sum((L\X).^2,1)';
        end
    end
    % Cache result
    self.mahal_dist = delta;
else
    % Mahalanobis distance is cached from the M-step
    delta = self.mahal_dist;
    for k = 1:K
        logSqrtDetC(k) = det(self.C(:,:,k)); % Not logSqrt yet
    end
    logSqrtDetC = log(logSqrtDetC) / 2;
end

% Calculate Z
% Start by computing log(like(n,k)) = log(alpha_k) + log(P(Y_n | mu_k,C_k,nu))
% Start by setting Z = like(n,k) = alpha_k * P(Y_n | mu_k, C_k, nu)

% Start by setting Z := log(like(n,k)) = log(alpha_k * P(Y_n | mu_k,C_k,nu))
nu = self.nu;
logAlpha = log(self.alpha);
if isinf(nu)
    % Gaussian
    logNormConst = logAlpha - D/2*log(2*pi) - logSqrtDetC;
    Z = bsxfun(@plus, logNormConst', -delta/2);
else
    % T-distribution
    logNormConst = logAlpha + gammaln((nu+D)/2) - gammaln(nu/2) ...
        - (D/2)*log(nu*pi) - logSqrtDetC;
    Z = bsxfun(@plus, logNormConst', -(nu+D)/2*log(1 + delta/nu));
end
% Factorize the likelihood into like(n,k) = Z(n,k) * exp(loglike_offset(n))
loglike_offset = max(Z,[],2);
Z = exp(bsxfun(@minus, Z, loglike_offset));
% Convert Z to the posterior likelihood Z(n,k) = P(spk n from cluster k)
sum_Z = sum(Z,2);
Z = bsxfun(@rdivide, Z, sum_Z);
% This also gives us the unweighted data log-likelihood
unweighted_data_ll = log(sum_Z) + loglike_offset;

% Calculate U
if isinf(nu)
    U = 1;
else
    U = (nu + D) ./ (nu + delta);
end

% Compute the weighted data likelihood
if isscalar(self.spk_w)
    weighted_data_ll = sum(unweighted_data_ll) * self.spk_w;
else
    weighted_data_ll = unweighted_data_ll' * self.spk_w;
end
if self.use_gpu, weighted_data_ll = gather(weighted_data_ll); end

% Update the cache
self.posterior = Z;
self.data_ll = weighted_data_ll;
self.prior_ll = self.calcPriorLL();

end
