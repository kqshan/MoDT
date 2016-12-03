function Mstep(self, Z, U)
% Perform the M-step given latent variables Z,U.
%   Mstep(self, Z, U)
%
% Required arguments:
%   Z       [N x K] expected value of Z, i.e. the posterior cluster likelihoods
%   U       [N x K] expected value of U, the t-distribution scaling variable
%
% This method updates the fitted model parameters alpha, mu, C. It also clears
% cached intermediate results and updates the cached Mahalanobis distance.

K = self.K; N = self.N;

% Calculate some sums and construct wzu = w .* Z .* U
if isscalar(self.spk_w)
    wzu = self.spk_w * Z;
    sum_w = self.spk_w * N;
else
    if issparse(Z)
        [i,j,v] = find(Z);
        wzu = sparse(i,j,v.*self.spk_w(i),N,K);
    else
        wzu = bsxfun(@times, Z, self.spk_w);
    end
    sum_w = sum(self.spk_w);
end
sum_wz = sum(wzu,1)'; % [K x 1] sum of w.*z for each cluster
wzu = wzu .* U;
if self.use_gpu
    sum_wz = gather(sum_wz);
    sum_w = gather(sum_w);
end

% Clear the cache
self.clearCache();

% Update alpha
alpha_ = sum_wz / sum_w;
alpha_ = alpha_ / sum(alpha_); % Just to make sure it sums to 1

% Update mu
mu_ = self.optimizeMu( wzu, self.C );

% Update C and compute Mahalanobis distances
[C_, delta] = self.optimizeC( wzu, mu_, sum_wz );

% Get data from GPU
if self.use_gpu
    mu_ = gather(mu_);
end

% Update self
self.alpha = alpha_;
self.mu = mu_;
self.C = C_;
self.mahal_dist = delta;

end
