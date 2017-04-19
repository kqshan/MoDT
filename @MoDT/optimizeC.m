function [C, delta] = optimizeC( self, wzu, mu, sum_wz )
% Solve the optimization to update C; also, compute the Mahalanobis distance
%   [C, delta] = optimizeC( self, wzu, mu )
%
% Returns:
%   C       [D x D x K] optimized estimate for cluster scale
%   delta   [N x K] squared Mahalanobis distance
% Required arguments:
%   wzu     [N x K] weight w * posterior likelihood Z * t-dist scaling U
%   mu      [D x T x K] estimate for cluster location
%   sum_wz  [K x 1] sum of w(n,k)*Z(n,k) over all spikes for each component

[N,K] = size(wzu); D = size(mu,1);
% To make sure that the output of optimizeC() will be accepted by setParams()
% (which checks the condition number of the C matrix it's given), we are going
% to use a slightly more conservative condition number (by 5%) here.
min_rcond = 1.05/self.max_cond;

% Allocate memory
C = zeros(D,D,K);
if self.use_gpu
    delta = zeros(N,K,'gpuArray');
else
    delta = zeros(N,K);
end

% Loop over clusters
for k = 1:K
    % X = spk_Y - mu
    mu_k = mu(:,:,k);
    X = self.spk_Y - mu_k(:,self.spike_frameid);
    
    % Update the covariance
    % C_k = X_scaled * diag(wzu(:,k)) * X_scaled' / sum_wz(k)
    if self.use_mex && self.use_gpu && ~issparse(wzu)
        C_k = self.weightCovGpu(X, wzu, k) / sum_wz(k);
    else
        if issparse(wzu)
            [i,~,v] = find(wzu(:,k));
            X_scaled = bsxfun(@times,X(:,i),sqrt(v)');
        else
            X_scaled = bsxfun(@times,X,sqrt(wzu(:,k))');
        end
        C_k = (X_scaled * X_scaled') / sum_wz(k);
    end
    if self.use_gpu, C_k = gather(C_k); end
    % Add a diagonal ridge for regularization
    C_k = C_k + self.C_reg*eye(D);
    % Make sure it is well-conditioned
    if rcond(C_k) < min_rcond % This is the 1-norm condition #, but close enough
        [U,S,~] = svd(C_k);
        s = diag(S);
        s = max(max(s)*min_rcond, s);
        U = U * diag(sqrt(s));
        C_k = U*U';
    end
    % Update the overall matrix
    C(:,:,k) = C_k;
    
    % Compute the Mahalanobis distance
    L = chol(C_k,'lower');
    % delta(:,k) = sum((L\K).^2, 1)';
    if self.use_mex && self.use_gpu
        delta(:,k) = self.calcMahalDistGpu(L,X);
    elseif self.use_gpu
        % As of R2015b, gpuArray gemm is much faster than trsm for small D
        delta(:,k) = sum((inv(L)*X).^2,1)'; %#ok<MINV>
    else
        delta(:,k) = sum((L\X).^2,1)';
    end
    % Go on to the next component
end

end
