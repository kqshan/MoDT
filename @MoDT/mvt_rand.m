function X = mvt_rand(C, df, N)
% Sample from a multivariate t-distribution
%   X = mvt_rand(C, df, N)
%
% Returns:
%   X       [D x N] matrix where each column is a sample
%
% Required arguments:
%   C       [D x D] scale matrix
%   df      Degrees of freedom
%   N       Number of samples to generate
%
% This differs from MATLAB's mvtrnd function in that it allows you to
% specify a scale matrix rather than a correlation matrix

% The idea is to generate U ~ gamma(df/2, df/2), then generate
% X | U=u ~ normal(0, C/u), which we will do by scaling a standard
% multivariate Gaussian by (L/sqrt(u)) where L*L' = C
D = size(C,1);
L = chol(C,'lower');
% Matlab uses shape/rate instead of shape/scale for gamrnd
u = gamrnd(df/2, 2/df, [1 N]);
X = bsxfun(@rdivide, L * randn(D, N), sqrt(u));

end
