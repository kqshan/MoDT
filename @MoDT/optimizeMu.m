function mu = optimizeMu(self, wzu, C)
% Solve the quadratic optimization to update mu
%   mu = optimizeMu(self, wzu)
%
% Returns:
%   mu      [D x T x K] optimized estimate for cluster locations
% Required arguments:
%   wzu     [N x K] weight w * posterior likelihood Z * t-dist scaling U
%   C       [D x D x K] estimate for cluster scale
D = self.D; K = self.K; T = self.T;

% Special case for T==1
if T==1
    mu = bsxfun(@rdivide, self.spk_Y * wzu, sum(wzu,1));  % [D x K]
    mu = reshape(mu, [D 1 K]);
    return
end

% Calculate weighted sums for each time block
[wzuY, sum_wzu] = calc_weighted_sums( ...
    wzu, self.spk_Y, self.spike_frameid, self.frame_spklim, self.use_mex);

% Solve mu = H \ b
% Start with the process information matrix
B_Q = make_Qinv_matrix( self.Q, D, T );
% This Qinv matrix determines the banded structure of H, so we can pre-compute
% some structural information that will be useful for our call to sparse()
% later. We don't need this for the MEX version, which uses this banded storage
% format directly.
if ~self.use_mex
    nSuperDiag = (size(B_Q,1) - 1) / 2;
    H_i = bsxfun(@plus, (-nSuperDiag:nSuperDiag)', 1:D*T);
    H_j = repmat(1:D*T, 2*nSuperDiag+1, 1);
    H_mask = (H_i > 0) & (H_i <= D*T);
    H_i = H_i(H_mask);
    H_j = H_j(H_mask);
end
% Allocate memory
if self.use_gpu
    mu = zeros(D, T, K, 'gpuArray');
else
    mu = zeros(D, T, K);
end
% Do each cluster
for k = 1:K
    % Make H = Qinv_matrix + Cinv_matrix
    % Get the observation information matrix
    C_k = C(:,:,k);
    B_C = make_Cinv_matrix( C_k, sum_wzu(:,k) );
    % Add the two matrices together to get H
    B = B_Q;
    offset = (size(B_Q,1) - size(B_C,1)) / 2;
    B(offset+1:end-offset,:) = B(offset+1:end-offset,:) + B_C;
    
    % Calculate b = C_k \ wzuY(:,:,k)
    b = C_k \ wzuY(:,:,k);
    
    % Solve H \ b
    if ~self.use_mex
        % Construct H as a sparse matrix
        H = sparse(H_i, H_j, B(H_mask), D*T, D*T);
        % Solve
        mu_k = H \ b(:);
    else
        % Call our MEX function, which uses this banded format directly
        mu_k = self.bandPosSolve(B, b(:));
    end
    mu(:,:,k) = reshape(mu_k, [D T]);
end

end


% =========================     Helper functions     ===========================


function [wzuY, sum_wzu] = calc_weighted_sums(wzu, Y, spk_fid, f_spklim, use_mex)
% Calculate the weighted sums for each time block
%   [wzuY, sum_wzu] = calc_weighted_sums( wzu, Y, spk_fid, f_spklim, use_mex )
%
% Returns:
%   wzuY        [D x T x K] weighted sum of points in each time block
%   sum_wzu     [T x K] sum of the weights in each time block
% Required arguments:
%   wzu         [N x K] weighting for each (cluster x spike)
%   Y           [D x N] spike data
%   spk_fid     [N x 1] time frame ID (1..T) for each spike
%   f_spklim    [T x 2] [first,last] spike ID (1..N) in each time frame
%   use_mex     Whether to use MEX routines (boolean)
%
% More specifically, this returns wzuY and sum_wzu such that
%   wzuY(:,t,k) = Y(:,n_t)*wzu(k,n_t)'  and  sum_wzu(t,k) = sum(wzu(k,n_t))
% where n_t = frame_spklim(t,1):frame_spklim(t,2). There is some optimization
% for the case where wzu is sparse.
[N,K] = size(wzu);
D = size(Y,1);
T = size(f_spklim,1);
% We have some optimization for sparse weight matrices
if issparse(wzu)
    % Sparse wzu
    [i,j,v] = find(wzu);
    % Rearrange wzu into a [N x T*K] matrix so columns are consecutive frames
    j = spk_fid(i) + T*(j-1);
    wzu_byframe = sparse(i,j,v,N,T*K);
    % Now we can get wzuY by simply multiplying Y * wzu_byframe
    wzuY = Y * wzu_byframe;
    % Similarly, we can get sum_wzu by summing the columns by wzu_byframe
    sum_wzu = sum(wzu_byframe,1);
    % Finally, reshape to match our desired shape
    wzuY = reshape(wzuY, [D T K]);
    sum_wzu = reshape(sum_wzu, [T K]);
else
    % Dense wzu
    % Start by computing them as [K x T] and [D x K x T]
    if ~use_mex
        % Allocate memory
        if isa(Y,'gpuArray')
            sum_wzu = zeros(K,T,'gpuArray');
            wzuY = zeros(D,K,T,'gpuArray');
        else
            sum_wzu = zeros(K,T);
            wzuY = zeros(D,K,T);
        end
        % For loop over time frames
        for t = 1:T
            n1 = f_spklim(t,1); n2 = f_spklim(t,2);
            wzu_t = wzu(n1:n2,:);
            sum_wzu(:,t) = sum(wzu_t,1)';
            wzuY(:,:,t) = Y(:,n1:n2) * wzu_t;
        end
    else
        % Call a MEX routine
        if isa(Y,'gpuArray')
            [wzuY, sum_wzu] = MoDT.sumFramesGpu(Y, wzu, f_spklim);
        else
            [wzuY, sum_wzu] = MoDT.sumFrames(Y, wzu, f_spklim);
        end
    end
        
    % Reshape to match our desired shape
    sum_wzu = sum_wzu'; % now [T x K]
    wzuY = permute(wzuY, [1 3 2]); % now [D x T x K]
    % Gather
    if isa(sum_wzu,'gpuArray')
        sum_wzu = gather(sum_wzu);
        wzuY = gather(wzuY);
    end
end
end


function B = make_Qinv_matrix( Q, D, T )
% Construct the banded representation of the process information matrix
%   B = make_Qinv_matrix( Q, D, T )
%
% Returns:
%   B       [p x D*T] column-major banded storage of a [D*T x D*T] matrix
% Required arguments:
%   Q       Drift covariance: [D x D] matrix, or scalar (for diagonal Q)
%   D       Number of dimensions in the spike feature space
%   T       Number of time frames
%
% This represents the [D*T x D*T] process information matrix Qinv:
%   [  Q^-1  -Q^-1               ]
%   [ -Q^-1  2Q^-1  -Q^-1        ]
%   [        -Q^-1   ...   -Q^-1 ]
%   [               -Q^-1   Q^-1 ]
% in column-major banded storage. In this storage format, the columns of B still
% correspond to the columns of Qinv, but the rows of B correspond to the
% diagonals of Qinv. In other words:
%   [ 11  12   0   0 ]         [ **  12  23  34 ]
%   [ 21  22  23   0 ]   ==>   [ 11  22  33  44 ]
%   [  0  32  33  34 ]         [ 21  32  43  ** ]
%   [  0   0  43  44 ]
% 
% Note that is not the same format as returned by spdiags!!
assert(T > 1);
if isscalar(Q)
    % This means a diagonal Q matrix with the scalar Q along the diagonal,
    % so the inverse is simply 1/Q along the diagonal
    B = zeros(2*D+1, D*T);
    % Off-diagonal bands of inv(Q)
    B(1, D+1:end) = -1/Q;
    B(end, 1:end-D) = -1/Q;
    % inv(Q) and 2*inv(Q) along the main diagonal
    B(D+1, 1:D) = 1/Q;
    B(D+1, D+1:end-D) = 2/Q;
    B(D+1, end-D+1:end) = 1/Q;
else
    % General case: arbitrary Q^-1
    Qinv = inv(Q);
    Qinv = (Qinv + Qinv')/2; % Enforce symmetry
    % Make the 3x3 block case
    Q3 = [Qinv, -Qinv, zeros(D); -Qinv, 2*Qinv, -Qinv; zeros(D), -Qinv, Qinv];
    % Convert the [3*D x 3*D] full format into a [p x 3*D] banded format
    B3 = zeros(4*D-1, 3*D);
    idx_i = bsxfun(@plus, (-2*D+1:2*D-1)', 1:3*D);
    idx_j = 1:3*D;
    idx_lin = bsxfun(@plus, idx_i, (idx_j-1)*(3*D));
    mask = (idx_i > 0) & (idx_i <= 3*D);
    B3(mask) = Q3(idx_lin(mask));
    % Trim away extraneous diagonals
    nz_offset = find(~all(B3==0,2), 1, 'first') - 1;
    B = B3(nz_offset+1:end-nz_offset,:);
    % Repeat this as necessary
    B = horzcat(B(:,1:D), repmat(B(:,D+1:2*D),[1,T-2]), B(:,2*D+1:3*D));
end
end


function B = make_Cinv_matrix( C, wzu )
% Construct the banded representation of the observation information matrix
%   B = make_Cinv_matrix( C, wzu )
%
% Returns:
%   B       [2*D-1 x D*T] column-major banded storage of a [D*T x D*T] matrix
% Required arguments:
%   C       [D x D] Observation covariance matrix
%   wzu     [T x 1] weight for each time frame
%
% This represents the [D*T x D*t] observation matrix:
%   [ wzu(1)*C^-1                                     ]
%   [             wzu(2)*C^-1                         ]
%   [                             ...                 ]
%   [                                     wzu(T)*C^-1 ]
% in the column-major banded storage format. In this format,
%   [ 11  12   0   0 ]         [ **  12  23  34 ]
%   [ 21  22  23   0 ]   ==>   [ 11  22  33  44 ]
%   [  0  32  33  34 ]         [ 21  32  43  ** ]
%   [  0   0  43  44 ]
% 
% Note that is not the same format as returned by spdiags!!
Cinv = inv(C);
Cinv = (Cinv + Cinv')/2; % Enforce symmetry
% Convert the [D x D] full format into a [2*D-1 x D] banded format
D = size(C,1);
B1 = zeros(2*D-1, D);
idx = bsxfun(@plus, (1:D)', (D-1:-1:0) + (0:D-1)*size(B1,1));
B1(idx) = Cinv(:);
% Repeat and scale by wzu
B = kron(wzu', B1);
end
