function [wzuY, sum_wzu] = sumFrames(Y, wzu, f_spklim)
% Compute the weighted sum of data points for each time frame
%   [wzuY, sum_wzu] = sumFrames(Y, wzu, f_spklim)
%
% Returns:
%   wzuY        [D x K x T] weighted sums of data points in each time frame
%   sum_wzu     [K x T] sums of weights in each time frame
% Required arguments:
%   Y           [D x N] data points (D dimensions x N points)
%   wzu         [N x K] weights for each (data point) x (cluster)
%   f_spklim    [T x 2] [first,last] data index (1..N) in each time frame
%
% This performs the following for loop:
% for t = 1:T
%     n1 = frame_spklim(t,1); n2 = frame_spklim(t,2);
%     sum_wzu(:,t) = sum(wzu(n1:n2,:),1)';
%     wzuY(:,:,t) = Y(:,n1:n2) * wzu(n1:n2,:);
% end
%
% The MEX version performs the same exact operations, but reduces the overhead
% associated with the for loop.

% Get sizes
D = size(Y,1);
K = size(wzu,2);
T = size(f_spklim,1);
% Alocate memory
wzuY = zeros(D, K, T);
sum_wzu = zeros(K, T);

% For loop over time frames
for t = 1:T
    n1 = f_spklim(t,1); n2 = f_spklim(t,2);
    sum_wzu(:,t) = sum(wzu(n1:n2,:),1)';
    wzuY(:,:,t) = Y(:,n1:n2) * wzu(n1:n2,:);
end

end
