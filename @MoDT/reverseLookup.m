function clust_spkId = reverseLookup( spk_clustId )
% Given cluster ID of each spike, return spike IDs that belong to each cluster
%   clust_spkId = reverseLookup( spk_clustId )
%
% Returns:
%   clust_spkId     {K x 1} of [n_k x 1] spike IDs (1..N) for each cluster
% Required arguments:
%   spk_clustId     [N x 1] cluster ID (1..K) for each spike
%
% This infers the number of clusters K from max(spk_clustId)

% Sort spk_clustId. This groups similar clustIds together, and the second output
% argument tracks the original spike IDs.
[sorted_clustIds, sorted_spkId] = sort( spk_clustId );

% Infer the number of clusters
K = sorted_clustIds(end);

% Count how many spikes are in each cluster.
N_k = accumarray(sorted_clustIds, 1, [K 1]);

% Split the sorted spike IDs into a cell array
clust_spkId = mat2cell(sorted_spkId, N_k, 1);

end
