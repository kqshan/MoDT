% Script that demonstrates how to use the MoDT model to compute unit isolation
% metrics. 

% Load some data
fprintf('Loading data\n');
example_data = load_example_data();

% These are the parameters that we recommend in the paper
nu = 7;                 % t-distribution nu parameter (smaller = heavier tails)
q_perhour = 2;          % Drift regularization (smaller = more smoothing)
timeframe_minutes = 1;  % Time frame duration (mostly a computational thing)

% Construct an MoDT object using these parameters
q_perframe = q_perhour * (timeframe_minutes/60);
model = MoDT('nu',7, 'Q',q_perframe);

% Attach the data to the model
timeframe_ms = timeframe_minutes * 60e3;
model.attachData( example_data.spk_Y, example_data.spk_t, ...
    'frameDur',timeframe_ms);

% Fit the model parameters based on our spike assignments
fprintf('Fitting model based on spike assignments\n');
clustAssigned = example_data.spk_clustId;
model.initFromAssign( clustAssigned, 'verbose',true );
% Obtain the posterior probability that spike n came from cluster k
posterior = model.getValue('posterior');

% Let's also fit a drifting Gaussian model
fprintf('Fitting a drifting Gaussian model by setting nu=Inf\n');
gaussModel = model.copy();
gaussModel.setParams('nu',Inf);
gaussModel.initFromAssign( clustAssigned );
[gaussPosterior, gaussMahalSq] = gaussModel.getValue('posterior','mahalDist');

% Report some unit isolation metrics
fprintf('\nUnit isolation quality metrics:\n');
fprintf(['%6s  %8s  ' repmat('%14s  ',[1 4]) '\n'], ...
    'Clust#','#Spikes','FP+FN(%),nu=7','FP+FN(%),Gauss','Iso.Dist.','L-ratio');
nClust = model.K;
nDims = model.D;
% Display the results in sorted order
for k = 1:nClust
    % False positive/negative ratios
    is_assigned_to_k = (clustAssigned == k);
    N_k = sum(is_assigned_to_k);
    otherClusts = [1:k-1, k+1:nClust];
    % T-distribution (nu=7)
    prob_came_from_k = posterior(:,k);
    prob_came_from_other = sum(posterior(:,otherClusts), 2);
    falsePos7 = sum( prob_came_from_other(is_assigned_to_k) ) / N_k;
    falseNeg7 = sum( prob_came_from_k(~is_assigned_to_k) ) / N_k;
    % Repeat this for the Gaussian
    prob_came_from_k = gaussPosterior(:,k);
    prob_came_from_other = sum(gaussPosterior(:,otherClusts), 2);
    falsePosGauss = sum( prob_came_from_other(is_assigned_to_k) ) / N_k;
    falseNegGauss = sum( prob_came_from_k(~is_assigned_to_k) ) / N_k;
    
    % Compute the isolation distance and L-ratio as well
    mahalDistSq_otherSpikes = gaussMahalSq(~is_assigned_to_k, k);
    % Isolation distance
    mahalDistSq_sorted = sort(mahalDistSq_otherSpikes);
    isolationDist = mahalDistSq_sorted(N_k);
    % L-ratio
    Lratio = sum(chi2cdf(mahalDistSq_otherSpikes, nDims, 'upper')) / N_k;
    
    % Report these values
    fprintf(['%6d  %8d  ' repmat('%14.3f  ',[1 3]) '%14.6f  \n'], ...
        k, N_k, (falsePos7+falseNeg7)*100, ...
        (falsePosGauss+falseNegGauss)*100, isolationDist, Lratio);
end
