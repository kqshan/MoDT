% Script that measures the fitting time for the MoDT vs a mixture of Gaussians

% Load some data
fprintf('Loading data\n');
example_data = load_example_data();
[D,N] = size(example_data.spk_Y);
K = max(example_data.spk_clustId);

% These are the parameters that we recommend in the paper
nu = 7;                 % t-distribution nu parameter (smaller = heavier tails)
q_perhour = 2;          % Drift regularization (smaller = more smoothing)
timeframe_minutes = 1;  % Time frame duration (mostly a computational thing)
% For benchmarking we are going to use a fixed number of iterations
nIter = 20;

fprintf('Benchmarking: D=%d, N=%d, K=%d, %d EM iterations each\n', ...
    D, N, K, nIter);

%% Benchmark fitgmdist ---------------------------------------------------------

fprintf('Running fitgmdist to fit a stationary mixture of Gaussians\n');

% Use the spike assignments to construct a starting point
MoG_Y = example_data.spk_Y';
MoG_Start = struct('mu',zeros(K,D), 'Sigma',zeros(D,D,K), ...
    'ComponentProportion',accumarray(example_data.spk_clustId,1)/N );
for k = 1:K
    mask = (example_data.spk_clustId == k);
    MoG_Start.mu(k,:) = mean(MoG_Y(mask,:), 1);
    MoG_Start.Sigma(:,:,k) = cov(MoG_Y(mask,:));
end
% Disable the warning about not converging
warnState = warning('off','stats:gmdistribution:FailedToConverge');
warnStateCleanup = onCleanup(@() warning(warnState));
% Call it once to make sure it's warmed up
options = statset('MaxIter',3);
MoG_model = fitgmdist(MoG_Y, K, 'Start',MoG_Start, 'Options',options);
% Call it for real. Set TolFun very low so it'll hit MaxIter first
options = statset('MaxIter',nIter, 'TolFun',1e-16);
t_start = tic();
MoG_model = fitgmdist(MoG_Y, K, 'Start',MoG_Start, 'Options',options);
MoG_runtime = toc(t_start);
clear warnStateCleanup
% Display the results
MoG_loglike = -MoG_model.NegativeLogLikelihood / N;
fprintf('  Completed in %.2f sec, log-likelihood (per spike) = %.4f\n', ...
    MoG_runtime, MoG_loglike);

%% Benchmark MoDT --------------------------------------------------------------

% Try three different cases
nCases = 4;
% Construct a stationary starting point 
starting_mu = permute(MoG_Start.mu, [2 3 1]);
starting_C = MoG_Start.Sigma;
starting_alpha = MoG_Start.ComponentProportion;
% Construct a base object with some params set
q_perframe = q_perhour * (timeframe_minutes / 60);
modt_base = MoDT('nu',nu, 'Q',q_perframe);
% Use MEX files if available
try
    modt_base.setParams('use_mex',true);
catch
    warning('demo_benchmarking:NoMex', ...
        'MEX files not found for your platform; performance may be impacted');
end
% Run the GPU case if possible
try %#ok<TRYNC>
    modt = modt_base.copy();
    modt.setParams('use_gpu',true, 'use_mex',true);
    nCases = 6;
end
% Try each of the different cases
modt_desc = cell(nCases,1);
modt_runtime = zeros(nCases,1);
modt_loglike = zeros(nCases,1);
for ii = 1:nCases
    % Spike data for fitting
    spk_Y = example_data.spk_Y;
    spk_t = example_data.spk_t;
    spk_w = 1;
    % Start with the base case
    modt = modt_base.copy();
    frameDur = timeframe_minutes * 60e3;
    T = ceil(max(spk_t)/frameDur);
    mu_t = (0:T)' * frameDur;
    % Modify it to fit our case
    switch ii
        case 1
            modt_desc{ii} = 'MoDT with T=1, nu=Inf (MoG equivalent)';
            mu_t = [0; T] * frameDur;
            T = 1;
            modt.setParams('nu',Inf);
        case 2
            modt_desc{ii} = sprintf('MoDT with T=%d, nu=Inf', T);
            modt.setParams('nu',Inf);
        case {3,5}
            modt_desc{ii} = sprintf('MoDT with T=%d, nu=%g', T, nu);
        case {4,6}
            subsetMask = (example_data.weighting > 0);
            subsetFrac = mean(subsetMask);
            modt_desc{ii} = sprintf('MoDT with T=%d, nu=%g, %.0f%% subset', ...
                T, nu, subsetFrac * 100);
            spk_Y = spk_Y(:,subsetMask);
            spk_t = spk_t(subsetMask);
            spk_w = example_data.weighting(subsetMask);
    end
    starting_params = struct('alpha',starting_alpha, 'C',starting_C, ...
        'mu',repmat(starting_mu,[1 T 1]), 'mu_t',mu_t );
    % Enable GPU if specified
    if (ii >= 5)
        modt_desc{ii} = [modt_desc{ii} ', GPU'];
        modt.setParams('use_gpu',true, 'use_mex',true);
    end
    % Try a dry run first
    modt.setParams(starting_params);
    modt.attachData(spk_Y, spk_t);
    modt.EM('maxIter',3);
    modt.detachData();
    % Do it for real
    fprintf('Running %s\n', modt_desc{ii});
    t_start = tic();
    modt.setParams(starting_params);
    modt.attachData(spk_Y, spk_t);
    modt.EM('minIter',nIter, 'maxIter',nIter);
    t_elapsed = toc(t_start);
    % If this was a subset, then measure the log-likelihood on the full dataset
    if modt.N < numel(example_data.spk_t)
        modt.attachData(example_data.spk_Y, example_data.spk_t);
    end
    % Save and report the results
    modt_runtime(ii) = t_elapsed;
    ll_perSpike = modt.getValue('dataLogLike') / modt.N;
    modt_loglike(ii) = ll_perSpike;
    fprintf('  Completed in %.2f sec, log-likelihood (per spike) = %.4f\n', ...
        t_elapsed, ll_perSpike);
end

%% Display the results in a table ----------------------------------------------

% Table header
descLen = max(cellfun(@length, modt_desc));
fprintf('%-*s  %12s  %12s\n', descLen, 'Description', 'LLR/spk', 'Runtime (s)');
% MoG
fprintf('%-*s  %12s  %12.2f\n', descLen, ...
    'fitgmdist (mixture of Gaussians)', '(reference)', MoG_runtime);
% MoDT
for ii = 1:nCases
    fprintf('%-*s  %+12.4f  %12.2f\n', descLen, ...
        modt_desc{ii}, modt_loglike(ii)-MoG_loglike, modt_runtime(ii));
end



