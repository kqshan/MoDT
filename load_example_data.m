function example_data = load_example_data( varargin )
% Load example data for testing the MoDT class. Data is downloaded if necessary.
%   example_data = load_example_data( ... )
%
% Returns:
%   example_data    Struct with fields:
%     spk_Y           [D x N] spikes (N spikes in a D-dimensional feature space)
%     spk_t           [N x 1] spike times (ms)
%     spk_clustId     [N x 1] cluster ID this spike is assigned to
%     weighting       [N x 1] suggested weighting for training on a subset
%
% Optional parameters (key/value pairs) [default]:
%   prompt_user     Prompt user before downloading              [ true ]
%   force_download  Download even if the file already exists    [ false ]
%   url             Download URL                                [ see below ]
%   local_filename  Local filename for saving the file          [ see below ]
%
% The default URL is http://www.its.caltech.edu/~kshan/example_spikes.mat, and
% the default local filename is example_spikes.mat in the same directory as this
% function.
%
% This example data contains 1.9 million spikes acquired from a single tetrode
% over the course of 5.3 hours. For more details, see:
%   [ insert citation here after publication ]

errid_pfx = 'load_example_data';

% Parse optional parameters
ip = inputParser();
ip.addParameter('prompt_user', true, @isscalar);
ip.addParameter('force_download', false, @isscalar);
ip.addParameter('url', ...
    'http://www.its.caltech.edu/~kshan/example_spikes.mat', ...
    @ischar);
func_dir = fileparts(mfilename('fullpath'));
ip.addParameter('local_filename', ...
    fullfile(func_dir, 'example_spikes.mat'), ...
    @ischar);
ip.parse( varargin{:} );
prm = ip.Results;

% Check if the file already exists
fname = prm.local_filename;
file_exists = (exist(fname,'file') == 2);
% Download if the file does not exist, or force_download==true
if ~file_exists || prm.force_download
    % Ask the user before continuing
    if prm.prompt_user
        disp('This is about to download a 43 MB file from www.caltech.edu.');
        reply = input('Do you wish to continue? Y/N [N]:','s');
        if isempty(reply) || ~strncmpi(reply,'y',1)
            disp('Download aborted.');
            error([errid_pfx ':NoDownload'], 'Download aborted');
        end
    end
    % Download the file
    websave(fname, prm.url);
end

% Load the file from disk
web_data = load(fname);

% I have a 128 MB limit on personal Caltech-hosted web space, so I had to do a
% few type conversions to make sure this data would fall below this limit.

% Spike data: convert from int16 back to double, add random noise to remove
% quantization artifacts, and scale it to the original feature space (which was
% microvolts RMS).
spk_Y = double(web_data.spike_features_scaled100);
uniform_noise = rand(size(spk_Y)) - 0.5; % noise ranges from -.5 to +.5
spk_Y = (spk_Y + uniform_noise) / 100;
% Also, transpose to be [D x N]
spk_Y = spk_Y';

% Spike times: Go from index diff to indices, then convert to time (ms)
sample_rate = 50; % kHz
spk_t = cumsum(double(web_data.spike_dataIdx50kHz_diff)) / sample_rate;

% Cluster assignment: Type conversion only
spk_clustId = double(web_data.spike_clusterId);

% Spike weighting: I stored it as numerator/denominator because these would
% always be integers due to the way the weighting was performed.
weighting = double(web_data.spike_weights_numerator) ...
    ./ double(web_data.spike_weights_denominator);

% Package everything into a struct
example_data = struct('spk_Y',spk_Y, 'spk_t',spk_t, ...
    'spk_clustId',spk_clustId, 'weighting',weighting);

end
