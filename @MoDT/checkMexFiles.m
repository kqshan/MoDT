function [mex_exists, gpu_mex_exists] = checkMexFiles()
% Check whether compiled subroutines (MEX files) exist for this platform
%   [mex_exists, gpu_mex_exists] = checkMexFiles()
%
% Returns:
%   mex_exists      All expected (non-GPU) MEX files exist (logical)
%   gpu_mex_exists  All expected GPU-based MEX files exist (logical)
%
% This checks for the following MEX files:
%   non-gpu: sumFrames, bandposSolve
%   gpu    : weightCovGpu, calcMahalDistGpu, sumFramesGpu

% Here are the functions to check
non_gpu_list = {'sumFrames','bandPosSolve'};
gpu_list = {'weightCovGpu','calcMahalDistGpu','sumFramesGpu'};

% Get the path
class_dir = fileparts(mfilename('fullpath'));
% Get the MEX file extension for this platform
ext = mexext();

% Check that the non-GPU MEX files exist
file_list = fullfile(class_dir, strcat(non_gpu_list,'.',ext));
file_exists = cellfun(@(x) exist(x,'file')==3, file_list);
mex_exists = all(file_exists);

% Check that the GPU MEX files exist
file_list = fullfile(class_dir, strcat(gpu_list,'.',ext));
file_exists = cellfun(@(x) exist(x,'file')==3, file_list);
gpu_mex_exists = all(file_exists);

end
