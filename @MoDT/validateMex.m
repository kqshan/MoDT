function validateMex()
% Run unit tests on the MEX routines to make sure they are working properly
%   validateMex()
%
% This compares the output of the MEX routines to the output of the non-MEX
% "fallback" functions (which are written in plain MATLAB). Since the MEX file
% takes precedence, we copy these functions to the MoDT class's parent directory
% so that we can call them explicitly. 

% We will throw an error at the very end if there was a problem
errid_pfx = 'MoDT:validateMex';
errid = [errid_pfx ':ValidationFailed'];
errmsg = '';

% Check the normal MEX files
dne_msg = '%s : MEX file does note exist; cannot validate\n';
% sumFrames
if mexFileExists('sumFrames')
    cleanup = copyNonMexFile('sumFrames');
    new_err = unitTest_sumFrames();
    errmsg = [errmsg, new_err];
    delete(cleanup);
else
    errmsg = [errmsg, sprintf(dne_msg,'sumFrames')];
end
% bandPosSolve
if mexFileExists('bandPosSolve')
    cleanup = copyNonMexFile('bandPosSolve');
    new_err = unitTest_bandPosSolve();
    errmsg = [errmsg, new_err];
    delete(cleanup);
else
    errmsg = [errmsg, sprintf(dne_msg,'sumFrames')];
end

% See if there is a GPU installed
gpu_ok = false;
gpu_supports_double = false;
try
    gpu_device = gpuDevice();
    gpu_supports_double = gpu_device.SupportsDouble;
catch exc
    if ~strcmp(exc.identifier,'parallel:gpu:device:DriverRedirect')
        rethrow(exc);
    end
end
% Skip the GPU MEX tests if no appropriate GPU is installed
if ~gpu_ok
    warning([errid_pfx ':NoGPU'], 'No GPU installed; skipping GPU MEX tests');
elseif ~gpu_supports_double
    warning([errid_pfx ':NoGPU'], ['GPU does not support doubles; ' ...
        'skipping GPU MEX tests']);
else
    % Run the GPU MEX unit tests
    % weightCovGpu
    % calcMahalDistGpu
    % sumFramesGpu
end

% Throw an error if there was a problem
fprintf([repmat('-',[1 72]) '\n']);
if isempty(errmsg)
    fprintf('MEX validation completed successfully\n');
else
    errmsg = [sprintf('Validation failed on the following:\n'), errmsg];
    error(errid, errmsg);
end
end


% -------------------------     Helper functions     ---------------------------


function tf = mexFileExists( funcName )
% Check that a MEX file exists for this function on this platform
%   tf = mexFileExists( funcName )
% Returns:
%   tf          Whether a MEX file exists for this function on this platform
% Required arguments:
%   funcName    Name of the MoDT static method
class_dir = fileparts(mfilename('fullpath'));
file_name = fullfile(class_dir, strcat(funcName, '.', mexext()));
tf = ( exist(file_name,'file')==3 );
end

function deleter = copyNonMexFile( funcName )
% Temporarily copy a non-MEX MoDT method to the parent directory
%   deleter = copyNonMexFile( funcName )
% Returns:
%   deleter     onCleanup object that deletes the temporary file
% Required arguments:
%   funcName    Name of the MoDT static method to copy 
class_dir = fileparts(mfilename('fullpath'));
parent_dir = fileparts(class_dir);
funcFile = strcat(funcName,'.m');
srcFile = fullfile(class_dir, funcFile);
tgtFile = fullfile(parent_dir, funcFile);
% Check if a file of the same name already exists in the target directory
errid = 'MoDT:validateMex:ValidationAborted';
assert(~exist(tgtFile,'file'), errid, ['Could not copy "%s" to parent ' ...
    'directory; a file with the same name already exists'], funcFile);
% Copy the file
copyfile(srcFile, parent_dir);
% Create the deleter
deleter = onCleanup(@() delete(tgtFile));
% Double-check that this is the one that will be called
calledFile = which(funcName);
assert(strcmp(tgtFile, calledFile), errid, ...
    ['The following file on the MATLAB search path\n' ...
    'is interfering with the unit test for %s():\n%s'], funcName, calledFile);
end


% ---------------------------     Unit tests     -------------------------------


function errmsg = unitTest_sumFrames()
% Unit test for the sumFrames() method
%   errmsg = unitTest_sumFrames()
fprintf([repmat('-',[1 72]) '\n']);
fprintf('sumFrames()\n');

% Select some dimensions that cover some potential edge cases
% Start with a typical case
DNKT_typ = [12 5e5 15 480];
% Try setting various combinations of dimensions to 1
DNKT = repmat(DNKT_typ,[32 1]);
mask = bsxfun(@ge,bsxfun(@mod,(0:31)',pow2(1:4)),pow2(0:3));
DNKT(mask) = 1;
% Add on a few random cases
m = 32;
DNKT = [ DNKT ; 
    randi(36,[m 1]), randi(1e4,[m 1]), randi(80,[m 1]), randi(1600,[m 1])];
% Make it into a cell array so we can do deal(DNKT{:})
DNKT = num2cell(DNKT);
nTests = size(DNKT,1);

% Start with no errors
errmsg = '';

% 1. It should give exact results on integer values
fprintf('%-68s','1. Testing for exact results with integer inputs');
failed = false;
for ii = 1:nTests
    [D,N,K,T] = deal(DNKT{ii,:});
    % Generate random inputs
    Y = round(randn(D,N)*10);
    wzu = round(rand(N,K)*10);
    frame_edge = sort(randi(N+1,[T-1, 1]))-1;
    f_spkLim = [[1; frame_edge+1], [frame_edge; N]];
    % Call the non-MEX version to get the correct answer
    [wzuY_ref, sum_wzu_ref] = sumFrames(Y, wzu, f_spkLim);
    % Call the MEX version
    [wzuY, sum_wzu] = MoDT.sumFrames(Y, wzu, f_spkLim);
    % Compare the outputs
    err_1 = max(abs(wzuY(:) - wzuY_ref(:)));
    err_2 = max(abs(sum_wzu(:) - sum_wzu_ref(:)));
    if (err_1 > 0) || (err_2 > 0)
        if ~failed
            fprintf('FAIL\n');
            failed = true;
            fprintf('    Failed on the following problem:\n');
            fprintf('    %21s | %18s\n', 'Dimensions      ','Error      ');
            fprintf('    %2s  %7s  %2s  %4s | %8s  %8s\n', ...
                'D','N','K','T','wzuY','sum_wzu');
        end
        fprintf('    %2d  %7d  %2d  %4d | %8.3g  %8.3g\n', D,N,K,T,err_1,err_2);
    end
end
if failed
    errmsg = sprintf('%s%s\n', errmsg, ...
        'sumFrames : Failed to produce exact results with integer inputs');
else
    fprintf('PASS\n');
end

% 2. It should give approximate results for double-precision floating point
fprintf('%-68s','2. Testing on double-precision floating-point values');
failed = false;
rel_err_thresh = 1000 * eps;
for ii = 1:nTests
    [D,N,K,T] = deal(DNKT{ii,:});
    % Generate random inputs
    Y = randn(D,N) + 2;
    wzu = rand(N,K);
    frame_edge = sort(randi(N+1,[T-1, 1]))-1;
    f_spkLim = [[1; frame_edge+1], [frame_edge; N]];
    % Call the non-MEX version to get the correct answer
    [wzuY_ref, sum_wzu_ref] = sumFrames(Y, wzu, f_spkLim);
    % Call the MEX version
    [wzuY, sum_wzu] = MoDT.sumFrames(Y, wzu, f_spkLim);
    % Compare the outputs
    abs_err_1 = abs(wzuY(:,:) - wzuY_ref(:,:));
    norm_1 = sqrt(sum(wzuY_ref(:,:).^2,1));
    err_1 = max(max(bsxfun(@rdivide, abs_err_1, norm_1)));
    err_2 = max(abs(sum_wzu(:) - sum_wzu_ref(:)) ./ abs(sum_wzu_ref(:)));
    if (err_1 > rel_err_thresh) || (err_2 > rel_err_thresh)
        if ~failed
            fprintf('FAIL\n');
            failed = true;
            fprintf('    Failed on the following problem:\n');
            fprintf('    %21s | %18s\n', 'Dimensions      ','Relative error  ');
            fprintf('    %2s  %7s  %2s  %4s | %8s  %8s\n', ...
                'D','N','K','T','wzuY','sum_wzu');
        end
        fprintf('    %2d  %7d  %2d  %4d | %8.3g  %8.3g\n', D,N,K,T,err_1,err_2);
    end
end
if failed
    errmsg = sprintf('%s%s\n', errmsg, ...
        'sumFrames : Excessive error on floating-point values');
else
    fprintf('PASS\n');
end

% 3. Let's measure the speedup on a slightly larger problem
fprintf('%s','3. Measuring MEX file speedup: ');
% Generate random inputs
[D,N,K,T] = deal(12,1e6,20,720);
Y = randn(D,N) + 2;
wzu = rand(N,K);
frame_edge = sort(randi(N+1,[T-1, 1]))-1;
f_spkLim = [[1; frame_edge+1], [frame_edge; N]];
% Time it
t_base = timeit(@() sumFrames(Y,wzu,f_spkLim), 2);
t_mex = timeit(@() MoDT.sumFrames(Y,wzu,f_spkLim), 2);
speedup = (t_base/t_mex) - 1;
fprintf('%.0f%% (%.1f ms m-file vs. %.1f ms MEX)\n', ...
    speedup*100, t_base*1e3, t_mex*1e3);
end


function errmsg = unitTest_bandPosSolve()
% Unit test for the bandPosSolve() method
%   errmsg = unitTest_bandPosSolve()
fprintf([repmat('-',[1 72]) '\n']);
fprintf('bandPosSolve()\n');

% Select some dimensions that cover some potential edge cases
Nqm = [ ...
    5760  13  1;   % Typical case (D=12, T=480, diagonal Qinv)
    5760  24  1;   % Make sure we can handle full Qinv
      20  20  1;   % What if A is full
       1   1  1;   % Or if it's scalar
    ];
% We never call it with multiple b vectors (m > 1), but it should still work
Nqm = [Nqm; bsxfun(@plus, Nqm, [0 0 7])];
% Make it into a cell array so we can do deal(Nqm{:})
Nqm = num2cell(Nqm);
nTests = size(Nqm,1);

% Start with no errors
errmsg = '';

% 1. It should give approximate results for double-precision floating point
fprintf('%-68s','1. Testing on double-precision floating-point values');
failed = false;
rel_err_thresh = 1000 * eps;
for ii = 1:nTests
    [N,q,m] = deal(Nqm{ii,:});
    % Construct a random positive definite matrix via its Cholesky factorization
    L = spdiags(randn(N,q), -q+1:0, N, N);
    A = L*L' + 0.01*speye(N);
    % Get its banded representation
    p = 2*q-1;
    A_bands = zeros(p,N);
    [i,j,v] = find(A);
    A_bands(i-j+q + p*(j-1)) = v;
    % Generate a random x and get b = A*x
    x = randn(N,m);
    b = A * x;
    % Call the routine under test
    x_mex = MoDT.bandPosSolve(A_bands, b);
    % Compare the outputs
    norm_x = sqrt(sum(x.^2,1));
    err_mex = max(max(abs(x_mex(:) - x(:)),[],1)./norm_x);
    if (err_mex > rel_err_thresh)
        if ~failed
            fprintf('FAIL\n');
            failed = true;
            fprintf('    Failed on the following problem:\n');
            fprintf('    %12s | %8s\n', 'Dimensions ','Rel error');
            fprintf('    %5s %3s %2s | %8s\n', 'N','p','m','x');
        end
        fprintf('    %5d %3d %2d | %8.3g\n',N,p,m,err_mex);
    end
end
if failed
    errmsg = sprintf('%s%s\n', errmsg, ...
        'bandPosSolve : Excessive error on floating-point values');
else
    fprintf('PASS\n');
end

% 2. Let's measure the speedup on a slightly larger problem
fprintf('%s','2. Measuring MEX file speedup: ');
% Generate random inputs
[N,q,m] = deal(8640, 24, 1);
L = spdiags(randn(N,q), -q+1:0, N, N);
A = L*L' + 1e-2*speye(N);
p = 2*q-1;
A_bands = zeros(p,N);
[i,j,v] = find(A);
A_bands(i-j+q + p*(j-1)) = v;
x = randn(N,m);
b = A * x;
% Time it
t_ref = timeit(@() bandPosSolve(A_bands,b), 1);
t_mex = timeit(@() MoDT.bandPosSolve(A_bands,b), 1);
speedup = (t_ref/t_mex) - 1;
fprintf('%.0f%% (%.1f ms m-file vs. %.1f ms MEX)\n', ...
    speedup*100, t_ref*1e3, t_mex*1e3);
end




