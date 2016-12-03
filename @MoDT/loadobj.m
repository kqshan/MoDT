function obj = loadobj(s)
% Create an MoDT object based on a struct created by saveobj()
%   obj = loadobj(s)
%
% This method is called by MATLAB when loading an object from a .mat
% file, but you can also use this for your own serialization purposes.
% See also: MoDT/saveobj

% Get a list of all of the fieldnames
struct_fields = fieldnames(s);

% Start with a default initialization
obj = MoDT();

% Assign parameters using the appropriate functions
% Call setParams for everything it can handle
param_fields = obj.setParams();
params = rmfield(s, setdiff(struct_fields,param_fields));
obj.setParams(params);
% Call attachData for the data
data_fields = {'spk_Y';'spk_t';'spk_w'};
if ~isempty(s.spk_t), obj.attachData(s.spk_Y, s.spk_t, s.spk_w); end

% Error checks
% Check the dimensions
dim_fields = {'D';'K';'T';'N'};
for fn = dim_fields'
    assert(isequaln(obj.(fn{1}),s.(fn{1})), MoDT.badDimErrId, ...
        'Dimension dismatch while loading object');
end
% Make sure we got everything
obj_fields = [param_fields; data_fields; dim_fields];
assert(isempty(setxor(struct_fields,obj_fields)), ...
    'MoDT:loadobj:BadFields', ...
    'Fields provided to loadobj() do not match the fields we expected');

end
