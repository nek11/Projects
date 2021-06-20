% Classification : Scaling of input data
%
% [data_scaled, dataMax, dataMin] = classificationScaling(data, dataMax, dataMin, typeNorm) 
%
% type 'minmax' (default):
%     data scaled = (data - min of each dimension) / max of each dimension
% type 'std':
%     data scaled = (data - mean of each dimension) / std of each dimension
%
% Frank de Morsier, frank.demorsier@epfl.ch

function [data_scaled, dataMax, dataMin] = classificationScaling(data, dataMax, dataMin, typeNorm) 
if nargin<4, typeNorm='minmax'; end;
if nargin<3 || isempty(dataMin), dMinSet=1; else dMinSet=0; end;
if nargin<2 || isempty(dataMax), dMaxSet=1; else dMaxSet=0; end;
% Scaling
% data = data';
%data_scaled = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));
% data_scaled = data_scaled';
% normalize
%data_scaled = data./(ones(size(data,1),1)*(sqrt(diag(data'*data)))');

switch typeNorm
    case 'minmax'
if dMinSet==1, dataMin = min(data,[],1); end;
data=data-repmat(dataMin,[size(data,1),1]);

if dMaxSet==1, dataMax = max(data,[],1); end;
data_scaled=data./repmat(dataMax,[size(data,1),1]);

    case 'std'
if dMinSet==1, dataMin = mean(data,1); end;
data=data-repmat(dataMin,[size(data,1),1]);

if dMaxSet==1, dataMax = std(data,[],1); end;
data_scaled=data./repmat(dataMax,[size(data,1),1]);
end

end
