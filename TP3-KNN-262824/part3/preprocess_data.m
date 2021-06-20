function [X, Y, rk] = preprocess_data(table_data, ratio, data_type)
%PREPROCESS_DATA Preprocess the data in the adult dataset
%
%   input -----------------------------------------------------------------
%
%       o table_data    : (M x N), a cell array containing mixed
%                         categorical and continuous values
%       o ratio : (float) The pourcentage of M samples to extract
%       o data_type : (N x 1) boolean array, true if Ni is continuous
%
%   output ----------------------------------------------------------------
%       o X : (N-1, M*ratio) Data extracted from the table where
%             categorical values are converted to integer values
%       o Y : (1, M*ratio) The label of the data to classify. Values are 1
%             or 2
%       o rk : (N x 1) The range of values for continuous data (will be 0
%               if the data are categorical)


[M,N]=size(table_data);
M_ratio = floor(M*ratio);
data=table_data(randperm(M,M_ratio),:);
X=zeros(N,M_ratio);
Y=zeros(1,M_ratio);
rk=zeros(N,1);

for i=1:N
    if ~data_type{i}
        [X(i,:),~]=grp2idx(data{:,i});

    else
        X(i,:)=double(data{:,i});
        rk(i)=range(data{:,i});      
    end
end

Y=X(N,:);
X(N,:)=[];







end

