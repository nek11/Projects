function [S] = gower_similarity(X1,X2,data_type, rk)
%GOWER_SIMILARITY Compute the Gower similarity between X1 and X2
%
%   input -----------------------------------------------------------------
%       o X1 : (N x 1) First sample point
%       o X2 : (N x 1) Second sample point
%       o data_type : {N x 1}, a boolean cell array with true when
%                     feature is continuous
%       o rk : (N x 1) The range of values for continuous data 
%
%   output ----------------------------------------------------------------
%      o S : The similarity measure between two samples

N=length(X1);
Sk=zeros(N,1);

for i=1:N
    if ~data_type{i}
        if (X1(i)==X2(i))
            Sk(i)=1;
        else
            Sk(i)=0;
        end
    else 
       cont = abs(X1(i)-X2(i))/rk(i);
       Sk(i)=1-(cont);
    end
end

S=sum(Sk)/N;




end


