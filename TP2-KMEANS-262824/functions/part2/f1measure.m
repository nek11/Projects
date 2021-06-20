function [F1_overall, P, R, F1] =  f1measure(cluster_labels, class_labels)
%MY_F1MEASURE Computes the f1-measure for semi-supervised clustering
%
%   input -----------------------------------------------------------------
%   
%       o class_labels     : (1 x M),  M-dimensional vector with true class
%                                       labels for each data point
%       o cluster_labels   : (1 x M),  M-dimensional vector with predicted 
%                                       cluster labels for each data point
%   output ----------------------------------------------------------------
%
%       o F1_overall      : (1 x 1)     f1-measure for the clustered labels
%       o P               : (nClusters x nClasses)  Precision values
%       o R               : (nClusters x nClasses)  Recall values
%       o F1              : (nClusters x nClasses)  F1 values
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Clusters = unique(cluster_labels);
Classes = unique(class_labels);
K = size(Clusters,1);
J = size(Classes,1);
n=zeros(K,J);

for i=1:K
    for j = 1:J
        fnd = find(class_labels.'==Classes(j));
        for k =1:length(fnd)
            if(cluster_labels(fnd(k)).'==Clusters(i))
                n(i,j)=n(i,j)+1;
            end
        end
    end
end

c=sum(class_labels.'==Classes,2);
k=sum(cluster_labels.'==Clusters,2);

for i = 1:size(n,1)
    P(i,:)=n(i,:)./abs(k(i));
    for j=1:size(n,2)
        R(:,j)=n(:,j)./abs(c(j));
        if (n(i,j)~=0)
            F1(i,j)= 2.*R(i,j)*P(i,j)/(R(i,j)+P(i,j));
        end
    end
end

for i=1:length(c)
    F1_overall_tab(i)= (c(i)./size(class_labels.',2) * max(F1(:,i)));
end

F1_overall = sum(F1_overall_tab);

end
