function [Priors,Mu,Sigma] = maximization_step(X, Pk_x, params)
%MAXIMISATION_STEP Compute the maximization step of the EM algorithm
%   input------------------------------------------------------------------
%       o X         : (N x M), a data set with M samples each being of
%       dimension
%       o Pk_x      : (K, M) a KxM matrix containing the posterior probabilty
%                     that a k Gaussian is responsible for generating a point
%                     m in the dataset, output of the expectation step
%       o params    : The hyperparameters structure that contains k, the number of Gaussians
%                     and cov_type the coviariance type
%   output ----------------------------------------------------------------
%       o Priors    : (1 x K), the set of updated priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu        : (N x K), an NxK matrix corresponding to the updated centroids 
%                           mu = {mu^1,...mu^K}
%       o Sigma     : (N x N x K), an NxNxK matrix corresponding to the
%                   updated Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%%
% 
% [N,M] = size(X);
% K = params.k;
% Mu = zeros(N,K);
% Priors = zeros(1,K);
% Sigma = zeros(N,N,K);
% 
% summ = sum(Pk_x,2);
% 
% Priors = (1/M*(summ));
% Priors = Priors.';
% 
% 
% for k=1:K
%     Mu(:,k) = Pk_x(k,:)*X'/sum(Pk_x(k,:),2);
%     
%     comp = 0;
%     if (params.cov_type == "full")
%         for i=1:M
%                 comp = comp + Pk_x(k,i)*((X(:,i)-Mu(:,k))*(X(:,i)-Mu(:,k))');
%                 Sigma(:,:,k) = comp/sum(Pk_x(k,:),2);
%         end
%     elseif (params.cov_type == "diag")
%         for i=1:M
%             comp = comp + Pk_x(k,i)*((X(:,i)-Mu(:,k))*(X(:,i)-Mu(:,k))');
%             Sigma(:,:,k) = diag(diag(comp/sum(Pk_x(k,:),2)));
%         end
%     elseif (params.cov_type == "iso")
%         for i=1:M
%             comp = comp + Pk_x(k,i)*(norm(X(:,i)-Mu(:,k)))^2;
%             Sigma(:,:,k) = diag(comp/(N * sum(Pk_x(k,:),2))*ones(N,1));
%         end
%     end
%    Sigma(:,:,k) = Sigma(:,:,k) + diag(1e-5*ones(N,1));
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Priors = 1/size(X,2) *sum(Pk_x,2)';
Mu=zeros(size(X,1),length(Priors));
Sigma=zeros(size(X,1),size(X,1),length(Priors));
S=0;
for i=1:length(Priors)
    for j=1:size(X,2)
        Mu(:,i) = Mu(:,i)+(Pk_x(i,j).*X(:,j));
    end
    Mu(:,i)=Mu(:,i)/sum(Pk_x(i,:),2);
end

switch params.cov_type
    case 'full'
        for i =1:length(Priors)
            for j=1:size(X,2)
              Sigma(:,:,i)= Sigma(:,:,i) +(Pk_x(i,j)*(X(:,j)-Mu(:,i))*(X(:,j)-Mu(:,i))');
            end
            Sigma(:,:,i)=Sigma(:,:,i)/sum(Pk_x(i,:),2);
        end
    case 'diag'
        for i =1:length(Priors)
            for j=1:size(X,2)
              Sigma(:,:,i)= Sigma(:,:,i) +(Pk_x(i,j).*(X(:,j)-Mu(:,i))*(X(:,j)-Mu(:,i))');
            end
            Sigma(:,:,i)=diag(diag(Sigma(:,:,i)/sum(Pk_x(i,:),2)));
        end
        
    case 'iso'
        for i =1:length(Priors)
            for j=1:size(X,2)
              S= S +(Pk_x(i,j)*norm(X(:,j)-Mu(:,i))^2);
            end
           S=S/(size(X,1)*sum(Pk_x(i,:),2));
            Sigma(:,:,i)=diag(repmat(S,size(X,1),1));
        end
%         Sigma
        
end
% params.cov_type
% size(Sigma)
e=diag(1e-5*ones(1,size(Sigma,1)));
size(e);
Sigma=Sigma+e;

end

