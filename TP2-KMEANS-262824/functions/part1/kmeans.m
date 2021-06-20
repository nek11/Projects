function [labels, Mu, Mu_init, iter] =  kmeans(X,K,init,type,MaxIter,plot_iter)
%MY_KMEANS Implementation of the k-means algorithm
%   for clustering.
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o K        : (int), chosen K clusters
%       o init     : (string), type of initialization {'sample','range'}
%       o type     : (string), type of distance {'L1','L2','LInf'}
%       o MaxIter  : (int), maximum number of iterations
%       o plot_iter: (bool), boolean to plot iterations or not (only works with 2d)
%
%   output ----------------------------------------------------------------
%
%       o labels   : (1 x M), a vector with predicted labels labels \in {1,..,k} 
%                   corresponding to the k-clusters for each points.
%       o Mu       : (N x k), an Nxk matrix where the k-th column corresponds
%                          to the k-th centroid mu_k \in R^N 
%       o Mu_init  : (N x k), same as above, corresponds to the centroids used
%                            to initialize the algorithm
%       o iter     : (int), iteration where algorithm stopped
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Auxiliary Variable
if plot_iter == [];plot_iter = 0;end


% INSERT CODE HERE

D = size(X,1);
M = size(X,2);
Mu_init = kmeans_init(X, K, init);
d = distance_to_centroids(X, Mu_init, type);
labels = zeros(1,M);
data_clusters = zeros(K,1);
r = zeros(K,M);

%%%%%%%%%%%%%%%%%         TEMPLATE CODE      %%%%%%%%%%%%%%%%
% Visualize Initial Centroids if N=2 and plot_iter active
colors     = hsv(K);
if (D==2 && plot_iter)
    options.title       = sprintf('Initial Mu with %s method', init);
    ml_plot_data(X',options); hold on;
    ml_plot_centroids(Mu_init',colors);
end


%%%%% K-Means parameters %%%%%
iter      = 0;
tolerance = 1e-6;
tol_iter = 0;
Mu = Mu_init;
has_converged = false;
MaxTolIter = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while ~has_converged
    
    if (K == 1)
        labels(:) = 1;
    else
        [~,labels(:)]=(min(d(:, :)));
    end
    
    % INSERT CODE HERE
    for j=1:K      
        for i=1:M
            if (labels(i) == j) 
                r(j,i) = 1;
                data_clusters(j) = data_clusters(j)+1;          %count the number of datapoints assigned to a cluster
            else
                r(j,i) = 0;
            end
            Mu_previous = Mu;
            A(:,i) = r(j,i).*X(:,i);
        end
        num = sum(A,2);
        denum = sum((r(j,:)),2);
        Mu(:,j) = num/denum;
    end
    
    
    if (all(data_clusters))
        iter = iter + 1;
        [has_converged, tol_iter] = check_convergence(Mu, Mu_previous, iter, tol_iter, MaxIter, MaxTolIter, tolerance);
    else
        Mu = kmeans_init(X, K, init);
        iter = 0;
        tol_iter = 0;
    end
    
    d = distance_to_centroids(X, Mu, type);
    
    k_i = labels;    
    
        
    %%%%%%%%%%%%%%%%%         TEMPLATE CODE      %%%%%%%%%%%%%%%%       
    if (D==2 && iter == 1 && plot_iter)
        options.labels      = k_i;
        options.title       = sprintf('Mu and labels after 1st iter');
        ml_plot_data(X',options); hold on;
        ml_plot_centroids(Mu',colors);
    end    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    
     % INSERT CODE HERE
     

end


% INSERT CODE HERE





%%%%%%%%%%%   TEMPLATE CODE %%%%%%%%%%%%%%%
if (D==2 && plot_iter)
    options.labels      = labels;
    options.class_names = {};
    options.title       = sprintf('Mu and labels after %d iter', iter);
    ml_plot_data(X',options); hold on;    
    ml_plot_centroids(Mu',colors);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end