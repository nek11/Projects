function [ y_est ] =  knn(X_train,  y_train, X_test, params)
%MY_KNN Implementation of the k-nearest neighbor algorithm
%   for classification.
%
%   input -----------------------------------------------------------------
%   
%       o X_train  : (N x M_train), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_train  : (1 x M_train), a vector with labels y \in {1,2} corresponding to X_train.
%       o X_test   : (N x M_test), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o params : struct array containing the parameters of the KNN (k,
%                  d_type and eventually the parameters for the Gower
%                  similarity measure)
%
%   output ----------------------------------------------------------------
%
%       o y_est   : (1 x M_test), a vector with estimated labels y \in {1,2} 
%                   corresponding to X_test.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[N, M_test] = size(X_test);
M_train = size(X_train, 2);
y_est = zeros(1, M_test);

k = params.k;
%d = zeros(M_train, M_test);


for i=1:M_test
   for j=1:M_train
      d(j,i) = compute_distance(X_test(:,i), X_train(:,j), params);
   end
   distances = d(:,i);
   [~, idx] = sort(distances);
   labels = idx(1:k);
   labels_2 = y_train(labels);
   y_est(1,i) = mode(labels_2);
   
end




end