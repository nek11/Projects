function [Mu, C, EigenVectors, EigenValues] = compute_pca(X)
%COMPUTE_PCA Step-by-step implementation of Principal Component Analysis
%   In this function, the student should implement the Principal Component 
%   Algorithm
%
%   input -----------------------------------------------------------------
%   
%       o X      : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%
%   output ----------------------------------------------------------------
%
%       o Mu              : (N x 1), Mean Vector of Dataset
%       o C               : (N x N), Covariance matrix of the dataset
%       o EigenVectors    : (N x N), Eigenvectors of Covariance Matrix.
%       o EigenValues     : (N x 1), Eigenvalues of Covariance Matrix

Dimension_matrix = size(X);
N = Dimension_matrix(1,1);
M = Dimension_matrix(1,2);
Mean = mean(X, 2);
Mu = Mean;

X = X - Mu;

C = (1/(M-1))*(X*X.');

[EigenVectors, EigenValues] = svd(C);
EigenValues = diag(EigenValues);


end

