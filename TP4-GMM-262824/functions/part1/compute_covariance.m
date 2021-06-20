function [ Sigma ] = compute_covariance( X, X_bar, type )
%MY_COVARIANCE computes the covariance matrix of X given a covariance type.
%
% Inputs -----------------------------------------------------------------
%       o X     : (N x M), a data set with M samples each being of dimension N.
%                          each column corresponds to a datapoint
%       o X_bar : (N x 1), an Nx1 matrix corresponding to mean of data X
%       o type  : string , type={'full', 'diag', 'iso'} of Covariance matrix
%
% Outputs ----------------------------------------------------------------
%       o Sigma : (N x N), an NxN matrix representing the covariance matrix of the 
%                          Gaussian function
%%

[N,M] = size(X);
Sigma = zeros(N,N);
sum = 0;

switch type
    case "full"
        X = X-X_bar;
        Sigma = (1/(M-1))*X*X.';
        
    case "diag"
        sigma_x = std(X(1,:));
        sigma_y = std(X(2,:));
        Sigma(1,1) = sigma_x.^2;
        Sigma(2,2) = sigma_y.^2;
        Sigma(1,2) = 0;
        Sigma(2,1) = 0;
        
    case "iso"
        for i=1:M
            sum = sum + (norm(X(:,i)-X_bar)).^2;
            sigma_iso = (1/(N*M))*sum;
        end
        Sigma(1,1) = sigma_iso;
        Sigma(2,2) = sigma_iso;
        Sigma(1,2) = 0;
        Sigma(2,1) = 0;        
end



end

