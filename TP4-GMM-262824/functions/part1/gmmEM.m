function [  Priors, Mu, Sigma, iter ] = gmmEM(X, params)
%MY_GMMEM Computes maximum likelihood estimate of the parameters for the 
% given GMM using the EM algorithm and initial parameters
%   input------------------------------------------------------------------
%       o X         : (N x M), a data set with M samples each being of 
%                           dimension N, each column corresponds to a datapoint.
%       o params : Structure containing the paramaters of the algorithm:
%           * cov_type: Type of the covariance matric among 'full', 'iso',
%           'diag'
%           * k: Number of gaussians
%           * max_iter: Max number of iterations

%   output ----------------------------------------------------------------
%       o Priors    : (1 x K), the set of FINAL priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu        : (N x K), an NxK matrix corresponding to the FINAL centroids 
%                           mu = {mu^1,...mu^K}
%       o Sigma     : (N x N x K), an NxNxK matrix corresponding to the
%                   FINAL Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%       o iter      : (1 x 1) number of iterations it took to converge
%%

[N,M] = size(X);
K = params.k;
max_iter = params.max_iter;

Priors = zeros(1,K);
Mu = zeros(N,K);
i = 1;
Sigma = zeros(N,N,K);

[Priors, Mu, Sigma, labels0] = gmmInit(X, params);
logl = zeros(max_iter);
logl_iter = 0;
max_logl_iter = 8;


while (i<=max_iter)
   
   prev = gmmLogLik(X, Priors, Mu, Sigma);
   [Pk_x] = expectation_step(X, Priors, Mu, Sigma, params);
   [Priors, Mu, Sigma] = maximization_step(X, Pk_x, params);
   logl = gmmLogLik(X, Priors.', Mu, Sigma);
   if (norm(logl-prev)<1e-5)
       break
   end
   i = i + 1;

end

iter = i ;

end

