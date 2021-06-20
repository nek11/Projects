clear; 
close all; 
clc;

addpath(genpath("functions/part1"))
addpath(genpath("evaluation_functions"))
addpath(genpath("evaluation_functions/part1"))
addpath(genpath("plot_functions"))

dataset_path = './data';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         1) Load 2D KMEAN Function Testing Dataset          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1a) Load 2D data sampled from a GMM

% Load Dataset
load(strcat(dataset_path,'/2d-gmm-4.mat'))

% Visualize Dataset
plot_2d_data(X, gmm)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%           Task 1: Test kmeans_init.m function              %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Default Init Values to Check kmeans_init.m
K = 4; init = 'sample';
Mu =  kmeans_init(X, K, init);

% Visualize Initialized Centroids
if exist('h1','var') && isvalid(h1), delete(h1);end
options.title       = sprintf('Centroid Initialization: %s', init);
h1 = ml_plot_data(X',options); hold on;
colors     = hsv(K);
ml_plot_centroids(Mu',colors);

% Test kmeans_init.m function for both 'random' and 'uniform'
evaluate_kmeans_initialization(X, K);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%           Task 2: compute_distance.m function              %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
evaluate_distance_computation(X);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         Task 3: distance_to_centroids.m function           %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
evaluate_distance_to_centroids(X, Mu);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         Task 4: check_convergence.m function               %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
evaluate_convergence();

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         Task 5: kmeans.m function                          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = 4; init='range'; type='L1'; MaxIter = 100; plot_iter = 1;
[labels, Mu, ~] =  kmeans(X, K, init, type, MaxIter, plot_iter);
pause(1);

evaluate_kmeans(X, K, init, type, MaxIter);

% plot
plot_boundaries(X, Mu, labels, K, type)
