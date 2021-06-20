addpath(genpath("functions/part1"))
addpath(genpath("functions/part2"))
addpath(genpath("evaluation_functions"))
addpath(genpath("evaluation_functions/part2"))
addpath(genpath("plot_functions"))

clear; 
close all; 
clc;

warning('off', 'all');

rng(42);
seed = rng;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              1) Load Digits Testing Dataset                %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
true_K = 4;
[X, labels] = ml_load_digits_64('data/digits.csv', 0:true_K-1);

% Generate Variables
[M, N]  = size(X);
sizeIm  = sqrt(N);
idx = randperm(M);
nSamples = round(M);

X = X';

% Plot 64 random samples of the dataset as images
if exist('h0','var') && isvalid(h0), delete(h0);end
h0  = ml_plot_images(X(:,idx(1:64))',[sizeIm sizeIm]);

% Plot the first 8 dimensions of the image as data points
plot_options = [];
plot_options.labels = labels(idx(1:nSamples));
plot_options.title = '';
if exist('h1','var') && isvalid(h1), delete(h1);end
h1  = ml_plot_data(X([1 2 3 4 5 6 7 8], idx(1:nSamples))',plot_options);
axis equal;

% K-means Evaluation Parameters
K_range=1:10; type='L2'; repeats = 10; init = 'sample'; MaxIter = 100;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%        Task 6 Metrics computation                          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

evaluate_metric_computations(X);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%        Task 7 Kmeans evaluation                            %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Evaluate K-means to find the optimal K
[RSS_curve,AIC_curve,BIC_curve] = kmeans_eval(X, K_range, repeats, init, type, MaxIter);

% evaluation
evaluate_kmeans_eval(X);

% plot
plot_eval_curves(RSS_curve, AIC_curve, BIC_curve);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     3) Project data to 4D using PCA and Run K-means EVAL   %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Perform PCA on the digits dataset
[Mu, ~, Vsorted, ~] = solution_compute_pca(X);

% Project Digits Dataset to its first 3 principal components
p = 4;
[Yproj, Ap] = solution_project_pca(X, Mu, Vsorted, p);

% Visualize Projected Dataset
plot_options = [];
plot_options.title = 'Digits projected to 4d-subspace';
plot_options.labels = labels;
h2  = ml_plot_data(Yproj',plot_options);
axis tight
legend('1','2','3','4')

%% K-means Evaluation Parameters
K_range=1:10; type='L2'; repeats = 10; init = 'sample'; MaxIter = 100;

% Evaluate K-means to find the optimal K
[RSS_curve,AIC_curve,BIC_curve] = kmeans_eval(Yproj, K_range, repeats, init, type, MaxIter);

% Plot Metric Curves
plot_eval_curves(RSS_curve, AIC_curve, BIC_curve);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%       Task 8 F1-measure implementation on projected data        %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run k-means Once for metric evaluation
K = 5; init='sample'; type='L2'; MaxIter = 100; plot_iter = 0;
[est_labels, Mu] =  kmeans(Yproj, K, init, type, MaxIter, plot_iter);

% Compute F1-Measure for estimated labels
[F1_overall, P, R, F1] =  f1measure(est_labels(:), labels(:));
fprintf('F1 measure = %2.4f for K = %d\n',F1_overall, K)

% evaluation
evaluate_f1measure(est_labels(:), labels(:));

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5) Find the F1-measure for the different number of clusters%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K_range=1:10; type='L2'; init='sample'; MaxIter = 100; repeats = 10;

% Evaluate k-means on Original Dataset see how f-measure varies with k
f1measure_eval(X, K_range,  repeats, init, type, MaxIter, labels, 'Clustering F1-Measure -- Original Dataset--');

pause(1);
% Evaluate k-means on Original Dataset see how f-measure varies with k
f1measure_eval(Yproj, K_range,  repeats, init, type, MaxIter, labels, 'Clustering F1-Measure -- Projected Dataset--');
