addpath("functions/part1")
addpath("functions/part2")
addpath("functions/part3")
addpath("evaluation_functions")
addpath("plot_functions")
addpath("utils")

clear; 
close all; 
clc;

dataset_path = 'data/';
rng(42);
seed = rng;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%           1) Load Dataset and Preprocess the Data          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ratio = 0.01;
training_data = readtable(strcat(dataset_path,'adults_data.csv'));
data_type = {true, false, true, true, false, false, false, false, false, true, true, true, false, false};
[X, y, rk] = preprocess_data(training_data,ratio, data_type);
params.data_type = data_type;
params.rk = rk;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%           Split the Dataset into Training and Testing      %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
valid_ratio = 0.7;
[X_train,y_train,X_test,y_test] = split_data(X,y,valid_ratio);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    3) Test kNN implementation (my_knn) and Visualize Results  %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Select k
params.k = 25; 
params.d_type = 'Gower';

% Compute y_estimate from k-NN
y_est =  knn(X_train, y_train, X_test, params);

% Check the accuracy
acc = accuracy(y_test, y_est);
conf = confusion_matrix(y_test,y_est);

plotConfMat(conf);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%            Choosing K by visualizing knn_eval.m            %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Select Range of K to test accuracy
M_train = length(X_train);
params.k_range = [1:10:100];
acc_curve = knn_eval(X_train, y_train, X_test, y_test, params); 

plot_accuracy(params.k_range,acc_curve)

params.k_range = [1:8:ceil(length(y)*(1-valid_ratio))];
[TP_rate, FP_rate] = knn_ROC( X_train, y_train, X_test, y_test, params );
plot_roc(FP_rate, TP_rate, params)

% Compute F-fold cross-validation
F_fold = 10;
[avgTP, avgFP, stdTP, stdFP] =  cross_validation(X, y, F_fold, valid_ratio, params);
plot_cross_validation(avgTP, avgFP, stdTP, stdFP, params)