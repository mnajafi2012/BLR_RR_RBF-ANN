function Task3_c
% MP1 Task 3. (c)
% run this code by simply typing Task3_c in the workspace.

% The 'housting.data' dataset is used.
% ref to the dataset:
% https://archive.ics.uci.edu/ml/machine-learning-databases/housing/

% The objective is to comapre the RR model to RBF-ANN model by evaluating
% their performance over the validation set

% Then evaluating the winning model (RR in my case)'s performance
% to do this Normalized MSE_test is employed and the result was promissing
% As the conclusion the RR model with the weight vector of 13 could win
% other models with NMSE = 0.0018 or 92% accuracy.


% Author: Maryam Najafi
% Created Date: Mar 13, 2016

close all
clear all
clc

load ('housing.data');

% initialization
data_size = size(housing,1);

%% 1. declare training and validation datasets

% lengths
train_length = 306; % as the problem requires
val_length = 100 ; % 506 - 306 = 100
test_length = data_size - train_length - val_length; % 506 - 306 - 100 = 100

% 1_3. testing set
Test_samples = housing (train_length + val_length + 1:data_size, 1:13);
test_target = housing (train_length + val_length + 1:data_size, 14);

%% 2. Evaluate on the test set to compare RR and LR models
% RR is chosen as a winning model between RBF-ANN and RR model since the
% MSE_val using RR model is ~94.6 whereas RBF-ANN returns ~122.2 of error

% 2_1. Using RR model
% 2_1_1. borrow Ridge Regression w_hat from (Task 3(a))
RR_w_hat = [0.453388229668412;0.0168444547529867;-0.0159863052295315;0.541760641357113;-1.27089920491848;7.53385599378387;-0.0420573385509527;-1.01250669231332;0.142305576841677;-0.0180275689312016;-0.707826494494815;0.0118620543860474;-0.252695299545605;]

% 2_1_2. calculate MSE_test
MSE_val = norm(test_target - Test_samples * RR_w_hat, 2)^2 / test_length

% 2_1_3. Normalize the MSE_test to see the performance
NMSE = MSE_val / norm (test_target,2)^2 % 0.0018


% 2_2. Using LR model
% 2_2_1. as the question asks assume w = (1/D)1_D for D = 13
D = 13; LR_w_hat = ones(D,1) * 1/D

% 2_2_2. calcualte MSE_test
MSE_val = norm(test_target - Test_samples * LR_w_hat, 2)^2 / test_length

% 2_2_3. Normalize the MSE_test
%NMSE = MSE_val / norm (test_target,2)^2 % 0.1562

end