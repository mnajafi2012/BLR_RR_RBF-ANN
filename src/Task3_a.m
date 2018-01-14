function Task3_a
% MP1 Task 3. (a)
% run this code by simply typing Task3_a in the workspace.

% The 'housting.data' dataset is used.
% ref to the dataset:
% https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
% The objective is to the predict median values of owner-occupied homes in
% suburbs of the Boston area
% The dataset has 506 samples each of which has 13 attributes plus 1 class

% The algorithm selects 306 samples as the training set
%                       100 samples as the validation set
% The result of the following code is a graph showing the MSE_val vs. MU
% where MSE_val stands for Mean Squared Error for the Validation dataset
% and MU is the Regularization Parameter of the Ridge Regression


% Author: Maryam Najafi
% Created Date: Mar 13, 2016

clc
close all
clear all

global MSE_val;

load ('housing.data');

% initialization
data_size = size(housing,1);
num_of_attrs = 13; % number of attributes
best_MSE_val = inf;

%% 1. declare training and validation datasets
% 1_1. training dataset
train_length = 306; % as the problem requires
Train_samples = housing(1:train_length, 1:13);
train_target = housing(1:train_length, 14);

% 1_2. validation dataset
val_length = data_size - train_length; % 506 - 306 = 100
Val_samples = housing (train_length + 1: data_size, 1:13);
val_target = housing (train_length + 1: data_size, 14);

figure();
for MU= 0:0.1:100
    %% 2. create the model
    
    % 2_1. calculate MLE of w (w_hat)
    % the regularization parameter of MU is engaged in this part.
    w_hat = inv(Train_samples' * Train_samples + MU * eye(num_of_attrs)) * Train_samples' * train_target;
    
    %% 3. calculate MSE_val
    % MSE_val: Mean Square Error over the hold-out data (100 samples)
    MSE_v = norm(val_target - Val_samples * w_hat, 2)^2 / val_length;
    record(MSE_v);
    
    % record the best Regularization Parameter (MU), w_hat, and MSE_val 
    % for the optimal value of MSE_val
    if MSE_v < best_MSE_val
        best_MU = MU;
        best_MSE_val = MSE_v;
        best_w_hat = w_hat;
    end
 
end

%% 4. plot

myplot(best_MU, best_MSE_val, best_w_hat);

end

function record(argin)
global MSE_val;

fprintf('MSE_val is %d \n', argin);

MSE_val = [MSE_val argin];

end

function myplot(best_MU, best_MSE_val, best_w_hat)

global MSE_val;

x_axis = (0:0.1:100); % based on the Regularization Parameter

plot (x_axis, MSE_val, 'Color' , 'g'); % the MSE_val curve

legend('MSE_{val}');
axis ([0 100 0 inf]);
xlabel('MU'); ylabel('MSE_val');
title(sprintf('MSE_{val}'));
legend('show');

hold on

plot(best_MU, best_MSE_val, 'go'); % the optimal MU

fprintf('best MSE_val: %d \n' , best_MSE_val);
fprintf('best w_hat: %d' , best_w_hat);


end