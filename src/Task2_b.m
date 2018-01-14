function Task2_b
% MP1 Task 2. (b)
% run this code by simply typing Task2_b in the workspace.
% The task1.mat dataset is used.
% The objective is to evaluate the variation of the Mean Square Error for
% the training and testing samples versus generated models using feature
% vectors that result models of the order p = 0 to p = 12.
% At the end a plot prompts two curves, shaped according to 13 models.
% Note: Models are trained using 10 samples of the dataset.
%       and the 990 remaining samples comprise the validation set.

% Author: Maryam Najafi
% Created Date: Mar 9, 2016

clc
clear all
close all

load ('task1.mat');

% record MSE_train and MSE_val to plot after the loop
% global variables
global MSE_train;
global MSE_val;

% initialization given the problem's requirements
N =  length(x); % the first N samples from the dataset (1000 in task1.mat)
x_train_length = 10; % e.g. 10
x_val_length = N - x_train_length; % e.g. 990
D = 12; % In this assignment p increases up to D = 12

        
%% 1. create datasets
% 1_1. create a training dataset
x_samples_train = x(1:x_train_length);
t_samples_train = t(1:x_train_length);

% 1_2. create a validation dataset
x_samples_val = x(x_train_length + 1:N);
t_samples_val = t(x_train_length + 1:N);

for p = 0 : D
    %% 2. create the model
    
    % 2_1. create the matrix X (input data to the model)
    % if the order of the model is p = 1 for instance, X is N x 2. (p + 1 = 2)
    % for 10 samples X is 10 by 1 if p is 0 and all its elements are 1 bcz. in
    % a polynomial x^0 is the 0th order which is 1. (y_n = w0*1 + w1*x_n + w2*x_n^2)
    X_train = ones(x_train_length, 1); % the first element of the polynomial
    for j = 1: p
        X_train = [X_train x_samples_train.^j]; % (X.^j: raise each element of X to the power of j.
    end
    
    % 2_2. create the matrix of input to model for the validation set
    X_val = ones(x_val_length, 1); % the first element of the polynomial
    for j = 1: p
        X_val = [X_val x_samples_val.^j]; % (X.^j: raise each element of X to the power of j.
    end
    
    % 2_3. calculate w_MLE
    % for the order p equals to 1 for instance, the dimension of w_MLE is 2x1.
    % (p + 1) = 2 where p = 1;
    w_MLE = pinv(X_train) * t_samples_train;
    
    
    %% 3. calculate the MSE_train and MSE_val
    
    % 3_1. MSE_train: Mean Square Error over the training data (10 samples)
    % Based on the notes (|| t - X*w||^2)/N
    MSE_t = norm(t_samples_train - X_train * w_MLE, 2)^2 / x_train_length;
    record(MSE_t, 'train');
    
    % 3_2. MSE_val: Mean Square Error over the hold-out data (990 samples)
    MSE_v = norm(t_samples_val - X_val * w_MLE, 2)^2 / x_val_length;
    record(MSE_v, 'val');
    
end

%% 4. plot
x_axis = (1:13);
plot (x_axis, MSE_train, 'Color', 'k');
axis ([1 13 0 inf]);
xlabel('D = p + 1'); ylabel('MSE');
title(sprintf('MSE_{train} and MSE_{test}'));
hold on

plot (x_axis, MSE_val, 'Color' , 'g');
legend('MSE_{train}', 'MSE_{val}');
legend('show');


end

function record(argin, argin_2)
global MSE_train;
global MSE_val;
%fprintf('MSE is %d, for the %s dataset \n', argin, argin_2);
if strcmp (argin_2, 'train')
    MSE_train = [MSE_train argin];
else
    MSE_val = [MSE_val argin];
    
end
end