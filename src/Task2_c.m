function Task2_c
% MP1 Task 2. (c)
% run this code by simply typing Task2_c in the workspace.
% The task1.mat dataset is used.
% The objective is to employ Ridge Regression to calculate MSE_train and 
% MSE_val for different values of regularization term of MU.
% The feature vector in this project is considered to be of order 5.
% The feature vector is denoted by phi(x).
% phi(x) = [1 x x^2 x^3 x^4 x^5]'
% For each candidate value of MU, the w_MLE should be computed.
% The resulted graph shows both MSE's with respect to MU.

% Author: Maryam Najafi
% Created Date: Mar 9, 2016
% Modified Date: Mar 12, 2016

clc
close all
clear all

load ('task1.mat');
global MSE_train;
global MSE_val;

% initialization
p = 5; % (phi(x)) feature vector's order
N =  length(x); % the first N samples from the dataset (1000 in task1.mat)
x_train_length = 10; % e.g. 10
x_val_length = N - x_train_length; % e.g. 990

best_MU = 0;
best_MSE_train = inf;
best_MSE_val = inf;
best_w_hat = [];

%% 1. create datasets
% 1_1. create a training dataset
x_samples_train = x(1:x_train_length);
t_samples_train = t(1:x_train_length);

% 1_2. create a validation dataset
x_samples_val = x(x_train_length + 1:N);
t_samples_val = t(x_train_length + 1:N);

c = 0;

figure();
for MU = 0 : 0.1: 100
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
    
    % 2_3. calculate MLE of w (w_hat)
    % the regularization parameter of MU is engaged in this part.
    w_hat = inv(X_train' * X_train + MU * eye(p + 1)) * X_train' * t_samples_train;
    
    %% 3. calcualte MSE_train and MSE_val
    
    % 3_1. MSE_train: Mean Square Error over the training data (10 samples)
    % Based on the notes (|| t - X*w||^2)/N
    MSE_t = norm(t_samples_train - X_train * w_hat, 2)^2 / x_train_length;
    record(MSE_t, 'train');
    
    % 3_2. MSE_val: Mean Square Error over the hold-out data (990 samples)
    MSE_v = norm(t_samples_val - X_val * w_hat, 2)^2 / x_val_length ;
    record(MSE_v, 'val');

    % ref: linreg_demo3.m
    % record the best Regularization Parameter (MU), w_hat, and MSE's 
    % for the optimal value of MSE_val
    if MSE_v < best_MSE_val
        best_MU = MU;
        best_MSE_val = MSE_v;
        best_MSE_train = MSE_t;
        best_w_hat = w_hat;
        
        fprintf(1,'w_hat: [%d %d]\n', best_w_hat(1), best_w_hat(2));
    end
    
end

%% 4. plot
x_axis = (0:0.1:100); % based on the Regularization Parameter
plot (x_axis, MSE_train, 'Color', 'k');
axis ([0 100 0 inf]);
xlabel('MU'); ylabel('MSE');
title(sprintf('MSE_{train} and MSE_{test}'));
hold on

plot (x_axis, MSE_val, 'Color' , 'g');
legend('MSE_{train}', 'MSE_{val}');
legend('show');
plot(best_MU, best_MSE_val, 'go')

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