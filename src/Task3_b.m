function Task3_b
% MP1 Task 3. (b)
% run this code by simply typing Task3_b in the workspace.

% The 'housting.data' dataset is used.
% ref to the dataset:
% https://archive.ics.uci.edu/ml/machine-learning-databases/housing/

% The objective is to find the best RBF ANN model using a base RBF 
% such as Gaussian RBF kernels
% [RBF ANN stands for Radial Basis Function Artificial Neural Networks]
% Transformation function: phi(x|x_n) = e^(-gamma * {|| x - x_n ||_2}^2)
% gamma determines the radius
% x_n represents the centre
% x is an input sample


% Author: Maryam Najafi
% Created Date: Mar 13, 2016

close all
clear all
clc

global MSE_val;

load ('housing.data');

% initialization
data_size = size(housing,1);
num_of_attrs = 13;
best_MSE_val = inf;

%trimmed_housing = trim(housing);

%% 1. declare training and validation datasets

% lengths
train_length = 306; % as the problem requires
val_length = 100 ; % 506 - 306 = 100
test_length = data_size - train_length - val_length; % 506 - 306 - 100 = 100

% 1_1. training set
Train_samples = housing(1:train_length, 1:13);
train_target = housing(1:train_length, 14);

% 1_2. validation set
Val_samples = housing (train_length + 1: data_size - test_length, 1:13);
val_target = housing (train_length + 1: data_size - test_length, 14);

% 1_3. testing set
Test_samples = housing (train_length + val_length + 1:data_size, 1:13);
test_target = housing (train_length + val_length + 1:data_size, 14);

%% 2. pre-allocation for the transformed sets
% 2_1. transformed training set
Train_samples_phi = zeros(train_length, train_length);

% 2_2. transformed validation set
Val_samples_phi = zeros(val_length, train_length);

% 2_3. transformed testing set
Test_samples_phi = zeros(100, train_length);

step = 0.01;
max = 0.5;

for gamma = 0: step:max
    disp(gamma);
%% 1. tranform datasets

% 1_1. transform the training set
for i = 1 : train_length
    
    % 1_1_1. candidate a sample from the training set
    x_n = Train_samples(i, :);
    
    % 1_1_2. transform w.r.t. all samples in the training dataset using phi(x, x_n)
    % where candidate_sample or x is an arbitrary input sample (vec)
    %       x_n is the nth training sample (vec) as the centre
    for j = 1 : train_length
        
       x = Train_samples(j, :);
       Train_samples_phi(j,i) = transform(gamma, x, x_n);
       
    end

end

% 1_2. transform the validation set
 for i = 1 : train_length
     
     % 1_2_1. candidate a sample from the validation dataset
     x_n = Train_samples(i, :);
     
     % 1_2_2. transform w.r.t. all samples in the validation dataset using 
     % the desing function phi(x, x_n)
     for j = 1 : val_length
       x = Val_samples(j, :);
       Val_samples_phi(j,i) = exp(-gamma * (norm (x - x_n, 2)^ 2));
    end
     
 end
 
 %% 2. calculate MLE for w (w_hat)
 w_hat = pinv(Train_samples_phi) * train_target;
 
 %% 3. calculate MSE_val
 % MSE_val: Mean Square Error over the hold-out data (100 samples)
 MSE_v = norm(val_target - Val_samples_phi * w_hat, 2)^2 / val_length;
 record(MSE_v);
 
 % find the best properties of the model
 % gamma, w_hat, and MSE_val
 % for the optimal value of MSE_val
 if MSE_v < best_MSE_val
     best_gamma = gamma;
     best_MSE_val = MSE_v;
     best_w_hat = w_hat;
 end
 
end
%% 4. plot
myplot (best_gamma, best_MSE_val, best_w_hat, step, max);


end

function scalar_output = transform(gamma, x, x_n)
   scalar_output = exp(-gamma * (norm (x - x_n, 2)^ 2));  
end

function record(argin)
global MSE_val;

%fprintf('MSE_val is %d \n', argin);

MSE_val = [MSE_val argin];

end

function myplot(best_gamma, best_MSE_val, best_w_hat, step, max)

global MSE_val;

x_axis = (0:step:max); % based on the gamma

plot (x_axis, MSE_val, 'Color' , 'g'); % the MSE_val curve

legend('MSE_{val}');
axis ([0 max 0 inf]);
xlabel('gamma'); ylabel('MSE_val');
title(sprintf('MSE_{val}'));
legend('show');

hold on

plot(best_gamma, best_MSE_val, 'go'); % the optimal MU

fprintf('best MSE_val: %d' , best_MSE_val);
end

function trimmed = trim (dataset)
trimmed = dataset(:, 1:13);

end