function Task2_a
% MP1 Task 2. (a)
% run this code by simply typing Task2_a in the workspace.
% The task1.mat dataset is used.
% The objective is to find the best fit model to our dataset.
% The algorithm divides the dataset in two: training and validation sets
% First 10 samples from the dataset are picked for this assignment.
% [The rest which are 990 will be candidates for the next code. (Task2_b)]
% Using the Linear Regression on those 10 sample, w_MLE is calculated for
% each selection of p. (p is the order of the feature vector).
% The goal is to show how different feature vectors with different orders
% to design a model varies for the samples. Subsequently, the model's
% outputs versus sample input are depicted in plots.

% Author: Maryam Najafi
% Created Date: Mar 8, 2016

clc
clear all
close all

% Load the dataset
load('task1.mat');

% initialization given the problem's requirements
N =  10; % the first N samples from the dataset
p_max = 12; % In this assignment p increases up to 12

% The entire operations below are repeated for 13 times
% For each iteration the result is a plot in the y_x plane
% Some of these plots are chosen for the report

%% 1. create a new dataset of 10 samples
x_samples = x(1:N);
t_samples = t(1:N);

% 1_1. plot the actual x_t from the dataset
% figure();
% scatter (x_samples, t_samples, 'filled');
% axis ([0 12 0 12]);
% xlabel('x'); ylabel('t');
% title('The first 10 actual samples' );

%figure();
for p = 0 : p_max
    %% 2. create the model
    
    % 2_1. create the matrix X
    % if the order of the model is p = 1 for instance, X is N x 2. (p + 1 = 2)
    % for 10 samples X is 10 by 1 if p is 0 and all its elements are 1 bcz. in
    % a polynomial x^0 is the 0th order which is 1. (y_n = w0*1 + w1*x_n + w2*x_n^2)
    X = ones(N, 1); % the first element of the polynomial
    for j = 1: p
        X = [X x_samples.^j]; % (X.^j: raise each element of X to the power of j.
    end
    
    % 2_2. calculate w_MLE
    % for the order p equals to 1 for instance, the dimension of w_MLE is 2x1.
    % (p + 1) = 2 where p = 1;
    w_MLE = pinv(X) * t_samples;
    %w_MLE = [1; 1];
    
    % 2_3. generate iid random noise
    mu = 0;
    sigma =  0.7;
    e = sigma^2.*randn(N,1) + mu;
    
    % 2_3. calculate y vector
    % The result for y is a column vector of N x 1.
    y = X * w_MLE + e ;
    
    %% 3. plot
    
    % original
    % scatter (x_samples, t_samples, 'filled');
    % axis ([0 12 0 12]);
    % hold on
    
    % model's output
    scatter (x_samples, y);
    axis ([0 12 0 12]);
    xlabel('x'); ylabel('y');
    if p ~= 1
        title (sprintf('p = %d', p));
    else
        title (sprintf('p = %d, w_{MLE}=[%d %d]', p, w_MLE(1), w_MLE(2)));
    end
    hold off
    %pause(3);
    drawnow
end
end