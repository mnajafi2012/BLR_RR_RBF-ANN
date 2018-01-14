function Task1_f
% MP1 Task 1. (f)
% run this code by simply typing Task1_f in the workspace.
% The task1.mat dataset is used.
% The objective is to generate an animation showing each (x_n, t_n) pair as
% a new sample from the training dataset in the x_t plane along with the
% Prediction Interval of 90%.
% The alogirthm is sequential as N increases gradually from 0 to 1000.
% The current code needs some modification for a proper report!

% Author: Maryam Najafi
% Created Date: Mar 6, 2016

clc
clear all
close all

% Load the dataset
load('task1.mat');
N = length(x);  % the number of given samples

% variables initialization for the predictive distribution
alpha = 0.1; % 90% Prediction Interval is requested. Unused at the moment!
D = 2; % D x 1 is the dimension of weight vector
s = 2; % DM's certainity


% define the error vector e[n] which is normally distributed
% Generate errors based on given criteria: mu = 0; sigma = 1/4;
% e is a zero mean Gaussian random vector of length N and i.i.d
% Ref: http://www.mathworks.com/help/matlab/math/random-numbers-with-specific-mean-and-variance.html
mu = 0;
sigma = 1/2;
e = sigma^2.*randn(N,1) + mu;

%% Generate test x values to plot model outputs
% Ref: lingreg_demo2.m
x_test = (11) .* rand(20,1); % 20 uniformly distributed random numbers
%x_test = (0 : 0.1 : 11)';
length_x_test = length(x_test);
X_test = [ ones(length_x_test,1) ]; % add 1 to have elements as a pair
X_test = [ X_test x_test ]; % Append the entire x_test to the right-hand side

figure();
hold on
xlabel('x')
ylabel('t')
tmp_boundary = 0;

%% Read the samples from the dataset
% input is an N x D=2 matrix
% target is an N x 1 column vector

X_train = [1, 0]; % capital X is used since we are going to have a matrix out of it by appending the rest to the bottom.
target = 0;
last_input = [1; 0];

for i = 1 : N
    
    if i > 1
        [X_train] = [X_train; [1, x(i)]];
        [target] = [target; t(i)];
        last_input = [1; x(i)];
        last_target = t(i); % unused!
    end
    %% calculate w_MLE
    % w_MLE is a D=2 x 1 vector
    % pinv return the Pseudo inverse of X, based on the notes inv(X' * X) * X' is called c-cross
    w_MLE = pinv(X_train) * (target);
    
    % To calculate the predictive distribution both variance and mu are needed
    %% 1. calculate posterior Covariance matrix (C_{w|t})
    post_covariance = inv((1/s^2) * eye(D) + (1/sigma^2) * (X_train)' * (X_train)); % should have used equations from Task1_e though!
    
    %% 2. calculate posterior mean (E{w|t}); = w_MAP bcz. of Gaussian distr.
    post_mean = (post_covariance) * ((1/sigma^2) * (X_train)' * (target));
    
    %% 3. calculate Predictive distr. variance, given the posterior cov
    variance = sqrt ((sigma^2) + (last_input)' * (post_covariance) * (last_input));
    
    %% 4. calculate Predictive distr. mu (mean), given the posterior mean
    mu = (last_input)' * post_mean;
    
    upper_boundary = (post_mean)' * last_input + variance; % unused!
    lower_boundary = (post_mean)' * last_input - variance; % unused!
    
    %% 5. calculate Mean Square Error (MSE) of the training dataset
    % Taking an L2 norm
    length_x_train = length(X_train(:,2)); % the number of seen input samples so far (this changes)
    MSE = norm(target - X_train * w_MLE, 2)^2 / length_x_train;
    
    %% Compute output
    % Ref: linreg_demo2.m
    t_test = X_test * w_MLE;
    
    %% plot
    % Ref: linreg_demo2.m
    boundary = sqrt( MSE ) * ones( length_x_test , 1); % boundary
    [tmp_boundary] = [tmp_boundary boundary(1)]; % To observe the changes of the boundary
    tmp_input = X_train (:, 2); % unused!
    myPlot(x_test, t_test, boundary);
    
    x_n = X_train (i,2);
    t_n = w_MLE(1) + w_MLE(2) * (x_n) + e(i); % corrupted target sample given an input sample
    
    plot(x_n, t_n, 'o', 'Color', 'b');
    axis([0 10 0 11]);
    drawnow
    
    if mod(i , 100) == 0
        pause
    end
    
    
end

end

function myPlot(argin_x, argin_y, boundary)
% Ref:
% http://www.mathworks.com/matlabcentral/fileexchange/27485-boundedline-m
% Zip file is downloaded. Due to some errors out of using the boundedline.m
% the default resouces are overriden on the downloaded one.

boundedline(argin_x, argin_y, boundary);
plot(argin_x, argin_y, 'r');
axis([0 11 0 12]);
end
