% MP1 Task 1. (e)
% The task1.mat dataset is used.
% The objective is to generate an animation that shows the changes of 
% parameters in the problem. To do this an iterative algorithm is suggested,
% which is called the Sequantial Bayesian Linear Regression (BLR). It seems
% that the problem is examining the changes of the Posterior distribution
% corresponding to the number of sample increment, and is estimating w_MAP
% performace in comparison with Maximum-Likelihood Estimate of w (w_MLE).

% Author: Maryam Najafi
% Created Date: Feb 29, 2016
% Modified Date: Mar 6, 2016

% The system is assumed as t[n] = w_1 + w_2 * x[n] + e[n]               (1)
% % Where e[n] (error)'s distribution is Gaussian with the mean and variance
% % are 0 and 1/4 respectively. e[n] ~ N (0, 1/4)
% % The Decision Maker's (DM) a-priori is w ~ N_d (0, s^2 I); d = 2;

function run()
clc
clear all
close all

D = 2; % Bcz. the vector w has two coefficients (w_1 and w_2 based on Equation (1))

% 
% Load the dataset
load('task1.mat');
N = length(x);
% As presumed and used for creating the dataset the true value of w is:
% The w_TRUE is [1 1]' and is shown like o
% The Maximum Likelihood Estimate of w (w_MLE) is shown like +
% The Maximum A Posterior of w (w_MAP) is shown like x

%%
% Bayesian Inference
% 1. Making some prior assumptions about the distribution (a-priori)
% 2. Parameters (w's) are considered as random variables
%    Hyperparameters are included in these assumptions
% 3. Declare and initialize error to create the system equation
% 4. Compute the Posterior distribution (a-posteriori)
% 5. Revise the parameters (w's) if needed

%%
% 1. Make assumptions of the distribution of w's. Define mu & covariance.
s = 2; % A user-defined numerical value for the hyperparameter s in (s^2 I); s^2>0
mu = [0, 0]; % mean of w
variance = (s ^ 2) * eye(D); % Covariance Matrix of w
rng default  % For reproducibility (Ref: http://www.mathworks.com/help/stats/mvnrnd.html)
w = mvnrnd(mu, variance, D);


% 2. assign random values to the parameters (weight vector)
% These values are refined during each iteration based on the a-posteriori.
w_1 = w(1); % a Gaussianly distributed random number w.r.t the problem's criteria
w_2 = w(2);


% 3. define the error vector e[n] which is normally distributed
% Generate errors based on given criteria: mu = 0; sigma = 1/4;
% e is a zero mean Gaussian random vector of length N and i.i.d
% Ref: http://www.mathworks.com/help/matlab/math/random-numbers-with-specific-mean-and-variance.html
mu = 0;
sigma = 1/2;
e = sigma^2.*randn(N,1) + mu;

% To avoid cumbersome calculation of inversion in every iteration we have a
% history of our prior parameters and then given the formulas, previously 
% discussed in the problem b, c, and d, we calculate the posterior parameters

% Find the w_MLE
% At first w_MLE can be 0
w_MLE = zeros(1,D)'; % then we update it in each loop
% If w_MLE tends to 0 means the Sum of Squared Error (SSE) is minimized

% w_TRUE is [1 1] based on the problem
w_TRUE = [1 1]'; % the ultimate value does NOT change!

% Find the covariance for N = 0;
% It turned out that the covariance when there is no data, is the same Prior
% so covariance(1) is the prior for N = 0;
% Store covariance in history: e.g. index 1 and 2 for N = i and N = i + 1; 
% where N is the number of samples and i + 1 is a sample coming after i_th sample
covariance(:,:,D) = zeros(D,D);
covariance(:,:,1) = (s^2) * eye(D);

% Find the w_MAP for N = 0;
% It turned out that the Maximum A Posteriori estimate of w, when there
% is no data, is the same Prior
% Since the distr. is Gaussian (mode == mean) w_MAP is equivalent to the mean 
% Since the Prior mean is 0:
w_MAP(:,1) = zeros(1,D)';

%% Offline Batch processing
% to demonstrate the hypothesis that the MAP estimate of w should be
% initialized to the value of mean which is 0 in our case.
sample_i = [1; x(1)];
sample_t = [t(1)];

%calculate w_hat

% 1. Assuming that prior w_MAP is equal to 0 so that G(N+1)*w_MAP(N) is
% eliminated from the equation (4) of Task1. c.
w_hat_1 = ( (sample_t) * covariance(:,:,1) * sample_i) / (( sigma^2) + (sample_i)' * covariance(:,:,1) * sample_i);

% 2. Using the Bayesian Linear Regression formula of the w_MAP calculation
sample_i = [1 x(1)];
w_hat_2 = inv((sample_i' * sample_i) + ((sigma^2)/s^2) * eye(D)) * sample_i' * sample_t;

% 3. The numerical value of w_hat_1 and w_hat_2 is the same.
%%
% variables initialization
r = 1;
l = 2;
input = [1, 0];
target = [0];

% axis range
leftboundary = 0.5;
rightboundary = 1.5;

fig = figure();
G_tmp = [0 0; 0 0];

% Iterate over X (samples)
for i = 1 : N
scatter(1,1, 'x');
hold on
% more moving forward through the loop, more meeting new samples
% expanding the dataset with augmenting a new sample for each iteration

%% Add a new sample pair
% input expands in every iteration gradually.
[input] = [[input]; [1, x(i)]];
[target] = [ [target]; t(i)];
last_input = [1; x(i)];
last_target = t(i);

%% 1. Calculate the Posterior distribution p (w|t)
% by adding dataset a sample at a time
% Doing so, I compute the mean and the covariance matrix then plug them in
% the Gaussian distr. equation.
% Based on the given formula in the Gaussian Linear Regression notes

% 1_1. In order to find the posterior covariance first calculate G(N+1)
res = (covariance(:,:,r) * (last_input) * (last_input)') / ((sigma^2) + (last_input)' * covariance(:,:,r) * (last_input));
G = eye(D) - (res);
G - G_tmp
G_tmp = G;

%1_2. Now plug in the G into the equation 2: cov(N+1) = G(N+1) + cov(N)
covariance(:,:,l) = G * covariance(:,:,r); % verify deep-copy?
% Note: covariance(l) is the most updated one for N + 1 samples (current i)
%       covariance(r) is for N samples

%1_3. calculate w_MAP
w_MAP = (G * w_MAP) + ((last_target) * covariance(:,:,r) * last_input /( (sigma^2) + ( last_input' * covariance(:,:,r) * last_input)));

%% 2. Calculate w_MLE
% Based on the given formula in the Gaussian Linear Regression notes
w_MLE = pinv(input) * (target); % pinv return the Pseudo inverse of X, based on the notes inv(X' * X) * X' is called c-cross

scatter(w_MLE(1),w_MLE(2), 'o');
legend('w_{TRUE}','w_{MLE}', 'location', 'northwest');

%% 3. Plot the D-variate Gaussian distribution
% To avoid Positive-Definite error I added a very small value (at this time, not neccessary though!)
SIGMA = covariance(:,:,r) + .0001 * ones(2);

myPlot(leftboundary, rightboundary, w_MAP', SIGMA);

%% swap
if r == 1;
    r = 2;
    l = 1;
else
    r = 1;
    l = 2;
end

axis([leftboundary rightboundary leftboundary rightboundary]); % e.g. [0.5 1.5 0.5 1.5]
%% record Plot
%M(i) = getframe;
hold off

end
%% record the movie then play it once more
%movie2avi(M , 't2.avi');
%movie(M,1);
end

function myPlot(leftboundary, rightboundary, mu, sigma)
% Ref: http://www.mathworks.com/help/stats/multivariate-normal-distribution.html

step = 0.001; % Gaussian plot resolution

x1 = leftboundary:step:rightboundary; x2 = leftboundary:step:rightboundary;
[X1,X2] = meshgrid(x1,x2);

F = mvnpdf([X1(:) X2(:)],mu,sigma);
F = reshape(F,length(x2),length(x1));

contour(x1,x2,F,[.0001 .001 .01 .05:.1:.95 .99 .999 .9999]);
xlabel('x'); ylabel('y');

end
