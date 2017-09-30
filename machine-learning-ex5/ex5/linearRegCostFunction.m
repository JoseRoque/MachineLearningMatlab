function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
num_feat = size(theta); % number of features
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

regularization_part = (lambda/ (2 * m )) * sum(theta(2:end).^2); %working

J = 1/(2 * m) * sum( ((X * theta) - y).^2 ) + regularization_part;

% theta = 2 x 1
% X = 12 x 2
% y = 12 x 1 
% X * theta = 12 x 1 
h_x = X * theta;

regularized_theta = theta * (lambda/m);
grad = (1/m) * sum(( h_x - y).*X)' + regularized_theta;  % try col and row wise 
grad(1) = grad(1) - (theta(1) * (lambda/m));

%grad = theta(:); % 2 x 1

end
