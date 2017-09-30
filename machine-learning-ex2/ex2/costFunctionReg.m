function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%                y =    m x 1
%                log(X * theta) = m x 1
%               sigmoid_res = m x 1
%               X = m x (n+1)
%               theta = (n+1) x 1
%               grad = 1 x n+1
% x subj supi = 

%% cost
y_neg = y * -1;
sigmoid_res = sigmoid( X * theta );
regularization_part = (lambda/ (2 * m )) * sum(theta(2:end).^2); %working
J = (1/m) * sum (y_neg.* log(sigmoid_res)-(1 - y).* log(1-sigmoid_res)) + regularization_part; 

%% gradient
regularized_theta = theta * (lambda/m);
grad = (1/m) * sum((sigmoid_res - y).*X)' + regularized_theta;  % try col and row wise 
% grad = (1 / m) * sum( (sigmoid_res - y).*X ); % try col and row wise 

grad(1) = grad(1) - (theta(1) * (lambda/m));

% =============================================================

end
