function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1); % examples
num_labels = size(Theta2, 1); % 10 in this case

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); % 5000 x 1

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% thetas are parameters pre trained
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
% X or a1 has size 5000 x 401 w/ bias unit added

% let X = a1
% z1 = theta1 * a1;
% a2 = g(z1);
% h(x) = a3 = g(theta2 * a2) % 5000x 10

X = [ones(m,1) X]; % add bias unit

z1 = X * Theta1'; % 5000 x 25

a2 = sigmoid(z1); % 5000 x 25

a2 = [ones(size(a2,1),1) a2]; % 5000 x 26

z2 = a2 * Theta2'; % 5000 x 10

a2 = sigmoid(z2); % 5000 x 10

[M, p] = max(a2, [], 2); % 5000 x 1
% =========================================================================


end
