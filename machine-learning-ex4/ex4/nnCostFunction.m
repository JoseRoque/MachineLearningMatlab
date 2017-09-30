function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Implementation Notes: (i.e., X(i,:)' is the i-th training example x^(i),
% expressed as a n x 1 vector.)

% dimensions:
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
% X has size 5000 x 401
% y has size 5000 x 1

% pre-formatting
X = [ones(m,1) X]; % add bias unit 5000 x 401
yv = bsxfun(@eq, y, 1:num_labels); % 5000 x 10

%% Part 1

a1 = X; 
z2 = a1 * Theta1'; % 5000 x 25
a2 = sigmoid(z2); % 5000 x 25
a2 = [ones(size(a2,1),1) a2]; % 5000 x 26
z3 = a2 * Theta2'; % 5000 x 10
a3 = sigmoid(z3); 
h_x = a3; % 5000 x 10
reg_part = (lambda/(2*m)) * ( sum(sum(Theta1(:,2:end).^2,2) ) + sum( sum(Theta2(:,2:end).^2,2) )); %1 x 1 
temp =  -yv.* log(h_x)-(1 - yv).* log(1-h_x); %5000 x 10
J = (1/m) *  sum(sum ( temp,2 )) + reg_part; %1 x 1

%% Part 2


% Layer 3
b3 = a3 - yv; % 5000 x 10

% Layer 2
g_prime2 = a2(:,:).*(1-a2(:,:)); %5000 x 25

% disp(size(g_prime2))
% disp(size(b3))
% disp(size(Theta2))

%  b3 = 5000x 10, Theta2(:,2:end) = 10x 25, g_prime2 = 5000 x 25
b2 = (Theta2(:,:)'*b3')'.*g_prime2; % 5000 x 25

% a2 = 5000 x 26 , a1 = 5000 x 401
delta2 = a2(:,:)'*b3; 
delta1 = a1(:,:)'*b2(:,2:end); % exclude bias unit

% delta2 = 25 x 10 
% delta1 = 400 x 25

delta2 = delta2';
delta1 = delta1';

 D2 = (1/m) * delta2 + lambda/m * Theta2(:,:);
 D2(:,1) = (1/m) * delta2(:,1);
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
disp(size(delta2)) % 10 x 26
disp(size(delta1)) % 26 x 401
 D1 = (1/m) * delta1 + lambda/m * Theta1(:,:);
 D1(:,1) = (1/m) * delta1(:,1); %
% 
 
Theta1_grad = D1;
Theta2_grad = D2;

%% Part 3
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
