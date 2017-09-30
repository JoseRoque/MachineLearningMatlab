function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

Theta_sq = Theta.^2;
X_sq = X.^2;

reg1 = sum(sum(Theta_sq,2));
reg2 = sum(sum(X_sq,2));

J = 1/2 * sum( sum( ( (X * Theta') - Y).^2.*R) ) + lambda/2 * (reg1 + reg2);

for i=1:num_movies
  idx = find(R(i,:)==1); % which users voted on movie i
  Theta_temp = Theta(idx,:); % get all users that voted for movie i
  Y_temp = Y(i,idx); % for a certain movie what users voted on it
  
  reg_x = lambda * X(i,:);
  X_grad(i,:) =( ( (X(i,:) * Theta_temp') - Y_temp) * Theta_temp ) + reg_x; 
end % 5 x 3

for i=1:num_users
  idx = find(R(:,i)==1); % which movies did user j vote
  X_temp = X(idx,:); % get all movies user j voted for
  Y_temp = Y(idx,i); % for a certain user what movies they vote on
 
%   disp('sizes');
%     disp(size(idx));
%     disp(size(X_temp));
%     disp(size(Y_temp));
%     disp(size(Theta(i,:)'))
%     disp(size(( (X_temp * Theta(i,:)') - Y_temp)))
   reg_theta = lambda * Theta(i,:);
   Theta_grad(i,:) =( ( (X_temp * Theta(i,:)') - Y_temp)' * X_temp ) + reg_theta; 
end % 4 x 3


% Theta_temp = Theta(j,:); % 
% X_temp = X(i,:); % 
% Y_temp = Y(i,j); % 
% 
% disp(size(Theta_temp))
% disp(size(X_temp))
% disp(size(Y_temp))
% disp(size(X_temp * Theta_temp'))
% 
% X_grad = ((X_temp * Theta_temp') - Y_temp ) * X_temp;
% Theta_grad = ((X_temp * Theta_temp') - Y_temp ) * Theta_temp;
% 
% disp(size(X_grad))
% disp(size(Theta_grad))

% =============================================================
% 
% main_calc = ((X * Theta') - Y).*R;
% 
% disp(size(main_calc)) % 5x4
% % X - 5 x 3
% % Theta - 4 x 3
% % Y - 5 x 4
% disp(size(X_grad)) % 5x3
% disp(size(Theta_grad)) %4 x3
% 
% %  Y - num_movies x num_users matrix of user ratings of movies
% %  X - num_movies  x num_features matrix of movie features
% X_grad = main_calc'*X; % 5 x 3
% Theta_grad = main_calc'*Theta'; % 4 x 3

    grad = [X_grad(:); Theta_grad(:)]; 

end
