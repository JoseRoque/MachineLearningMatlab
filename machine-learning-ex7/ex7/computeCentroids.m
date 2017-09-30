function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% disp(size(X))
% disp(size(idx))
% disp(size(K))
% disp(size(centroids))
%    300     2 300 examples w/ 2 features in 2d space
%    300     1 
%      1     1 K=3
%      3     2

 for k=1:K
%     
%     find all x_1 and x_2 features with assigned centroid
%     find average of points assigned to each k if there are none avg
%     remains 0
    
    [ row, ~, ~ ] = find(idx == k); % find all samples assigned to specific K
    val = X(row,:);
    centroids(k,:) = (1/size(val,1)) * sum(val);
    
 end

% =============================================================


end

