function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1); % k % 1 or 3 x 1 in this ex

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1); % m x 1
m = size(X,1); % m x n or 300 x  2 in this ex
% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
% disp(K) % 3x1
% disp(size(X)) % 300 x 2
% disp(size(centroids,1)) % 3 x 2
% disp(size(idx)) % 300 x 1


 for i =1:m 
     x_i = X(i,:); % 1 x 2
     min_norm = realmax();
     for j=1:size(centroids,1)
        jth_centroid = centroids(j,:);
        current_norm = norm( x_i - jth_centroid );
        if( current_norm < min_norm )
           idx(i) = j;
           min_norm = current_norm;
        end
     end
 end


% =============================================================

end

