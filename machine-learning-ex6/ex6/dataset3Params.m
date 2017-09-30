function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
    
    C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
    a = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

    min_C = C(1);
    min_a = a(1);
    min_prediction_error = 1.0;
    for i= 1:size(C,2) 
        for j=1:size(a,2) 
    
            % train model
            model = svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, a(j)));
            
            % predict labels on cross validation set
            predictions = svmPredict(model, Xval);

            % compute prediction error on cross validation set
            prediction_error = mean(double(predictions ~= yval));
            
            if( prediction_error < min_prediction_error )
                min_C = C(i);
                min_a = a(j);
                min_prediction_error = prediction_error;
            end
            
        end
    end
    C = min_C;
    sigma = min_a;

% =========================================================================

end
