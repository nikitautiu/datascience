function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
% NOTE: generalized the size so it works with matrices of examples as well
m = size(X, 1);
n = size(X, 2);
X_poly = zeros(m, p * n);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 

% horizontally concat the matrices
for i=1:p
    X_poly(:, ((i-1)*n+1):(i*n)) = X .^ i;
end

% =========================================================================

end
