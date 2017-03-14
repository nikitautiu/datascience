function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% we're using the same function as in the previous week
% we already did it vectorized

m = length(y); % number of training examples
pred = sigmoid(X * theta); % the predictions given the theta
diff = pred - y;  % the deviation from the exprected results

lambdaCostDiff = lambda/(2*m) * sum(theta(2:end).^2); % lambda regularization to add to the cost func
J = -(1/m) * sum(y.*log(pred) + (1-y).*log(1-pred)) + lambdaCostDiff; % logistic error + lambda diff

lambdaGradDiff = (lambda/m) .* [0; theta(2:end)]; % regularization diff to add to the grad(0 for the bias)
grad = (1/m) * (X' * diff) + lambdaGradDiff;  % the same as the linear regression gradient but diffrente pred







% =============================================================

grad = grad(:);

end
