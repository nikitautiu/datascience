function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



% Initialize some useful values
m = length(y); % number of training examples
m = length(y);
pred = sigmoid(X * theta);
diff = pred - y;  % the deviation from the exprected results

costDiff = lambda/(2*m) * sum(theta(2:end).^2); % lambda regularization
J = -(1/m) * sum(y.*log(pred) + (1-y).*log(1-pred)) + costDiff; 

gradDiff = (lambda/m) .* [0; theta(2:end)]; % regularization diff to add to the grad(first is 0)
grad = (1/m) * (X' * diff) + gradDiff;  % the same as the linear regression gradient but diffrente pred






% =============================================================

end
