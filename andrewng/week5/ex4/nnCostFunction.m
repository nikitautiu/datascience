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
                 hidden_layer_size, (input_layer_size + 1)); % hidden layer size X how many inputs + 1(bias)

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));  % same

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
% -------------------------------------------------------------

% compute the expected output matrix from y
% initialize the expected values with 0
expectedOutput = zeros(m, num_labels);
% compute the linear indices from the 2D ones
inds = sub2ind(size(expectedOutput), 1:m, y');
expectedOutput(inds) = 1; % set to ones based on y values

% pad the inputs with 1 for the bias 
% and compute the predictions
X = [ones(size(X, 1), 1) X];
z2 = X * Theta1';
a2 = sigmoid(z2); % hiden lyer predictions

% pad the predictions with 1 as the bias
a2 = [ones(size(a2, 1), 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% each row of Theta should countain the coefs for the coresponding unit on
% the next layer. therefore, multiplying with its transpose will yield
% multiple ROWS, each representig the values for a particular training
% example
unregCost = expectedOutput .* log(a3) + (1 - expectedOutput) .* log(1 - a3);
unregCost = -(1/m) .* unregCost;
unregCost = sum(unregCost(:));

% caclulate the regularization. don't forget to ignore the first column
% which is the weight of the bias
theta1Sqrs = Theta1(:, 2:end) .^ 2;
theta2Sqrs = Theta2(:, 2:end) .^ 2;
regDiff = lambda/(2*m) .* (sum(theta1Sqrs(:)) + sum(theta2Sqrs(:)));

J = unregCost + regDiff; % unroll it to calculate the total sum

% ----------------------------- DO BACKPROP -------------------------------
delta3 = a3 - expectedOutput; % the errors for the outputs
% each line represents a training set

% vectorized version of the algorithm
delta2 = (delta3 * Theta2);
delta2 = delta2(:, 2:end)  .* sigmoidGradient(z2); % ignore the bias

% batch vectorisation for getting the deltas
% Motivation:
% Delta(l) = delta(l+1) * a(l)'  -- in the algebraic formula
% however, due to batch processing deltal = delta(l)'
% and al = a(l)' transposing the matrix product, we get the formula used
% below
Delta2 = delta3' * a2;
Delta1 = delta2' * X;

% compute the gradients and regularise
Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2; % the 1/m is kinda like what we do in stochastic

% regularise all except first colum which is the bias column
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m) * Theta2(:, 2:end);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
