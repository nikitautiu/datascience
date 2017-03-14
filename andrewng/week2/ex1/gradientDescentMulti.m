function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of features
J_history = zeros(num_iters, 1);

for iter = 1:num_iters  
    J_history(iter) = computeCost(X, y, theta);
    
    pred = X * theta;
    diff = pred - y;
%     delta_theta = zeros(n, 1);
%     for i = 1:n
%         % elemnt-wise product of errors and the
%         % corresponding feature for every theta
%         delta_theta(i) = sum(diff .* X(:, i)); 
%     end 
    delta_theta = (1/m) * X' * diff;  % exactly the same but vectorised
    
    theta = theta - alpha*delta_theta;
end

end 