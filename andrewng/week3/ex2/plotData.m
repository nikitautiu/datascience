function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
posMask = (y == 1);     % use the masks to select which sets to plot to what
negMask = not(posMask);

plot(X(posMask, 1), X(posMask, 2), 'k+','LineWidth', 1, 'MarkerSize', 4);
plot(X(negMask, 1), X(negMask, 2), 'ko','LineWidth', 1, 'MarkerFaceColor', 'y', 'MarkerSize', 4);

legend('Positive', 'Negative');



% =========================================================================



hold off;

end
