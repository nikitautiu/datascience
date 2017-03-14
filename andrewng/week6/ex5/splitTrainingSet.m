function [Xtrain, ytrain, Xcv, ycv, Xtest, ytest] = splitTrainingSet(X, y)
% randomly splits the set int 3 sets of 60/20/20% the size of the original

m = size(X, 1);
shuf_ind = randperm(m);
Xshuf = X(shuf_ind, :);
yshuf = y(shuf_ind, :);

% get the split positions
train_end = floor(m * 80 / 100);
cv_end = floor(m * 90 / 100);

% do the spilt
Xtrain = Xshuf(1:train_end, :);
ytrain = yshuf(1:train_end);
Xcv = Xshuf(train_end+1:cv_end, :);
ycv = yshuf(train_end+1:cv_end, :);
Xtest = Xshuf(cv_end+1:end, :);
ytest = yshuf(cv_end+1:end, :);

end
