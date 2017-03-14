function [error_train, error_cv, error_test] = ...
    polyCurve(X, y, Xcv, ycv, Xtest, ytest, max_deg)
% plot the error with different polynomial models, the cross validation,
% and the test set
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(max_deg, 1);
error_cv    = zeros(max_deg, 1);
error_test  = zeros(max_deg, 1);


for i = 1:max_deg
   X_poly = polyFeatures(X, i);
   Xcv_poly = polyFeatures(Xcv, i);
   Xtest_poly = polyFeatures(Xtest, i);
   trained_theta = trainLinearReg(X_poly, y, 0);
   
   error_train(i) = linearRegCostFunction(X_poly, y, trained_theta, 0);
   error_cv(i) = linearRegCostFunction(Xcv_poly, ycv, trained_theta, 0);
   error_test(i) = linearRegCostFunction(Xtest_poly, ytest, trained_theta, 0);

end


end