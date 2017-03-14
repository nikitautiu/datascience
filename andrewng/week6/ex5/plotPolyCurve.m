function plotPolyCurve(totalX, totaly, max_deg)
[X y Xcv ycv Xtest ytest] = splitTrainingSet(totalX, totaly);
[err_tr, err_cv, err_test] = polyCurve(X, y, Xcv, ycv, Xtest, ytest, max_deg);
plot(1:max_deg, err_tr(1:max_deg), 1:max_deg, err_cv(1:max_deg), 1:max_deg, err_test(1:max_deg));
legend('Train', 'Cross Validation', 'Test');
end