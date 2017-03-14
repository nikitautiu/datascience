function plotLambdaCurve(totalX, totaly)
[X, y, Xcv, ycv, Xtest, ytest] = splitTrainingSet(totalX, totaly);
[lambda_vec, err_tr, err_cv] = validationCurve(X, y,Xcv, ycv);
plot(lambda_vec, err_tr, lambda_vec, err_cv);
legend('Train', 'Cross Validation');

xlabel('lambda');
ylabel('Error');

end