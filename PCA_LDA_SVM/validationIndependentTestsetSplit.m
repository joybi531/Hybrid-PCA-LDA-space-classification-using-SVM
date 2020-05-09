
load traintestPL

tic
[train test] = crossvalind('holdout',testClass,0.60);

splitDataForValidation = testDataPL(train,:);
splitDataForTest = testDataPL(test,:);
splitClassForValidation = testClassPL(train,:);
splitClassForTest = testClassPL(test,:);

toc

save traintestSplit2.mat splitClassForValidation splitClassForTest splitDataForValidation splitDataForTest
