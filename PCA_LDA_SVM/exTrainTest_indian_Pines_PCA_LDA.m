
load traintestPCA
load traintestLDA


trainDataPL = [trainData(1:4615,:) trainDataLDA(1:4615,:)];
trainClassPL = trainClass(1:4615) ;
testDataPL = [testData testDataLDA];
testClassPL = testClass;



save traintestPL trainDataPL testDataPL trainClassPL testClassPL;

