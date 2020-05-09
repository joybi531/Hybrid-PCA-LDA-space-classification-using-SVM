
tic
load traintestPL
load traintestSplit2

cMinValue = 2;
cMaxValue = 3;
sigmaMinValue = 1;
sigmaMaxValue = 2;

predictionList = zeros(size(splitClassForValidation, 1), 1);
uniqueClass = unique(trainClassPL);

numOfMetric = 4;
dimC = cMaxValue - cMinValue + 1;
dimSigma = sigmaMaxValue - sigmaMinValue + 1;

performanceInfo = zeros(dimC, dimSigma, numOfMetric);

a = 1;

for cVal = cMinValue : 0.10 : cMaxValue
   actualC=2^cVal;
   b = 1;

   for sigmaVal = sigmaMinValue : 0.10 : sigmaMaxValue
      actualSigma=2^sigmaVal;
      
      
      for i=1:size(uniqueClass,1)
          groups = ismember(trainClassPL, uniqueClass(i));
          opts = statset('Display','final','MaxIter', 500000);
          svmStruct(i) = svmtrain(trainDataPL,groups,'Method','SMO','SMO_Opts',opts,'BoxConstraint', actualC, 'Kernel_Function','rbf', 'RBF_Sigma', actualSigma,'KernelCacheLimit',10000);
      end
      
      prediction = predictTestClass(svmStruct, splitDataForValidation, uniqueClass);
    
      [overallAccuracy, accuracyPerClass] = performanceMeasure(prediction, splitClassForValidation);
      
      performanceInfo(a, b, 1) = cVal;
      performanceInfo(a, b, 2) = sigmaVal;
      performanceInfo(a, b, 3) = overallAccuracy;
      accuracyPerClass;
      
      b = b + 1;
   end
   a = a + 1;
end

bestC = 1;
bestSigma = 1;
accuracy = 0.0;
for i = 1:dimC
   for j = 1:dimSigma
      if accuracy < performanceInfo(i,j,3)
         accuracy = performanceInfo(i,j,3);
         bestC = performanceInfo(i, j, 1);
         bestSigma = performanceInfo(i, j, 2);
      end
   end
end

fprintf('Best c = %d and Best sigma = %d\n', bestC, bestSigma);

toc

save c_and_sigma.mat bestC bestSigma
