
YValPredHW=[];
for i=1:size(testset_imds.Labels,1)
    
    tmpimg=readimage(testset_imds,i);
    YValPredHWtmp= classify(SimNet,tmpimg);
    YValPredHW=[YValPredHW;YValPredHWtmp];
    
    
end
validationErrorHW = mean(YValPredHW ~= testset_imds.Labels);
disp("HW Validation error: " + validationErrorHW*100 + "%")

% P170009495824