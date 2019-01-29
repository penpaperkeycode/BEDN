function SimNet=HardwareSimulation(net,XValidation,YValidation,XTrain,YTrain)

%====================:Define Network Architecture:====================%

HardwareLayers=AutoConstructHWLayer(net);
% HardwareLayers=ConstructHWLayer_BatchFormBatchNorm(DeployNet64_ForSync);


SimNet=SeriesNetwork(HardwareLayers);


%====================:Test Network Architecture:======================%

YValPredSW= classify(net,XValidation);
validationErrorSW = mean(YValPredSW ~= YValidation);
YTrainPredSW = classify(net,XTrain);
trainErrorSW = mean(YTrainPredSW ~= YTrain);
disp("SW Training error: " + trainErrorSW*100 + "%")
disp("SW Validation error: " + validationErrorSW*100 + "%")

YValPredHW= classify(SimNet,XValidation);
validationErrorHW = mean(YValPredHW ~= YValidation);
YTrainPredHW = classify(SimNet,XTrain);
trainErrorHW = mean(YTrainPredHW ~= YTrain);
disp("HW Training error: " + trainErrorHW*100 + "%")
disp("HW Validation error: " + validationErrorHW*100 + "%")

% layer ='SInpU1BatchNorm';
% tic;featuresOut = activations(DeployNet64,XTrain(:,:,:,1),layer,'OutputAs','rows');toc
% featuresOut = activations(SimNet,XTrain(:,:,:,1),SimNet.Layers(2).Name,'OutputAs','rows');

% tic
% for i=1:size(XTrain,4)
%     featuresOut=[featuresOut;activations(SimNet,XTrain(:,:,:,i),SimNet.Layers(20).Name,'OutputAs','rows')];
% end
% toc
end