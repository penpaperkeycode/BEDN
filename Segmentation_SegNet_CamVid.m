% BNNpath='C:\NeuralNetwork\Project_NeuralNet\BNN2019b';
% addpath(genpath(BNNpath))

imageSize = [360 480 3]; %[240,352,3];
classes = [
    "Sky"
    "Building"
    "Pole"
    "Road"
    "Pavement"
    "Tree"
    "SignSymbol"
    "Fence"
    "Car"
    "Pedestrian"
    "Bicyclist"
    ];

cmap = camvidColorMap;
labelIDs = camvidPixelLabelIDs();

numClasses = numel(classes);

% outputFolder = fullfile(tempdir,'CamVid');
outputFolder = fullfile('E:\Datasets\','CamVid');
imgDir = fullfile(outputFolder,'images','701_StillsRaw_full');
imds = imageDatastore(imgDir);
labelDir = fullfile(outputFolder,'labels');
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);


imageFolder = fullfile(outputFolder,'imagesResized',filesep);
imds = resizeCamVidImages(imds,imageFolder);

labelFolder = fullfile(outputFolder,'labelsResized',filesep);
pxds = resizeCamVidPixelLabels(pxds,labelFolder);

[imdsTrain,imdsTest,pxdsTrain,pxdsTest] = partitionCamVidData2(imds,pxds);


pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain);

netWidth = 16;

lgraph=segnetLayers(imageSize,numClasses,'vgg16');


learnRate=1*1e-3; %learnRate=1e-6;
miniBatchSize=5;
valFrequency = floor(421/miniBatchSize);
MaxEpochs=5000;
options = trainingOptions('sgdm', ...
    'InitialLearnRate',learnRate,...
    'MaxEpochs',MaxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress',...
    'Verbose',true, ...   % For txtal progress checking
    'VerboseFrequency',valFrequency,...
    'ExecutionEnvironment','gpu');
%     'Plots','training-progress',...

[OriginNet, info] = trainNetwork(pximds,lgraph,options);


pxdsResults = semanticseg(imdsTest,OriginNet,'WriteLocation',tempdir,'Verbose',false,'MiniBatchSize',miniBatchSize);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',true);

metrics.DataSetMetrics

metrics.ClassMetrics

% learnRate=1e-7;
% miniBatchSize=4;
% valFrequency = floor(421/miniBatchSize);
% MaxEpochs=50;
% options2 = trainingOptions('adam', ...
%     'InitialLearnRate',learnRate,...
%     'MaxEpochs',MaxEpochs, ...
%     'MiniBatchSize',miniBatchSize, ...
%     'Shuffle','every-epoch', ...
%     'Plots','training-progress',...
%     'Verbose',true, ...   % For txtal progress checking
%     'VerboseFrequency',valFrequency,...
%     'LearnRateSchedule','piecewise',...
%     'LearnRateDropFactor',0.1,...
%     'LearnRateDropPeriod',ceil(MaxEpochs/10),...
%     'ExecutionEnvironment','gpu');
% %     'Plots','training-progress',...
% lgraph2=createLgraphUsingConnections(BinSegNet.Layers,BinSegNet.Connections);
% [BinSegNet_learnrateDrop, info_learrateDrop] = trainNetwork(pximds,lgraph2,options2);
%
