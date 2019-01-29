BNNpath='C:\Project_NeuralNet\BinarizedNeuralNetwork2018b';
addpath(genpath(BNNpath))

% load('C:\Project_NeuralNet\BinarizedNeuralNetwork2018b\TrainedNetwork\CIFAR10_16Layer\OriginNet.mat');
load('C:\Project_NeuralNet\BinarizedNeuralNetwork2018b\4.TrainedNetwork\CIFAR10_12Lyaer\MaxPool\OriginNet.mat');

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

lgraph=BinarizedSegnetArchitectureMaker_test2(CIFAR10_12layer,imageSize,classes);
% lgraph=BinarizedSegnetArchitectureMaker(ForSegNet2,imageSize);

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

[imdsTrain,imdsTest,pxdsTrain,pxdsTest] = partitionCamVidData(imds,pxds);

pixelLabelColorbar(cmap,classes);
tbl = countEachLabel(pxds)
frequency = tbl.PixelCount/sum(tbl.PixelCount);
bar(1:numel(classes),frequency)
xticks(1:numel(classes))
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')

numTrainingImages = numel(imdsTrain.Files)
numTestingImages = numel(imdsTest.Files)

numClasses = numel(classes);

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq

pxLayer = pixelClassificationLayer('Name','labels','ClassNames',tbl.Name,'ClassWeights',classWeights);
lgraph = removeLayers(lgraph,'labels');
lgraph = addLayers(lgraph, pxLayer);
lgraph = connectLayers(lgraph,'softmax','labels');

augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);

pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain,...
    'DataAugmentation',augmenter);

% pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain,...
%     'DataAugmentation',augmenter,'OutputSize',[240,352,3]);

learnRate=1e-6;
miniBatchSize=6;
valFrequency = floor(421/miniBatchSize);
MaxEpochs=180;
options = trainingOptions('adam', ...
    'InitialLearnRate',learnRate,...
    'MaxEpochs',MaxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress',...
    'Verbose',true, ...   % For txtal progress checking
    'VerboseFrequency',valFrequency,...
    'ExecutionEnvironment','multi-gpu');
%     'Plots','training-progress',...
[BinSegNet, info] = trainNetwork(pximds,lgraph,options);


I = read(imdsTest);
C = semanticseg(I, BinSegNet);

B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0);
figure(1)
imshow(B)
pixelLabelColorbar(cmap, classes);
figure(2)
imshow(I)

expectedResult = read(pxdsTest);
actual = uint8(C);
expected = uint8(expectedResult);
imshowpair(actual, expected)

iou = jaccard(C, expectedResult);
table(classes,iou)

pxdsResults = semanticseg(imdsTest,BinSegNet,'WriteLocation',tempdir,'Verbose',false,'MiniBatchSize',miniBatchSize);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);

metrics.DataSetMetrics

metrics.ClassMetrics