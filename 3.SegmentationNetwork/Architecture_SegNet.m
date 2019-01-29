lgraph=BinarizedSegnetLayers()

% outputFolder = fullfile(tempdir,'CamVid');
outputFolder = fullfile('D:\','CamVid');
imgDir = fullfile(outputFolder,'images','701_StillsRaw_full');
imds = imageDatastore(imgDir);
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
labelIDs = camvidPixelLabelIDs();
labelDir = fullfile(outputFolder,'labels');
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);
cmap = camvidColorMap;
pixelLabelColorbar(cmap,classes);
tbl = countEachLabel(pxds)
frequency = tbl.PixelCount/sum(tbl.PixelCount);
bar(1:numel(classes),frequency)
xticks(1:numel(classes))
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')

imageFolder = fullfile(outputFolder,'imagesResized',filesep);
imds = resizeCamVidImages(imds,imageFolder);

labelFolder = fullfile(outputFolder,'labelsResized',filesep);
pxds = resizeCamVidPixelLabels(pxds,labelFolder);

[imdsTrain,imdsTest,pxdsTrain,pxdsTest] = partitionCamVidData(imds,pxds);

numTrainingImages = numel(imdsTrain.Files)
numTestingImages = numel(imdsTest.Files)

imageSize = [360 480 3];
numClasses = numel(classes);

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq

pxLayer = pixelClassificationLayer('Name','labels','ClassNames',tbl.Name,'ClassWeights',classWeights)
lgraph = removeLayers(lgraph,'pixelLabels');
lgraph = addLayers(lgraph, pxLayer);
lgraph = connectLayers(lgraph,'softmax','labels');

augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);

pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain,...
    'DataAugmentation',augmenter);


learnRate=1e-6;
miniBatchSize=1;
valFrequency = 421;
options = trainingOptions('sgdm', ...
    'InitialLearnRate',learnRate,...
    'L2Regularization',0.0005, ...
    'MaxEpochs',20, ...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',45,...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...   % For txtal progress checking
    'VerboseFrequency',valFrequency,...
    'ExecutionEnvironment','gpu');
%     'Plots','training-progress',...
[BinSegNet, info] = trainNetwork(pximds,lgraph,options);