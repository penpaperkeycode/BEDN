

vgg16();
outputFolder = fullfile(tempdir,'CamVid');
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
lgraph = segnetLayers(imageSize,numClasses,'vgg16');

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

BinLayers=AutoConstructBinLayer(lgraph)

Binlgraph = createLgraphUsingConnections(BinLayers,lgraph.Connections)


learnRate=1e-4;
miniBatchSize=1;
valFrequency = 2;
options = trainingOptions('adam', ...
    'InitialLearnRate',learnRate*0.1,...
    'L2Regularization',0.0005, ...
    'MaxEpochs',100, ...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',25,...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress',...
    'Verbose',true, ...   % For txtal progress checking
    'VerboseFrequency',valFrequency,...
    'ExecutionEnvironment','gpu');

% options = trainingOptions('sgdm', ...
%     'Momentum',0.9, ...
%     'InitialLearnRate',learnRate,...
%     'L2Regularization',0.0005, ...
%     'MaxEpochs',160, ...
%     'LearnRateDropFactor',0.1,...
%     'LearnRateDropPeriod',120,...
%     'MiniBatchSize',miniBatchSize, ...
%     'Shuffle','every-epoch', ...
%     'Plots','training-progress',...
%     'Verbose',true, ...   % For txtal progress checking
%     'VerboseFrequency',valFrequency,...
%     'ExecutionEnvironment','gpu');


[BinSegNet, info] = trainNetwork(pximds,Binlgraph,options);

% Binlgraph = createLgraphUsingConnections(BinSegNet.Layers,lgraph.Connections)



I = read(imdsTest);
C = semanticseg(I, BinSegNet);

B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.4);
imshow(B)
pixelLabelColorbar(cmap, classes);

expectedResult = read(pxdsTest);
actual = uint8(C);
expected = uint8(expectedResult);
imshowpair(actual, expected)

iou = jaccard(C, expectedResult);
table(classes,iou)

pxdsResults = semanticseg(imdsTest,BinSegNet,'WriteLocation',tempdir,'Verbose',false);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);

metrics.DataSetMetrics

metrics.ClassMetrics

