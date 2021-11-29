
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

[imdsTrain,imdsTest,pxdsTrain,pxdsTest] = partitionCamVidData(imds,pxds);

% augmenter = imageDataAugmenter('RandXReflection',true,...
%     'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);

pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain);

netWidth = 16;

FrontLayer = [
    imageInputLayer(imageSize,'Name','input','Normalization','none') %32*32*3
    InputScaleLayer('Scale1')
    
    Binarizedconvolution2dLayer(3,4*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv1')
    batchNormalizationLayer('Name','BatchNorm1')
    SignumActivation('Sign1')
    Binarizedconvolution2dLayer(3,4*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv2')
    batchNormalizationLayer('Name','BatchNorm2')
    SignumActivation('Sign2')
    
    Binarizedconvolution2dLayer(3,8*netWidth,'Padding','same','Stride',2,'BiasLearnRateFactor',0,'Name','binConv3')
    batchNormalizationLayer('Name','BatchNorm3')
    SignumActivation('Sign3')
    Binarizedconvolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv4')
    batchNormalizationLayer('Name','BatchNorm4')
    SignumActivation('Sign4')
    
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',2,'BiasLearnRateFactor',0,'Name','binConv5')
    batchNormalizationLayer('Name','BatchNorm5')
    SignumActivation('Sign5')
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv6')
    batchNormalizationLayer('Name','BatchNorm6')
    SignumActivation('Sign6')
    
    BinarizedtransposedConv2dLayer(3,8*netWidth,'Cropping','same','Stride',2,'BiasLearnRateFactor',0,'Name','binConv7')
    batchNormalizationLayer('Name','BatchNorm7')
    SignumActivation('Sign7')
    Binarizedconvolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv8')
    batchNormalizationLayer('Name','BatchNorm8')
    SignumActivation('Sign8')
    
    BinarizedtransposedConv2dLayer(3,4*netWidth,'Cropping','same','Stride',2,'BiasLearnRateFactor',0,'Name','binConv9')
    batchNormalizationLayer('Name','BatchNorm9')
    SignumActivation('Sign9')
    Binarizedconvolution2dLayer(3,4*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv10')
    batchNormalizationLayer('Name','BatchNorm10')
    SignumActivation('Sign10')
    ];


TailLayer=[
    Binarizedconvolution2dLayer(3,numClasses,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv11')
    batchNormalizationLayer('Name','BatchNorm11')
    softmaxLayer('Name', 'softmax')
    pixelClassificationLayer('Name', 'labels')
    ];

layers=[
    FrontLayer;
    TailLayer
    ];

lgraph=layerGraph(layers);

tbl = countEachLabel(pxds)
numTrainingImages = numel(imdsTrain.Files)
numTestingImages = numel(imdsTest.Files)

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq

pxLayer = pixelClassificationLayer('Name','labels','ClassNames',tbl.Name,'ClassWeights',classWeights);
lgraph = removeLayers(lgraph,'labels');
lgraph = addLayers(lgraph, pxLayer);
lgraph = connectLayers(lgraph,'softmax','labels');

% lgraph = createLgraphUsingConnections(OriginNet.Layers,OriginNet.Connections)

% pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain,...
%     'DataAugmentation',augmenter,'OutputSize',[240,352,3]);

learnRate=1*1e-3; %learnRate=1e-6;
miniBatchSize=8;
valFrequency = floor(421/miniBatchSize);
MaxEpochs=5000;
options = trainingOptions('sgdm', ...
    'InitialLearnRate',learnRate,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',4000,...
    'LearnRateDropFactor',0.1,...
    'MaxEpochs',MaxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress',...
    'Verbose',true, ...   % For txtal progress checking
    'VerboseFrequency',valFrequency,...
    'ExecutionEnvironment','gpu');
%     'Plots','training-progress',...

[OriginNet, info] = trainNetwork(pximds,lgraph,options);


I = read(imdsTest);
C = semanticseg(I, OriginNet);

B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.5);
figure(1)
imshow(B)
pixelLabelColorbar(cmap, classes);

figure(2)
imshow(I)

expectedResult = read(pxdsTest);
actual = uint8(C{:});
expected = uint8(expectedResult);
imshowpair(actual, expected)

iou = jaccard(C, expectedResult);
table(classes,iou)

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