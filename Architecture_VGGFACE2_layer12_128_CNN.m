BNNpath='C:\Works\NeuralNetworks\Matlab2018b_BNN';
addpath(genpath(BNNpath))

%===========================:Prepare Data:===========================%
imageSize = [32 32 3] ; %[360 480 3];
imageFolder='C:\Works\NeuralNetworks\Datasets\CustomVGGFace2\Class300\imagesize32\trainset';
validesetFolder='C:\Works\NeuralNetworks\Datasets\CustomVGGFace2\Class300\imagesize32\testset';

imds = imageDatastore(imageFolder,'IncludeSubfolders',true,'LabelSource','foldernames');
testset_imds = imageDatastore(validesetFolder,'IncludeSubfolders',true,'LabelSource','foldernames');
catsize=size(unique(imds.Labels),1);

pixelRange = [-3 3];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(imageSize,imds, ...
    'DataAugmentation',imageAugmenter, ...
    'ColorPreprocessing','gray2rgb',...
    'DispatchInBackground',true,...
    'OutputSizeMode','resize');

%====================:Define Network Architecture:====================%
%1. Struct Basis Architecture
netWidth = 16;
layers = [
    imageInputLayer(imageSize,'Name','input','Normalization','none') %32*32*3
    
    convolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv1')
    batchNormalizationLayer('Name','BatchNorm1')
    reluLayer('Name','relu1')
    convolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv2')
    batchNormalizationLayer('Name','BatchNorm2')
    reluLayer('Name','relu2')
    convolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv3')
    batchNormalizationLayer('Name','BatchNorm3')
    reluLayer('Name','relu3')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool1')
    
    convolution2dLayer(3,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv4')
    batchNormalizationLayer('Name','BatchNorm4')
    reluLayer('Name','relu4')
    convolution2dLayer(3,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv5')
    batchNormalizationLayer('Name','BatchNorm5')
    reluLayer('Name','relu5')
    convolution2dLayer(3,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv6')
    batchNormalizationLayer('Name','BatchNorm6')
    reluLayer('Name','relu6')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool2')
    
    convolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv7')
    batchNormalizationLayer('Name','BatchNorm7')
    reluLayer('Name','relu7')
    convolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv8')
    batchNormalizationLayer('Name','BatchNorm8')
    reluLayer('Name','relu8')
    convolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv9')
    batchNormalizationLayer('Name','BatchNorm9')
    reluLayer('Name','relu9')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool3')
    
    convolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv10')
    batchNormalizationLayer('Name','BatchNorm10')
    reluLayer('Name','relu10')
    convolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv11')
    batchNormalizationLayer('Name','BatchNorm11')
    reluLayer('Name','relu11')
    convolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv12')
    batchNormalizationLayer('Name','BatchNorm12')
    reluLayer('Name','relu12')
    
    averagePooling2dLayer(imageSize(1,1)/8,'Name','avePool1')
    reluLayer('Name','reluAve')
    %======= :Classifier: =======%
    %     dropoutLayer('Name','drop1')
    convolution2dLayer(1,catsize,'Stride',1,'Name','binAffine1','BiasLearnRateFactor',1,'BiasL2Factor',1) %1/10
    batchNormalizationLayer('Name','BatchNorm16')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
    ];

custom_architecture = layerGraph(layers);

%=========================:Train Network:============================%
miniBatchSize = 1000; %128
learnRate = 0.0001;%0.0005*miniBatchSize/128; %0.005
% valFrequency = 399;
valFrequency = floor(size(imds.Files,1)/miniBatchSize);
options1 = trainingOptions('adam', ...  %  sgdm,adam,rmsprop
    'InitialLearnRate',learnRate,...
    'MaxEpochs',300, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch',...
    'Plots','training-progress',...
    'Verbose',true, ...   % For txtal progress checking
    'VerboseFrequency',valFrequency,...
    'ValidationData',testset_imds,...
    'ValidationFrequency',valFrequency,...
    'ValidationPatience',Inf,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',300,'ExecutionEnvironment','multi-gpu');

Net= trainNetwork(augimdsTrain,custom_architecture,options1);
