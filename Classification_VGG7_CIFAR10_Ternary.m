%===========================:Prepare Data:===========================%
%Download the CIFAR-10 data set.
datadir = tempdir;
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
helperCIFAR10Data.download(url,datadir);

%Load the CIFAR-10 images and use the CIFAR-10 test images for network validation.
[XTrain,YTrain,XValidation,YValidation] = helperCIFAR10Data.load(datadir);

catsize=size(unique(YValidation),1);

imageSize = [32 32 3];
pixelRange = [-3 3];
% rotateRange= [-3 3];
% scaleRange=[0.9 1.1];
% shearRange= [-5 5];

imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

augimdsTrain = augmentedImageDatastore(imageSize,XTrain,YTrain, ...
    'DataAugmentation',imageAugmenter);


%====================:Define Network Architecture:====================%

%1. Struct Basis Architecture
netWidth = 16;
layers = [
    imageInputLayer(imageSize,'Name','input','Normalization','none') %32*32*3
    
    Binarizedconvolution2dLayer(3,4*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv1')
    batchNormalizationLayer('Name','BatchNorm1')
    TernaryActivation(rand([1 1 64]),-rand([1 1 64]),'Sign1')
    Binarizedconvolution2dLayer(3,4*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv2')
    batchNormalizationLayer('Name','BatchNorm2')
    TernaryActivation(rand([1 1 64]),-rand([1 1 64]),'Sign2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool1')
        
    Binarizedconvolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv3')
    batchNormalizationLayer('Name','BatchNorm3')
    TernaryActivation(rand([1 1 128]),-rand([1 1 128]),'Sign3')
    Binarizedconvolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv4')
    batchNormalizationLayer('Name','BatchNorm4')
    TernaryActivation(rand([1 1 128]),-rand([1 1 128]),'Sign4')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool2')
    
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv5')
    batchNormalizationLayer('Name','BatchNorm5')
    TernaryActivation(rand([1 1 256]),-rand([1 1 256]),'Sign5')
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv6')
    batchNormalizationLayer('Name','BatchNorm6')
    TernaryActivation(rand([1 1 256]),-rand([1 1 256]),'Sign6')
      
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool3')
    averagePooling2dLayer(imageSize(1,1)/8,'Name','avePool1')
    batchNormalizationLayer('Name','BatchNorm7')
    TernaryActivation(rand([1 1 256]),-rand([1 1 256]),'SignAve')
    %======= :Classifier: =======%
    %     dropoutLayer('Name','drop1')
    Binarizedconvolution2dLayer(1,catsize,'Stride',1,'Name','binAffine1','BiasLearnRateFactor',1,'BiasL2Factor',1) %1/10
    batchNormalizationLayer('Name','BatchNorm16')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
    ];

lgraph = layerGraph(layers);

%=========================:Train Network:============================%
miniBatchSize = 1000; %128

learnRate = 0.0001;%0.0005*miniBatchSize/128; %0.005
valFrequency = 50;
options1 = trainingOptions('adam', ...  %  sgdm,adam,rmsprop
    'InitialLearnRate',learnRate,...
    'MaxEpochs',20, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch',...
    'Plots','training-progress',...
    'Verbose',true, ...   % For txtal progress checking
    'VerboseFrequency',valFrequency,...
    'LearnRateSchedule','piecewise',...
    'ExecutionEnvironment','multi-gpu');

BaselineNetwork = trainNetwork(augimdsTrain,lgraph,options1);

