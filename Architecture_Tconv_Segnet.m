%===========================:Prepare Data:===========================%


imageSize=[360 480 3];



%====================:Define Network Architecture:====================%

%1. Struct Basis Architecture
netWidth = 16;
layers = [
    imageInputLayer(imageSize,'Name','input','Normalization','none') %32*32*3
    
    Binarizedconvolution2dLayer(3,8*netWidth,'Padding','same','Stride',2,'BiasLearnRateFactor',0,'Name','binConv1')
    batchNormalizationLayer('Name','BatchNorm1')
    SignumActivation('Sign1')
    Binarizedconvolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv2')
    batchNormalizationLayer('Name','BatchNorm2')
    SignumActivation('Sign2')
    Binarizedconvolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv3')
    batchNormalizationLayer('Name','BatchNorm3')
    SignumActivation('Sign3')
    
    
   
        
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',2,'BiasLearnRateFactor',0,'Name','binConv10')
    batchNormalizationLayer('Name','BatchNorm10')
    SignumActivation('Sign10')
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv11')
    batchNormalizationLayer('Name','BatchNorm11')
    SignumActivation('Sign11')
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv12')
    batchNormalizationLayer('Name','BatchNorm12')
    SignumActivation('Sign12')
    
    averagePooling2dLayer(imageSize(1,1)/8,'Name','avePool1')
    SignumActivation('SignAve')
    %======= :Classifier: =======%
    %     dropoutLayer('Name','drop1')
    Binarizedconvolution2dLayer(1,catsize,'Stride',1,'Name','binAffine1','BiasLearnRateFactor',1,'BiasL2Factor',1) %1/10
    batchNormalizationLayer('Name','BatchNorm16')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
    ];

custom_architecture = layerGraph(layers);

%=========================:Train Network:============================%
miniBatchSize = 1000; %128

% augimdsTrain.MiniBatchSize=miniBatchSize;

learnRate = 0.0001;%0.0005*miniBatchSize/128; %0.005
valFrequency = floor(size(XTrain,4)/miniBatchSize);
options1 = trainingOptions('adam', ...  %  sgdm,adam,rmsprop
    'InitialLearnRate',learnRate,...
    'MaxEpochs',200, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch',...
    'Plots','training-progress',...
    'Verbose',true, ...   % For txtal progress checking
    'VerboseFrequency',valFrequency,...
    'ValidationData',{XValidation,YValidation},...
    'ValidationFrequency',valFrequency,...
    'ValidationPatience',Inf,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',200,'ExecutionEnvironment','gpu');

options2 = trainingOptions('adam', ...  %  sgdm,adam,rmsprop
    'InitialLearnRate',learnRate,...
    'MaxEpochs',70, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch',...
    'Plots','training-progress',...
    'Verbose',true, ...   % For txtal progress checking
    'VerboseFrequency',valFrequency,...
    'ValidationData',{XValidation,YValidation},...
    'ValidationFrequency',valFrequency,...
    'ValidationPatience',Inf,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',7,'ExecutionEnvironment','gpu');

CIFAR10_12layer_tmp = trainNetwork(augimdsTrain,custom_architecture,options1);
% SyncNet8 = trainNetwork(augimdsTrain,Nettmp,options);

CIFAR10_12layer = trainNetwork(augimdsTrain,CIFAR10_12layer_tmp.Layers,options2);
