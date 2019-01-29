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
rotateRange= [-10 10];
scaleRange=[0.95 1.25];
% shearRange= [-5 5];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(imageSize,XTrain,YTrain, ...
    'DataAugmentation',imageAugmenter, ...
    'OutputSizeMode','randcrop');

%     'RandXScale',scaleRange,'RandYScale',scaleRange, ...
%     'RandRotation',rotateRange, ...
%     'RandXShear',shearRange,'RandYShear',shearRange, ...
%     'RandYReflection',true, ...
%====================:Define Network Architecture:====================%

%1. Struct Basis Architecture
netWidth = 16;
layers = [
    imageInputLayer(imageSize,'Name','input','Normalization','none') %32*32*3
    
    
    Binarizedconvolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv1')
    batchNormalizationLayer('Name','BatchNorm1')
    SignumActivation('Sign1')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool1')
    Binarizedconvolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv2')
    batchNormalizationLayer('Name','BatchNorm2')
    SignumActivation('Sign2')
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv3')
    batchNormalizationLayer('Name','BatchNorm3')
    SignumActivation('Sign3')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool2')
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv4')
    batchNormalizationLayer('Name','BatchNorm4')
    SignumActivation('Sign4')
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv5')
    batchNormalizationLayer('Name','BatchNorm5')
    SignumActivation('Sign5')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool3')
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv6')
    batchNormalizationLayer('Name','BatchNorm6')
        SignumActivation('Sign6')
    
    averagePooling2dLayer(imageSize(1,1)/8,'Name','avePool1')
    
    SignumActivation('Sign9')
    %======= :Classifier: =======%
    %     dropoutLayer('Name','drop1')
    Binarizedconvolution2dLayer(1,catsize,'Stride',1,'Name','binAffine10','BiasLearnRateFactor',1,'BiasL2Factor',1) %1/10
    batchNormalizationLayer('Name','BatchNorm9')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
    ];

custom_architecture = layerGraph(layers);

% figure('Units','normalized','Position',[0 0.1 0.3 0.8]);
% plot(custom_architecture)
% grid on
% title('Trained Network Architecture')
% ax = gca;
% outerpos = ax.OuterPosition;
% ti = ax.TightInset;
% left = outerpos(1) + ti(1);
% bottom = outerpos(2) + ti(2);
% ax_width = outerpos(3) - ti(1) - ti(3);
% ax_height = outerpos(4) - ti(2) - ti(4);
% ax.Position = [left bottom ax_width ax_height];
% % axis off
% set(gca,'XTickLabel',[],'YTickLabel',[]) %Axis number off
% fig = gcf;
% fig.PaperPositionMode = 'auto';
% fig_pos = fig.PaperPosition;
% fig.PaperSize = [fig_pos(3) fig_pos(4)];

%=========================:Train Network:============================%
miniBatchSize = 250; %128
learnRate = 0.0001;%0.0005*miniBatchSize/128; %0.005
valFrequency = floor(size(XTrain,4)/miniBatchSize);
options = trainingOptions('adam', ...  %  sgdm,adam,rmsprop
    'InitialLearnRate',learnRate,...
    'MaxEpochs',220, ...   
    'MiniBatchSize',miniBatchSize, ...
    'VerboseFrequency',valFrequency,...
    'Shuffle','every-epoch',...
    'Plots','training-progress',...
    'Verbose',true, ...   % For txtal progress checking
    'VerboseFrequency',valFrequency,...
    'ValidationData',{XValidation,YValidation},...
    'ValidationFrequency',valFrequency,...
    'ValidationPatience',Inf,...
    'LearnRateSchedule','piecewise',...
    'CheckpointPath','D:\cnncheckpoint',...
    'LearnRateDropFactor',0.1,...
    'CheckpointPath','D:\cnncheckpoint',...
    'LearnRateDropPeriod',220,'ExecutionEnvironment','gpu');  

% SyncNet7= trainNetwork(augimdsTrain,net.Layers,options);
CIFARNet7 = trainNetwork(augimdsTrain,custom_architecture,options);

options2 = trainingOptions('adam', ...  %  sgdm,adam,rmsprop
    'InitialLearnRate',learnRate,...
    'MaxEpochs',220, ...   
    'MiniBatchSize',miniBatchSize, ...
    'VerboseFrequency',valFrequency,...
    'Shuffle','every-epoch',...
    'Plots','training-progress',...
    'Verbose',true, ...   % For txtal progress checking
    'VerboseFrequency',valFrequency,...
    'ValidationData',{XValidation,YValidation},...
    'ValidationFrequency',valFrequency,...
    'ValidationPatience',Inf,...
    'LearnRateSchedule','piecewise',...
    'CheckpointPath','D:\cnncheckpoint',...
    'LearnRateDropFactor',0.1,...
    'CheckpointPath','D:\cnncheckpoint',...
    'LearnRateDropPeriod',22,'ExecutionEnvironment','gpu');  

% SyncNet7= trainNetwork(augimdsTrain,net.Layers,options);
CIFARNet7_2 = trainNetwork(augimdsTrain,CIFARNet7.Layers,options2);



