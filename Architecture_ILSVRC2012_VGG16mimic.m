%===========================:Prepare Data:===========================%
catsize=1000;
imds = imageDatastore('D:\ILSVRC2012\ILSVRC2012_img_train','IncludeSubfolders',true,'LabelSource','foldernames');


imageSize = [240 352 3] ; %[360 480 3];
pixelRange = [-3 3];
% rotateRange= [-3 3];
% scaleRange=[0.9 1.1];
% shearRange= [-5 5];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(imageSize,imds, ...
    'DataAugmentation',imageAugmenter, ...
    'ColorPreprocessing','gray2rgb',...
    'OutputSizeMode','resize');

%'resize','randcrop'

%     'RandXScale',scaleRange,'RandYScale',scaleRange, ...
%     'RandRotation',rotateRange, ...
%     'RandXShear',shearRange,'RandYShear',shearRange, ...
%     'RandYReflection',true, ...
%     'RandXReflection',true, ...


%====================:Define Network Architecture:====================%

%1. Struct Basis Architecture
netWidth = 16;
layers = [
    imageInputLayer(imageSize,'Name','input','Normalization','none') %32*32*3
    
    Binarizedconvolution2dLayer(3,4*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv1')
    batchNormalizationLayer('Name','BatchNorm1')
    SignumActivation('Sign1')
    Binarizedconvolution2dLayer(3,4*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv2')
    batchNormalizationLayer('Name','BatchNorm2')
    SignumActivation('Sign2')
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool1')
    
    Binarizedconvolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv3')
    batchNormalizationLayer('Name','BatchNorm3')
    SignumActivation('Sign3')
    Binarizedconvolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv4')
    batchNormalizationLayer('Name','BatchNorm4')
    SignumActivation('Sign4')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool2')
    
    
    
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv5')
    batchNormalizationLayer('Name','BatchNorm5')
    SignumActivation('Sign5')
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv6')
    batchNormalizationLayer('Name','BatchNorm6')
    SignumActivation('Sign6')
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv7')
    batchNormalizationLayer('Name','BatchNorm7')
    SignumActivation('Sign7')
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool3')
    

    Binarizedconvolution2dLayer(3,32*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv8')
    batchNormalizationLayer('Name','BatchNorm8')
    SignumActivation('Sign8')
    Binarizedconvolution2dLayer(3,32*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv9')
    batchNormalizationLayer('Name','BatchNorm9')
    SignumActivation('Sign9')
    Binarizedconvolution2dLayer(3,32*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv10')
    batchNormalizationLayer('Name','BatchNorm10')
    SignumActivation('Sign10')
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool4')
    

    Binarizedconvolution2dLayer(3,32*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv11')
    batchNormalizationLayer('Name','BatchNorm11')
    SignumActivation('Sign11')
    Binarizedconvolution2dLayer(3,32*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv12')
    batchNormalizationLayer('Name','BatchNorm12')
    SignumActivation('Sign12')
    Binarizedconvolution2dLayer(3,32*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv13')
    batchNormalizationLayer('Name','BatchNorm13')
    SignumActivation('Sign13')

    
    averagePooling2dLayer(floor([imageSize(1,1)/16,imageSize(1,2)/16]),'Name','avePool1')
    SignumActivation('SignAve')
    %======= :Classifier: =======%
    %     dropoutLayer('Name','drop1')
    Binarizedconvolution2dLayer(1,catsize,'Stride',1,'Name','binAffine1','BiasLearnRateFactor',1,'BiasL2Factor',1) %1/10
    batchNormalizationLayer('Name','BatchNorm16')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
    ];

% load('G:\Work\NeuralNetwork\QuantizedNeuralNetwork\BinarizedNeuralNetwork\DevelopmentHistoryBackup\Phase6_Stabilization\VGGFACE2\45people_600\OriginNet.mat')
for tmplayerbit=[2,4,7,9,12,14,16,19,21,23,26,28,30]
    layers(tmplayerbit,1).Weights=nettmp.Layers(tmplayerbit, 1).Weights  ;
end
custom_architecture = layerGraph(layers);


%=========================:Train Network:============================%
miniBatchSize = 10; %128
learnRate = 0.0001;%0.0005*miniBatchSize/128; %0.005
valFrequency = 1000;
options1 = trainingOptions('adam', ...  %  sgdm,adam,rmsprop
    'InitialLearnRate',learnRate,...
    'MaxEpochs',300, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch',...
    'Plots','training-progress',...
    'Verbose',true, ...   % For txtal progress checking
    'VerboseFrequency',valFrequency,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',300,'ExecutionEnvironment','gpu');

ForSegNet1 = trainNetwork(augimdsTrain,custom_architecture,options1);
