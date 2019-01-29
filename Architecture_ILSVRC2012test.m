%===========================:Prepare Data:===========================%

imds = imageDatastore('D:\ILSVRC2012\ILSVRC2012_img_train_Class10','IncludeSubfolders',true,'LabelSource','foldernames');

tmplabelunique=unique(imds.Labels);
tmplabelnumb=size(tmplabelunique,1);
labels=[];
numTrain=1300;
for i=1:tmplabelnumb
    
   labels=[labels;i*ones(numTrain,1)];

end
labels=categorical(labels);
imds = imageDatastore('D:\ILSVRC2012\ILSVRC2012_img_train_Class10','IncludeSubfolders',true,'Labels',labels);

catsize=tmplabelnumb;

imageSize = [240 352 3];    %[224 224 3] ; %[360 480 3]; %[240 352 3];
pixelRange = [-3 3];

imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(imageSize,imds, ...
    'DataAugmentation',imageAugmenter, ...
    'ColorPreprocessing','gray2rgb',...
    'OutputSizeMode','resize');

%====================:Define Network Architecture:====================%

%1. Struct Basis Architecture
netWidth = 16;
layers = [
    imageInputLayer(imageSize,'Name','input','Normalization','none') 
    
    convolution2dLayer(1,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv1')
    batchNormalizationLayer('Name','BatchNorm1')
    SignumActivation('Sign1')
    convolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv2')
    batchNormalizationLayer('Name','BatchNorm2')
    SignumActivation('Sign2')
    convolution2dLayer(1,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv3')
    batchNormalizationLayer('Name','BatchNorm3')
    SignumActivation('Sign3')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool1')
    

    convolution2dLayer(1,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv4')
    batchNormalizationLayer('Name','BatchNorm4')
    SignumActivation('Sign4')
    convolution2dLayer(3,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv5')
    batchNormalizationLayer('Name','BatchNorm5')
    SignumActivation('Sign5')
    convolution2dLayer(1,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv6')
    batchNormalizationLayer('Name','BatchNorm6')
    SignumActivation('Sign6')

    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool2')
    

    convolution2dLayer(1,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv7')
    batchNormalizationLayer('Name','BatchNorm7')
    SignumActivation('Sign7')
    convolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv8')
    batchNormalizationLayer('Name','BatchNorm8')
    SignumActivation('Sign8')
    convolution2dLayer(1,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv9')
    batchNormalizationLayer('Name','BatchNorm9')
    SignumActivation('Sign9')
    
    averagePooling2dLayer(floor([imageSize(1,1)/4,imageSize(1,2)/4]),'Name','avePool1')
    SignumActivation('SignAve')
    %======= :Classifier: =======%
    %     dropoutLayer('Name','drop1')
    convolution2dLayer(1,catsize,'Stride',1,'Name','binAffine1','BiasLearnRateFactor',1,'BiasL2Factor',1) %1/10
    batchNormalizationLayer('Name','BatchNorm16')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
    ];

custom_architecture = layerGraph(layers);

%=========================:Train Network:============================%
miniBatchSize = 10; %128
learnRate = 0.00001;%0.0005*miniBatchSize/128; %0.005
valFrequency = 1000;
MaxEpoch=1;
options1 = trainingOptions('adam', ...  %  sgdm,adam,rmsprop
    'InitialLearnRate',learnRate,...
    'MaxEpochs',MaxEpoch, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch',...
    'Plots','training-progress',...
    'Verbose',true, ...   % For txtal progress checking
    'VerboseFrequency',valFrequency,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',MaxEpoch,'ExecutionEnvironment','gpu');

net = trainNetwork(augimdsTrain,custom_architecture,options1);

