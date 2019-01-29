%===========================:Prepare Data:===========================%
imds = imageDatastore('D:\VGGFace2\vggface2_train\train','IncludeSubfolders',true,'LabelSource','foldernames');
catsize=size(unique(imds.Labels),1);

imageSize = [128 128 3] ; %[360 480 3];
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
    'DispatchInBackground',true,...
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
    
    Binarizedconvolution2dLayer(1,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv1')
    batchNormalizationLayer('Name','BatchNorm1')
    SignumActivation('Sign1')
    Binarizedconvolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv2')
    batchNormalizationLayer('Name','BatchNorm2')
    SignumActivation('Sign2')
    Binarizedconvolution2dLayer(1,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv3')
    batchNormalizationLayer('Name','BatchNorm3')
    SignumActivation('Sign3')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool1')
    

    Binarizedconvolution2dLayer(1,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv4')
    batchNormalizationLayer('Name','BatchNorm4')
    SignumActivation('Sign4')
    Binarizedconvolution2dLayer(3,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv5')
    batchNormalizationLayer('Name','BatchNorm5')
    SignumActivation('Sign5')
    Binarizedconvolution2dLayer(1,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv6')
    batchNormalizationLayer('Name','BatchNorm6')
    SignumActivation('Sign6')

    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool2')
    

    Binarizedconvolution2dLayer(1,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv7')
    batchNormalizationLayer('Name','BatchNorm7')
    SignumActivation('Sign7')
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv8')
    batchNormalizationLayer('Name','BatchNorm8')
    SignumActivation('Sign8')
    Binarizedconvolution2dLayer(1,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv9')
    batchNormalizationLayer('Name','BatchNorm9')
    SignumActivation('Sign9')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool3')
    

    Binarizedconvolution2dLayer(1,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv10')
    batchNormalizationLayer('Name','BatchNorm10')
    SignumActivation('Sign10')
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv11')
    batchNormalizationLayer('Name','BatchNorm11')
    SignumActivation('Sign11')
    Binarizedconvolution2dLayer(1,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv12')
    batchNormalizationLayer('Name','BatchNorm12')
    SignumActivation('Sign12')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool4')
    
    Binarizedconvolution2dLayer(1,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv13')
    batchNormalizationLayer('Name','BatchNorm13')
    SignumActivation('Sign13')
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv14')
    batchNormalizationLayer('Name','BatchNorm14')
    SignumActivation('Sign14')
    Binarizedconvolution2dLayer(1,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv15')
    batchNormalizationLayer('Name','BatchNorm15')
    SignumActivation('Sign15')
    
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
% for tmplayerbit=[2,6,9,13,16,20]
%     layers(tmplayerbit,1).Weights=FaceNet8.Layers(tmplayerbit, 1).Weights  ;
% end
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
miniBatchSize = 40; %128
learnRate = 0.001;%0.0005*miniBatchSize/128; %0.005
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

% options2 = trainingOptions('adam', ...  %  sgdm,adam,rmsprop
%     'InitialLearnRate',learnRate,...
%     'MaxEpochs',200, ...
%     'MiniBatchSize',miniBatchSize, ...
%     'Shuffle','every-epoch',...
%     'Plots','training-progress',...
%     'Verbose',true, ...   % For txtal progress checking
%     'VerboseFrequency',valFrequency,...
%     'ValidationData',{XValidation,YValidation},...
%     'ValidationFrequency',valFrequency,...
%     'ValidationPatience',Inf,...
%     'LearnRateSchedule','piecewise',...
%     'LearnRateDropFactor',0.1,...
%     'LearnRateDropPeriod',20,'ExecutionEnvironment','gpu');

VGGFace2BNet16 = trainNetwork(augimdsTrain,custom_architecture,options1);
% SyncNet8 = trainNetwork(augimdsTrain,Nettmp,options);

% ForSegNet2 = trainNetwork(augimdsTrain,ForSegNet1.Layers,options2);
