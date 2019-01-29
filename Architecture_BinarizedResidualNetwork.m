%===========================:Prepare Data:===========================%
%Download the CIFAR-10 data set.
datadir = tempdir;
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
helperCIFAR10Data.download(url,datadir);

%Load the CIFAR-10 images and use the CIFAR-10 test images for network validation.
[XTrain,YTrain,XValidation,YValidation] = helperCIFAR10Data.load(datadir);
% load('G:\Work\NeuralNetwork\QuantizedNeuralNetwork\BinarizedNeuralNetwork\DevelopmentHistoryBackup\Phase6_Stabilization\CIFARFACE5\matlab.mat')

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
    'DataAugmentation',imageAugmenter, ...
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
    
    Binarizedconvolution2dLayer(1,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv0')
    batchNormalizationLayer('Name','BatchNorm0')
    SignumActivation('Sign0')
    
    Binarizedconvolution2dLayer(1,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv1')
    batchNormalizationLayer('Name','BatchNorm1')
    SignumActivation('Sign1')
    Binarizedconvolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv2')
    batchNormalizationLayer('Name','BatchNorm2')
    SignumActivation('Sign2')
    Binarizedconvolution2dLayer(1,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv3')
    batchNormalizationLayer('Name','BatchNorm3')
    SignumActivation('Sign3')
    
    additionLayer(2,'Name','add1')
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool1')

    Binarizedconvolution2dLayer(1,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv4')
    batchNormalizationLayer('Name','BatchNorm4')
    SignumActivation('Sign4')
    Binarizedconvolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv5')
    batchNormalizationLayer('Name','BatchNorm5')
    SignumActivation('Sign5')
    Binarizedconvolution2dLayer(1,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv6')
    batchNormalizationLayer('Name','BatchNorm6')
    SignumActivation('Sign6')

    additionLayer(2,'Name','add2')
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool2')
    

    Binarizedconvolution2dLayer(1,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv7')
    batchNormalizationLayer('Name','BatchNorm7')
    SignumActivation('Sign7')
    Binarizedconvolution2dLayer(3,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv8')
    batchNormalizationLayer('Name','BatchNorm8')
    SignumActivation('Sign8')
    Binarizedconvolution2dLayer(1,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv9')
    batchNormalizationLayer('Name','BatchNorm9')
    SignumActivation('Sign9')
    
    additionLayer(2,'Name','add3')
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool3')
    

    Binarizedconvolution2dLayer(1,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv10')
    batchNormalizationLayer('Name','BatchNorm10')
    SignumActivation('Sign10')
    Binarizedconvolution2dLayer(3,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv11')
    batchNormalizationLayer('Name','BatchNorm11')
    SignumActivation('Sign11')
    Binarizedconvolution2dLayer(1,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv12')
    batchNormalizationLayer('Name','BatchNorm12')
    SignumActivation('Sign12')
    
    additionLayer(2,'Name','add4')
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
    
    additionLayer(2,'Name','add5')
    
    Binarizedconvolution2dLayer(1,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv16')
    batchNormalizationLayer('Name','BatchNorm16')
    SignumActivation('Sign16')
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv17')
    batchNormalizationLayer('Name','BatchNorm17')
    SignumActivation('Sign17')
    Binarizedconvolution2dLayer(1,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv18')
    batchNormalizationLayer('Name','BatchNorm18')
    SignumActivation('Sign18')
    
    additionLayer(2,'Name','add6')
    
    Binarizedconvolution2dLayer(1,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv19')
    batchNormalizationLayer('Name','BatchNorm19')
    SignumActivation('Sign19')
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv20')
    batchNormalizationLayer('Name','BatchNorm20')
    SignumActivation('Sign20')
    Binarizedconvolution2dLayer(1,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv21')
    batchNormalizationLayer('Name','BatchNorm21')
    SignumActivation('Sign21')
    
    additionLayer(2,'Name','add7')
    
    Binarizedconvolution2dLayer(1,catsize,'Stride',1,'Name','binAffine1','BiasLearnRateFactor',1,'BiasL2Factor',1) %1/10
    batchNormalizationLayer('Name','BatchNorm22')
    SignumActivation('SignAve')
    %======= :Classifier: =======%
    %     dropoutLayer('Name','drop1')
    averagePooling2dLayer(imageSize(1,1)/16,'Name','avePool1')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
    ];

% load('G:\Work\NeuralNetwork\QuantizedNeuralNetwork\BinarizedNeuralNetwork\DevelopmentHistoryBackup\Phase6_Stabilization\VGGFACE2\45people_600\OriginNet.mat')
% for tmplayerbit=[2,6,9,13,16,20]
%     layers(tmplayerbit,1).Weights=FaceNet8.Layers(tmplayerbit, 1).Weights  ;
% end

% bitConv=[];bitBatch=[];
% for tmplayerbit=1:size(ForSegNet2.Layers,1)
%     if ismethod(ForSegNet2.Layers(tmplayerbit,1),'BinarizedConvolution2DLayerFixer')
%         bitConv=[bitConv;tmplayerbit];
%     elseif ismethod(ForSegNet2.Layers(tmplayerbit,1),'BatchNormalizationLayer')
%         bitBatch=[bitBatch;tmplayerbit];
%     end
% end
% countConv=1;countBatch=1;
% for tmplayerbit=1:size(layers,1)
%     if ismethod(layers(tmplayerbit,1),'BinarizedConvolution2DLayerFixer')
%         layers(tmplayerbit,1).Weights=ForSegNet2.Layers(bitConv(countConv,1), 1).Weights  ;
%         countConv=countConv+1;
%     elseif ismethod(layers(tmplayerbit,1),'BatchNormalizationLayer')
%         layers(tmplayerbit,1)=ForSegNet2.Layers(bitBatch(countBatch,1), 1);
%         countBatch=countBatch+1;
%     end
% end


custom_architecture = layerGraph(layers);


custom_architecture = connectLayers(custom_architecture,'Sign0','add1/in2');
custom_architecture = connectLayers(custom_architecture,'MaxPool1','add2/in2');
% custom_architecture = connectLayers(custom_architecture,'MaxPool2','add3/in2');
custom_architecture = connectLayers(custom_architecture,'MaxPool3','add4/in2');
custom_architecture = connectLayers(custom_architecture,'add5','add6/in2');
custom_architecture = connectLayers(custom_architecture,'add6','add7/in2');


skip1 = [
    convolution2dLayer(1,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','skipConv1')
    batchNormalizationLayer('Name','skipBN1')];
custom_architecture = addLayers(custom_architecture,skip1);
custom_architecture = connectLayers(custom_architecture,'MaxPool2','skipConv1');
custom_architecture = connectLayers(custom_architecture,'skipBN1','add3/in2');

skip2 = [
    convolution2dLayer(1,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','skipConv2')
    batchNormalizationLayer('Name','skipBN2')];
custom_architecture = addLayers(custom_architecture,skip2);
custom_architecture = connectLayers(custom_architecture,'MaxPool4','skipConv2');
custom_architecture = connectLayers(custom_architecture,'skipBN2','add5/in2');

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
miniBatchSize = 800; %128
learnRate = 0.0001;%0.0005*miniBatchSize/128; %0.005
valFrequency = floor(size(XTrain,4)/miniBatchSize);
options1 = trainingOptions('adam', ...  %  sgdm,adam,rmsprop
    'InitialLearnRate',learnRate,...
    'MaxEpochs',250, ...
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
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',250,'ExecutionEnvironment','gpu');

options2 = trainingOptions('adam', ...  %  sgdm,adam,rmsprop
    'InitialLearnRate',learnRate,...
    'MaxEpochs',60, ...
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
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',20,'ExecutionEnvironment','gpu');

BinarizedResidualNettmp = trainNetwork(augimdsTrain,custom_architecture,options1);
% SyncNet8 = trainNetwork(augimdsTrain,Nettmp,options);
lgraph = createLgraphUsingConnections(BinarizedResidualNettmp.Layers,BinarizedResidualNettmp.Connections);
BinarizedResidualNet = trainNetwork(augimdsTrain,lgraph,options2);