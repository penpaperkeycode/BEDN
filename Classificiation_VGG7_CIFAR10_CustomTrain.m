%===========================:Prepare Data:===========================%
%Download the CIFAR-10 data set.
datadir = tempdir;
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
helperCIFAR10Data.download(url,datadir);

%Load the CIFAR-10 images and use the CIFAR-10 test images for network validation.
[XTrain,YTrain,XValidation,YValidation] = helperCIFAR10Data.load(datadir);
% load('G:\Work\NeuralNetwork\QuantizedNeuralNetwork\BinarizedNeuralNetwork\DevelopmentHistoryBackup\Phase6_Stabilization\CIFARFACE5\matlab.mat')

numClasses=size(unique(YValidation),1);
classes = categories(YTrain);
classes=string(classes);

imageSize = [32 32 3];
pixelRange = [-3 3];
% rotateRange= [-3 3];
% scaleRange=[0.9 1.1];
% shearRange= [-5 5];

imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

% imageAugmenter = imageDataAugmenter();
imdsTrain = augmentedImageDatastore(imageSize,XTrain,YTrain, ...
    'DataAugmentation',imageAugmenter);

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
    
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool3')
    averagePooling2dLayer(imageSize(1,1)/8,'Name','avePool1')
    SignumActivation('SignAve')
    %======= :Classifier: =======%
    %     dropoutLayer('Name','drop1')
    Binarizedconvolution2dLayer(1,numClasses,'Stride',1,'Name','binAffine1','BiasLearnRateFactor',1,'BiasL2Factor',1) %1/10
    batchNormalizationLayer('Name','BatchNorm16')
    softmaxLayer('Name','softmax')
    ];

lgraph = layerGraph(layers);

dlnet = dlnetwork(lgraph);

numEpochs=1;
miniBatchSize=1000;
initialLearnRate=1*1e-1; %learnRate=1e-6;
decay = 0.01;
momentum = 0.9;
plots = "training-progress";
executionEnvironment = "gpu"; %'auto'

if plots == "training-progress"
    figure
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on
end

velocity = []; %for SGDM

numObservations = numel(YTrain);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);
iteration = 0;
start = tic;

imdsTrain.MiniBatchSize = miniBatchSize;
% Loop over epochs.
for epoch = 1:numEpochs
    % Shuffle data.
    idx = randperm(numel(YTrain));
    XTrain = XTrain(:,:,:,idx);
    YTrain = YTrain(idx);
    
    % Loop over mini-batches.
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        
        % Read mini-batch of data and convert the labels to dummy
        % variables.
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        X = XTrain(:,:,:,idx);
        
        Y = zeros(numClasses, miniBatchSize, 'single');
        for c = 1:numClasses
            Y(c,YTrain(idx)==classes(c)) = 1;
        end
        
        % Convert mini-batch of data to dlarray.
        dlX = dlarray(single(X),'SSCB');
        
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients,state,loss] = dlfeval(@modelGradients,dlnet,dlX,Y);
        dlnet.State = state;
        
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);
        
        % Update the network parameters using the SGDM optimizer.
        [dlnet, velocity] = sgdmupdate(dlnet, gradients, velocity, learnRate, momentum);
        
        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
end
