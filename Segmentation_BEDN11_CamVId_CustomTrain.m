imageSize = [360 480 3];
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

root1=pwd;
outputFolder = fullfile(root1,'2.Dataset\CamVid');
TrainDB=fullfile(outputFolder,'train');
TrainLabel = fullfile(outputFolder,'trainannot');
ValDB=fullfile(outputFolder,'val');
ValLabel = fullfile(outputFolder,'valannot');
TestDB=fullfile(outputFolder,'test');
TestLabel = fullfile(outputFolder,'testannot');

pxds = pixelLabelDatastore(TrainLabel,classes,labelIDs);
labelIDs = 1:numel(pxds.ClassNames);
% labelIDs=labelIDs-1;
% pxds = pixelLabelDatastore(TrainLabel,classes,labelIDs);


imdsTrain = imageDatastore(TrainDB);
pxdsTrain = pixelLabelDatastore(TrainLabel,pxds.ClassNames,labelIDs);
imdsVal = imageDatastore(ValDB);
pxdsVal = pixelLabelDatastore(ValLabel,pxds.ClassNames,labelIDs);
imdsTest = imageDatastore(TestDB);
pxdsTest = pixelLabelDatastore(TestLabel,pxds.ClassNames,labelIDs);

pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain);
pximdsVal = pixelLabelImageDatastore(imdsVal,pxdsVal);

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
    ];

layers=[
    FrontLayer;
    TailLayer
    ];

lgraph=layerGraph(layers);

dlnet = dlnetwork(lgraph);

numEpochs=5000;
miniBatchSize=8;
initialLearnRate=1*1e-3; %learnRate=1e-6;
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

numObservations = numel(imdsTrain.Files); %numel(YTrain);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);
iteration = 0;
start = tic;

pximds.MiniBatchSize = miniBatchSize;

% Loop over epochs.
for epoch = 1:numEpochs
    % Reset and shuffle the datastore.
    reset(pximds);
    pximds = shuffle(pximds);
    
    % Loop over mini-batches.
    while hasdata(pximds)
        iteration = iteration + 1;
        
        % Read a mini-batch of data.
        pximdsXBatch = read(pximds);
        workerXBatch = cat(4,pximdsXBatch.inputImage{:});
        
        % Normalize the images.
        workerXBatch =  single(workerXBatch) ./ 255;
        
        workerY= cat(4,pximdsXBatch.pixelLabelImage{:});
        workerY=double(workerY);
        
        % Convert the mini-batch of data to dlarray.
        dlworkerX = dlarray(workerXBatch,'SSCB');
        
        % If training on GPU, then convert data to gpuArray.
        if executionEnvironment == "gpu"
            dlworkerX = gpuArray(dlworkerX);
        end
        
        % Evaluate the model gradients and loss on the worker.
        [workerGradients,dlworkerLoss,workerState] = dlfeval(@modelGradients_spatialentropy,dlnet,dlworkerX,workerY);
        dlnet.State = workerState;
        
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);
        
        % Update the network parameters using the SGDM optimizer.
        [dlnet.Learnables,velocity] = sgdmupdate(dlnet.Learnables,workerGradients,velocity,learnRate,momentum);
        
        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain,iteration,double(gather(extractdata(dlworkerLoss))))
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
end
