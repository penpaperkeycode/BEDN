%=========Dataset==========%
imageSize = [360 480 3]; %[240,352,3];
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

imageURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip';
labelURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip';

outputFolder = fullfile('E:\Datasets\','CamVid'); 
labelsZip = fullfile(outputFolder,'labels.zip');
imagesZip = fullfile(outputFolder,'images.zip');

if ~exist(labelsZip, 'file') || ~exist(imagesZip,'file')   
    mkdir(outputFolder)
       
    disp('Downloading 16 MB CamVid dataset labels...'); 
    websave(labelsZip, labelURL);
    unzip(labelsZip, fullfile(outputFolder,'labels'));
    
    disp('Downloading 557 MB CamVid dataset images...');  
    websave(imagesZip, imageURL);       
    unzip(imagesZip, fullfile(outputFolder,'images'));    
end

% outputFolder = fullfile(tempdir,'CamVid');
outputFolder = fullfile('E:\Datasets\','CamVid');
imgDir = fullfile(outputFolder,'images','701_StillsRaw_full');
imds = imageDatastore(imgDir);
labelDir = fullfile(outputFolder,'labels');
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);


imageFolder = fullfile(outputFolder,'imagesResized',filesep);
imds = resizeCamVidImages(imds,imageFolder,imageSize);

labelFolder = fullfile(outputFolder,'labelsResized',filesep);
pxds = resizeCamVidPixelLabels(pxds,labelFolder,imageSize);

[imdsTrain,imdsTest,pxdsTrain,pxdsTest] = partitionCamVidData2(imds,pxds);


pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain);

%=========Model==========%
netWidth = 16;

lgraph=segnetLayers(imageSize,numClasses,'vgg16');

%=========Training==========%
learnRate=1*1e-3; %learnRate=1e-6;
miniBatchSize=5;
valFrequency = floor(421/miniBatchSize);
MaxEpochs=5000;
options = trainingOptions('sgdm', ...
    'InitialLearnRate',learnRate,...
    'MaxEpochs',MaxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress',...
    'Verbose',true, ...   % For txtal progress checking
    'VerboseFrequency',valFrequency,...
    'ExecutionEnvironment','gpu');
%     'Plots','training-progress',...

[OriginNet, info] = trainNetwork(pximds,lgraph,options);


%=========Evaluation==========%
pxdsResults = semanticseg(imdsTest,OriginNet,'WriteLocation',tempdir,'Verbose',false,'MiniBatchSize',miniBatchSize);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',true);

metrics.DataSetMetrics

metrics.ClassMetrics