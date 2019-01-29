lgraph = layerGraph(SyncNet7_2.Layers);
plot(lgraph)

layersToRemove = {
    'binAffine10'
    'BatchNorm9'
    'softmax'
    'classoutput'
    };

lgraph = removeLayers(lgraph, layersToRemove);


% Specify the number of classes the network should classify.
numClasses = 1;
numClassesPlusBackground = numClasses + 1;


% Define new classification layers.
newLayers = [
    BinarizedfullyConnectedLayer(numClassesPlusBackground, 'Name', 'rcnnFC')
    batchNormalizationLayer('Name','bnrcnnFC')
    softmaxLayer('Name', 'rcnnSoftmax')
    classificationLayer('Name', 'rcnnClassification')
    ];

lgraph = addLayers(lgraph, newLayers);

plot(lgraph)
% Connect the new layers to the network. 
lgraph = connectLayers(lgraph, 'Sign9', 'rcnnFC');

% Define the number of outputs of the fully connected layer.
numOutputs = 4 * numClasses;

% Create the box regression layers.
boxRegressionLayers = [
    BinarizedfullyConnectedLayer(numOutputs,'Name','rcnnBoxFC')
    batchNormalizationLayer('Name','bnrcnnBoxFC')
    rcnnBoxRegressionLayer('Name','rcnnBoxDeltas')
    ];

% Add the layers to the network.
lgraph = addLayers(lgraph, boxRegressionLayers);

% Connect the regression layers to the layer named 'avg_pool'.
lgraph = connectLayers(lgraph,'Sign9','rcnnBoxFC');

% Select a feature extraction layer.
featureExtractionLayer = 'MaxPool3';

% Disconnect the layers attached to the selected feature extraction layer.
lgraph = disconnectLayers(lgraph, featureExtractionLayer,'binConv6');
% lgraph = disconnectLayers(lgraph, featureExtractionLayer,'res5a_branch1');

% Add ROI max pooling layer.
outputSize = [4 4];
roiPool = roiMaxPooling2dLayer(outputSize,'Name','roiPool');
lgraph = addLayers(lgraph, roiPool);

% Connect feature extraction layer to ROI max pooling layer.
lgraph = connectLayers(lgraph, featureExtractionLayer,'roiPool/in');

% Connect the output of ROI max pool to the disconnected layers from above.
lgraph = connectLayers(lgraph, 'roiPool','binConv6');
% lgraph = connectLayers(lgraph, 'roiPool','res5a_branch1');

% Define anchor boxes.
anchorBoxes = [
    8 8
    8 16
    16 8
    16 16
    ];

% Create the region proposal layer.
proposalLayer = regionProposalLayer(anchorBoxes,'Name','regionProposal');

lgraph = addLayers(lgraph, proposalLayer);

% Number of anchor boxes.
numAnchors = size(anchorBoxes,1);

% Number of feature maps in coming out of the feature extraction layer. 
numFilters = 256;

rpnLayers = [
    Binarizedconvolution2dLayer(3, numFilters,'padding',[1 1],'Name','rpnConv3x3')
    batchNormalizationLayer('Name','bnrpnConv3x3')
    SignumActivation('rpnSign')
    ];

lgraph = addLayers(lgraph, rpnLayers);

% Connect to RPN to feature extraction layer.
lgraph = connectLayers(lgraph, featureExtractionLayer, 'rpnConv3x3');

plot(lgraph)


% Add RPN classification layers.
rpnClsLayers = [
    Binarizedconvolution2dLayer(1, numAnchors*2,'Name', 'rpnConv1x1ClsScores')
    batchNormalizationLayer('Name','bnrpnConv1x1ClsScores')
    rpnSoftmaxLayer('Name', 'rpnSoftmax')
    rpnClassificationLayer('Name','rpnClassification')
    ];
lgraph = addLayers(lgraph, rpnClsLayers);

% Connect the classification layers to the RPN network.
lgraph = connectLayers(lgraph, 'rpnSign', 'rpnConv1x1ClsScores');



% Add RPN regression layers.
rpnRegLayers = [
    Binarizedconvolution2dLayer(1, numAnchors*4, 'Name', 'rpnConv1x1BoxDeltas')
    batchNormalizationLayer('Name','bnrpnConv1x1BoxDeltas')
    rcnnBoxRegressionLayer('Name', 'rpnBoxDeltas');
    ];

lgraph = addLayers(lgraph, rpnRegLayers);

% Connect the regression layers to the RPN network.
lgraph = connectLayers(lgraph, 'rpnSign', 'rpnConv1x1BoxDeltas');

% Connect region proposal network.
lgraph = connectLayers(lgraph, 'rpnConv1x1ClsScores', 'regionProposal/scores');
lgraph = connectLayers(lgraph, 'rpnConv1x1BoxDeltas', 'regionProposal/boxDeltas');

% Connect region proposal layer to roi pooling.
lgraph = connectLayers(lgraph, 'regionProposal', 'roiPool/roi');

% Show the network after adding the RPN layers.
% figure
plot(lgraph)
% ylim([30 42])




data = load('fasterRCNNVehicleTrainingData.mat');

trainingData = data.vehicleTrainingData;

trainingData.imageFilename = fullfile(toolboxdir('vision'),'visiondata', ...
    trainingData.imageFilename);

 options = trainingOptions('sgdm', ...
      'MiniBatchSize', 1, ...
      'InitialLearnRate', 1e-3, ...
      'MaxEpochs', 5, ...
      'VerboseFrequency', 200, ...
      'CheckpointPath', tempdir);

detector = trainFasterRCNNObjectDetector(trainingData, lgraph, options)