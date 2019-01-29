function [trainedNet, info] = trainNetwork(varargin)
% trainNetwork   Train a neural network
%
%   trainedNet = trainNetwork(ds, layers, options) trains and returns a
%   network trainedNet for a classification problem. ds is an
%   imageDatastore with categorical labels or a MiniBatchable Datastore 
%   with responses, layers is an array of network layers or a LayerGraph 
%   and options is a set of training options.
%
%   trainedNet = trainNetwork(X, Y, layers, options) trains and returns a
%   network, trainedNet. The format for X depends on the input layer. For
%   an image input layer, X is a numeric array of images arranged so that
%   the first three dimensions are the width, height and channels, and the
%   last dimension indexes the individual images. In a classification
%   problem, Y specifies the labels for the images as a categorical vector.
%   In a regression problem, Y contains the responses arranged as a matrix
%   of size number of observations by number of responses, or a four
%   dimensional numeric array, where the last dimension corresponds to the
%   number of observations. 
%
%   trainedNet = trainNetwork(C, Y, layers, options) trains an LSTM network
%   for classifcation and regression problems for sequence or time-series
%   data. layers must define an LSTM network. It must begin with a sequence
%   input layer. C is a cell array containing sequence or time-series
%   predictors. The entries of C are D-by-S matrices where D is the number
%   of values per timestep, and S is the length of the sequence. For
%   sequence-to-label classification problems, Y is a categorical vector of
%   labels. For sequence-to-sequence classification problems, Y is a cell
%   array of categorical sequences. For sequence-to-one regression
%   problems, Y is a matrix of targets. For sequence-to-sequence regression
%   problems, Y is a cell array of numeric sequences. For
%   sequence-to-sequence problems, the number of time steps of the
%   sequences in Y must be identical to the corresponding predictor
%   sequences in C. For sequence-to-sequence problems with one observation,
%   C can be a matrix, and Y must be a categorical sequence of labels or a
%   matrix of responses.
%
%   trainedNet = trainNetwork(tbl, layers, options) trains and returns a
%   network, trainedNet. For networks with an image input layer, tbl is a
%   table containing predictors in the first column as either absolute or
%   relative image paths or images. Responses must be in the second column
%   as categorical labels for the images. In a regression problem,
%   responses must be in the second column as either vectors or cell arrays
%   containing 3-D arrays or in multiple columns as scalars. For networks
%   with a sequence input layer, tbl is a table containing absolute or
%   relative MAT file paths of predictors in the first column. For a
%   sequence-to-label classification problem, the second column must be a
%   categorical vector of labels. For a sequence-to-one regression problem,
%   the second column must be a numeric array of responses or in multiple
%   columns as scalars. For a sequence-to-sequence classification problem,
%   the second column must be an absolute or relative file path to a MAT
%   file with a categorical sequence. For a sequence-to-sequence regression
%   problem, the second column must be an absolute or relative file path to
%   a MAT file with a numeric response sequence.
%
%   trainedNet = trainNetwork(tbl, responseNames, ...) trains and returns a
%   network, trainedNet. responseNames is a character vector, a string
%   array, or a cell array of character vectors specifying the names of the
%   variables in tbl that contain the responses.
%
%   [trainedNet, info] = trainNetwork(...) trains and returns a network,
%   trainedNet. info contains information on training progress.
%
%   Example 1:
%       Train a convolutional neural network on some synthetic images
%       of handwritten digits. Then run the trained network on a test
%       set, and calculate the accuracy.
%
%       [XTrain, YTrain] = digitTrain4DArrayData;
%
%       layers = [ ...
%           imageInputLayer([28 28 1])
%           convolution2dLayer(5,20)
%           reluLayer
%           maxPooling2dLayer(2,'Stride',2)
%           fullyConnectedLayer(10)
%           softmaxLayer
%           classificationLayer];
%       options = trainingOptions('sgdm', 'Plots', 'training-progress');
%       net = trainNetwork(XTrain, YTrain, layers, options);
%
%       [XTest, YTest] = digitTest4DArrayData;
%
%       YPred = classify(net, XTest);
%       accuracy = sum(YTest == YPred)/numel(YTest)
%
%   Example 2:
%       Train a long short-term memory network to classify speakers of a
%       spoken vowel sounds on preprocessed speech data. Then make
%       predictions using a test set, and calculate the accuracy.
%
%       [XTrain, YTrain] = japaneseVowelsTrainData;
%
%       layers = [ ...
%           sequenceInputLayer(12)
%           lstmLayer(100, 'OutputMode', 'last')
%           fullyConnectedLayer(9)
%           softmaxLayer
%           classificationLayer];
%       options = trainingOptions('adam', 'Plots', 'training-progress');
%       net = trainNetwork(XTrain, YTrain, layers, options);
%
%       [XTest, YTest] = japaneseVowelsTestData;
%
%       YPred = classify(net, XTest);
%       accuracy = sum(YTest == YPred)/numel(YTest)
%
%   Example 3:
%       Train a network on synthetic digit data, and measure its
%       accuracy:
%
%       [XTrain, YTrain] = digitTrain4DArrayData;
%
%       layers = [
%           imageInputLayer([28 28 1], 'Name', 'input')
%           convolution2dLayer(5, 20, 'Name', 'conv_1')
%           reluLayer('Name', 'relu_1')
%           convolution2dLayer(3, 20, 'Padding', 1, 'Name', 'conv_2')
%           reluLayer('Name', 'relu_2')
%           convolution2dLayer(3, 20, 'Padding', 1, 'Name', 'conv_3')
%           reluLayer('Name', 'relu_3')
%           additionLayer(2,'Name', 'add')
%           fullyConnectedLayer(10, 'Name', 'fc')
%           softmaxLayer('Name', 'softmax')
%           classificationLayer('Name', 'classoutput')];
%
%       lgraph = layerGraph(layers);
%
%       lgraph = connectLayers(lgraph, 'relu_1', 'add/in2');
%
%       plot(lgraph);
%
%       options = trainingOptions('sgdm', 'Plots', 'training-progress');
%       [net,info] = trainNetwork(XTrain, YTrain, lgraph, options);
%
%       [XTest, YTest] = digitTest4DArrayData;
%       YPred = classify(net, XTest);
%       accuracy = sum(YTest == YPred)/numel(YTest)
%
%   See also nnet.cnn.layer, trainingOptions, SeriesNetwork, DAGNetwork, LayerGraph.

%   Copyright 2015-2018 The MathWorks, Inc.

narginchk(3,4);

try
    [layersOrGraph, opts, X, Y] = iParseInputArguments(varargin{:});
    [trainedNet, info] = doTrainNetwork(layersOrGraph, opts, X, Y);
catch e
    iThrowCNNException( e );
end

end

function [trainedNet, info] = doTrainNetwork(layersOrGraph, opts, X, Y)

haveDAGNetwork = iHaveDAGNetwork(layersOrGraph);

analyzedLayers = iInferParameters(layersOrGraph);
layers = analyzedLayers.ExternalLayers;
internalLayers = analyzedLayers.InternalLayers;

% Validate training data
iValidateTrainingDataForProblem( X, Y, layers );

% Set desired precision
precision = nnet.internal.cnn.util.Precision('single');

% Set up and validate parallel training
isRNN = nnet.internal.cnn.util.isRNN( internalLayers );
executionSettings = nnet.internal.cnn.assembler.setupExecutionEnvironment(...
    opts, isRNN, X, precision );

% Create a training dispatcher
trainingDispatcher = iCreateTrainingDataDispatcher(X, Y, opts, ...
    executionSettings, layers);

% Create a validation dispatcher if validation data was passed in
validationDispatcher = iValidationDispatcher( opts, executionSettings, ...
    layers );

% Assert that training and validation data are consistent
iAssertTrainingAndValidationDispatcherHaveSameClasses( ...
    trainingDispatcher, validationDispatcher );

% Assemble internal network
strategy = nnet.internal.cnn.assembler.NetworkAssemblerStrategyFactory...
	.createStrategy(~haveDAGNetwork);
assembler = nnet.internal.cnn.assembler.TrainingNetworkAssembler(strategy);    
trainedNet = assembler.assemble(analyzedLayers, executionSettings, ...
    trainingDispatcher, validationDispatcher);

% Instantiate reporters as needed
networkInfo = nnet.internal.cnn.util.ComputeNetworkInfo(trainedNet);
[reporters, trainingPlotReporter] = iOptionalReporters(opts, ...
    internalLayers, analyzedLayers, executionSettings, ...
    networkInfo, trainingDispatcher, validationDispatcher, assembler);
errorState = nnet.internal.cnn.util.ErrorState();
cleanup = onCleanup(@()iFinalizePlot(trainingPlotReporter, errorState));

% Always create the info recorder (because we will reference it later) but
% only add it to the list of reporters if actually needed.
infoRecorder = iInfoRecorder( opts, internalLayers );
if nargout >= 2
    reporters.add( infoRecorder );
end

% Create a trainer to train the network with dispatcher and options
trainer = iCreateTrainer( opts, executionSettings.precision, reporters,...
    executionSettings );

% Do pre-processing work required for normalizing data
trainedNet = trainer.initializeNetworkNormalizations(trainedNet, ...
    trainingDispatcher, executionSettings.precision, executionSettings, ...
    opts.Verbose);

% Do the training
trainedNet = trainer.train(trainedNet, trainingDispatcher);

% Do post-processing work (if any)
trainedNet = trainer.finalizeNetwork(trainedNet, trainingDispatcher);
iComputeFinalValidationResultsForPlot(trainingPlotReporter, ...
    validationDispatcher, trainedNet);
trainedNet = iPrepareNetworkForOutput(trainedNet, analyzedLayers, assembler);
info = infoRecorder.Info;

% Update error state ready for the cleanup.
errorState.ErrorOccurred = false;
end

function [layers, opts, X, Y] = iParseInputArguments(varargin)
% iParseInputArguments   Parse input arguments of trainNetwork
%
% Output arguments:
%   layers  - An array of layers or a layer graph
%   opts    - An object containing training options
%   X       - Input data, this can be a data dispatcher, an image
%             datastore, a table, a numeric array or a cell array
%   Y       - Response data, this can be a numeric array or empty in case X
%             is a dispatcher, a table, an image datastore or a cell array

X = varargin{1};
if iIsADataDispatcher( X )
    % X is a custom dispatcher. The custom dispatcher api is for internal
    % use only.
    Y = [];
    layers = varargin{2};
    opts = varargin{3};
elseif iIsAnImageDatastore( X )
    iAssertOnlyThreeArgumentsForIMDS( nargin );
    Y = [];
    layers = varargin{2};
    opts = varargin{3};
elseif iIsAMiniBatchableDatastore(X)
    Y = [];
    layers = varargin{2};
    opts = varargin{3};
elseif iIsPixelLabelDatastore( X )
    Y = [];
    layers = varargin{2};
    opts = varargin{3};
elseif istable( X )
    secondArgument = varargin{2};
    if ischar(secondArgument) || iscellstr(secondArgument) || isstring(secondArgument)
        % ResponseName syntax
        narginchk(4,4);
        responseNames = convertStringsToChars(secondArgument);
        iAssertValidResponseNames(responseNames, X);
        X = iSelectResponsesFromTable( X, responseNames );
        Y = [];
        layers = varargin{3};
        opts = varargin{4};
    else
        narginchk(3,3);
        Y = [];
        layers = varargin{2};
        opts = varargin{3};
    end
elseif isnumeric( X )
    narginchk(4,4);
    Y = varargin{2};
    layers = varargin{3};
    opts = varargin{4};
elseif iscell( X )
    narginchk(4,4);
    Y = varargin{2};
    layers = varargin{3};
    opts = varargin{4};
else
    error(message('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:XIsNotValidType'));
end

% Bypass network analyzer check of layers since we do not accept networks.
if ~isa(layers, 'nnet.cnn.LayerGraph') && ~isa(layers, 'nnet.cnn.layer.Layer')
    error(message('nnet_cnn:internal:cnn:analyzer:NetworkAnalyzer:InvalidInput'))
end
    
% Validate options
iValidateOptions( opts );
end

function [X, Y] = iGetValidationDataFromOptions( opts )
X = opts.ValidationData;
if iIsADataDispatcher( X )
    % X is a custom dispatcher. The custom dispatcher api is for internal
    % use only.
    Y = [];
elseif iIsAnImageDatastore( X )
    Y = [];
elseif iIsAMiniBatchableDatastore( X )
    Y = [];
elseif istable( X )
    Y = [];
elseif iscell( X )
    Y = X{2};
    X = X{1};
else
    % Do nothing. Invalid type is already checked when creating
    % trainingOptions
end
end

function iValidateOptions( opts )
% iValidateOptions   Assert that opts is a valid training option object
if ~isa(opts, 'nnet.cnn.TrainingOptions')
    error(message('nnet_cnn:trainNetwork:InvalidTrainingOptions'))
end
end

function iValidateTrainingDataForProblem( X, Y, layers )
% iValidateTrainingDataForProblem   Assert that the input training data X
% and response Y are valid for the class of problem considered
trainingDataValidator = iTrainingDataValidator;
trainingDataValidator.validateDataForProblem( X, Y, layers );
end

function iValidateValidationDataForProblem( X, Y, layers )
% iValidateValidationDataForProblem   Assert that the input validation data
% X and response Y are valid for the class of problem considered
validationDataValidator = iValidationDataValidator;
validationDataValidator.validateDataForProblem( X, Y, layers );
end

function trainingDataValidator = iTrainingDataValidator()
trainingDataValidator = nnet.internal.cnn.util.NetworkDataValidator( ...
    nnet.internal.cnn.util.TrainingDataErrorThrower );
end

function validationDataValidator = iValidationDataValidator()
validationDataValidator = nnet.internal.cnn.util.NetworkDataValidator( ...
    nnet.internal.cnn.util.ValidationDataErrorThrower );
end

function iAssertTrainingAndValidationDispatcherHaveSameClasses(trainingDispatcher, validationDispatcher)
if ~isempty(validationDispatcher)
    numResponses = numel(trainingDispatcher.ResponseMetaData);
    for i = 1:numResponses
        if iIsClassificationMetaData(trainingDispatcher.ResponseMetaData(i))
            iAssertClassNamesAreTheSame( ...
                trainingDispatcher.ResponseMetaData(i).Categories, ...
                validationDispatcher.ResponseMetaData(i).Categories);
            iAssertClassesHaveSameOrdinality( ...
                trainingDispatcher.ResponseMetaData(i).Categories, ...
                validationDispatcher.ResponseMetaData(i).Categories);
        end
    end
end
end

function tf = iIsClassificationMetaData(responseMetaData)
tf = isa(responseMetaData, 'nnet.internal.cnn.response.ClassificationMetaData');
end

function iAssertClassNamesAreTheSame(trainingCategories, validationCategories)
if ~iHaveSameClassNames(trainingCategories, validationCategories)
    error(message('nnet_cnn:trainNetwork:TrainingAndValidationDifferentClasses'));
end
end

function tf = iHaveSameClassNames(trainingCategories, validationCategories)
% iHaveSameClassNames   Return true if the class names for the training
% categories match the class names for the validation categories. This does
% not catch the situation in which one set is a subset of the other - that
% situation will be caught when we compare the number of classes in the
% datasets to the number of classes expected by the network
trainingClassNames = categories(trainingCategories);
validationClassNames = categories(validationCategories);
tf = all(ismember(trainingClassNames, validationClassNames));
end

function iAssertClassesHaveSameOrdinality(trainingCategories, validationCategories)
if ~iHaveSameOrdinality(trainingCategories, validationCategories)
    error(message('nnet_cnn:trainNetwork:TrainingAndValidationDifferentOrdinality'));
end
end

function tf = iHaveSameOrdinality(trainingCategories, validationCategories)
% iHaveSameOrdinality   Return true if the categories from the training
% dispatcher have the same ordinality as those for the validation
% dispatcher
tf = isequal(isordinal(trainingCategories), isordinal(validationCategories));
end

function iThrowCNNException( exception )
% Wrap exception in a CNNException, which reports the error in a custom way
err = nnet.internal.cnn.util.CNNException.hBuildCustomError( exception );
throwAsCaller(err);
end

function externalNet = iPrepareNetworkForOutput(internalNet, ...
    analyzedLayers, assembler)
% If output network is on pool, retrieve it
if isa(internalNet, 'Composite')
    spmd
        [internalNet, labWithOutput] = iPrepareNetworkForOutputOnPool(internalNet);
    end
    internalNet = internalNet{labWithOutput.Value};
else
    internalNet = iPrepareNetworkForHostPrediction(internalNet);
end

% Convert to external network for user
externalNet = assembler.createExternalNetwork(internalNet, analyzedLayers);
end

function [internalNet, labWithResult] = iPrepareNetworkForOutputOnPool(internalNet)
if isempty(internalNet)
    labWithResult = gop(@min, inf);
else
    labWithResult = gop(@min, labindex);
end
if labindex == labWithResult
    % Convert to host network on pool, in case client has no GPU
    internalNet = iPrepareNetworkForHostPrediction(internalNet);
end
% Only labWithResult can be returned using AutoTransfer - network is too
% big
labWithResult = distributedutil.AutoTransfer( labWithResult, labWithResult );
end

function internalNet = iPrepareNetworkForHostPrediction(internalNet)
internalNet = internalNet.prepareNetworkForPrediction();
internalNet = internalNet.setupNetworkForHostPrediction();
end

function externalNetwork = iPrepareAndCreateExternalNetwork(...
    internalNetwork, analyzedLayers, assembler)
% Prepare an internal network for prediction, then create an external
% network
internalNetwork = internalNetwork.prepareNetworkForPrediction();
internalNetwork = internalNetwork.setupNetworkForHostPrediction();
externalNetwork = assembler.createExternalNetwork(internalNetwork, ...
    analyzedLayers);
end

function iComputeFinalValidationResultsForPlot(trainingPlotReporter, dispatcher, trainedNet)
trainingPlotReporter.computeFinalValidationResults(dispatcher, trainedNet);
end

function infoRecorder = iInfoRecorder( opts, internalLayers )
trainingInfoContent = iTrainingInfoContent( opts, internalLayers );
infoRecorder = nnet.internal.cnn.util.traininginfo.Recorder(trainingInfoContent);
end

function aContent = iTrainingInfoContent( opts, internalLayers )
isValidationSpecified = iIsValidationSpecified(opts);

if iIsClassificationNetwork(internalLayers)
    if isValidationSpecified
        aContent = nnet.internal.cnn.util.traininginfo.ClassificationWithValidationContent;
    else
        aContent = nnet.internal.cnn.util.traininginfo.ClassificationContent;
    end
else
    if isValidationSpecified
        aContent = nnet.internal.cnn.util.traininginfo.RegressionWithValidationContent;
    else
        aContent = nnet.internal.cnn.util.traininginfo.RegressionContent;
    end
end
end

function tf = iIsClassificationNetwork(internalLayers)
tf = iIsClassificationLayer(internalLayers{end});
end

function tf = iIsClassificationLayer(internalLayer)
tf = isa(internalLayer, 'nnet.internal.cnn.layer.ClassificationLayer');
end

function iAssertOnlyThreeArgumentsForIMDS( nArgIn )
if nArgIn~=3
    error(message('nnet_cnn:trainNetwork:InvalidNarginWithImageDatastore'));
end
end

function tf = iIsADataDispatcher(X)
tf = isa(X, 'nnet.internal.cnn.DataDispatcher');
end

function tf = iIsAMiniBatchableDatastore(X)
tf = isa(X,'matlab.io.Datastore') && isa(X, 'matlab.io.datastore.MiniBatchable');
end

function dispatcher = iCreateTrainingDataDispatcher(X, Y, opts, executionSettings, layers)
% Create a dispatcher.
dispatcher = nnet.internal.cnn.DataDispatcherFactory.createDataDispatcher( ...
    X, Y, opts.MiniBatchSize, 'discardLast', executionSettings.precision,...
    executionSettings, opts.Shuffle, opts.SequenceLength, ...
    opts.SequencePaddingValue, layers);
end

function dispatcher = iCreateValidationDataDispatcher(X, Y, opts, precision, trainingExecutionSettings, layers)
% iCreateValidationDataDispatcher   Create a dispatcher for validation data

% Validation execution settings
executionSettings = iSetValidationExecutionSettings(trainingExecutionSettings);
dispatcher = nnet.internal.cnn.DataDispatcherFactory.createDataDispatcher( ...
    X, Y, opts.MiniBatchSize, 'truncateLast', precision, executionSettings, ...
    opts.Shuffle, opts.SequenceLength, opts.SequencePaddingValue, layers );
end

function executionSettings = iSetValidationExecutionSettings(trainingExecutionSettings)
% Copy training settings for use with validation
executionSettings = trainingExecutionSettings;
% If the training execution environment is parallel, prefetching cannot be
% used by the validation dispatcher
if trainingExecutionSettings.useParallel
    executionSettings.backgroundPrefetch = false;
end
% Validation dispatcher cannot be parallel
executionSettings.useParallel = false;
end

function [reporter, trainingPlotReporter] = iOptionalReporters(opts, ...
    internalLayers, analyzedLayers, executionSettings, ...
    networkInfo, trainingDispatcher, validationDispatcher, assembler)
% iOptionalReporters   Create a vector of Reporters based on the given
% training options and the network type
%
% See also: nnet.internal.cnn.util.VectorReporter
reporter = nnet.internal.cnn.util.VectorReporter();

isValidationSpecified = iIsValidationSpecified(opts);

isAClassificationNetwork = iIsClassificationNetwork(internalLayers);
if opts.Verbose
    % If verbose is true, add a progress displayer
    if isAClassificationNetwork
        if isValidationSpecified
            columnStrategy = nnet.internal.cnn.util.ClassificationValidationColumns;
        else
            columnStrategy = nnet.internal.cnn.util.ClassificationColumns;
        end
    else
        if isValidationSpecified
            columnStrategy = nnet.internal.cnn.util.RegressionValidationColumns;
        else
            columnStrategy = nnet.internal.cnn.util.RegressionColumns;
        end
    end
    progressDisplayerFrequency = opts.VerboseFrequency;
    if isValidationSpecified
        progressDisplayerFrequency = [progressDisplayerFrequency opts.ValidationFrequency];
    end
    progressDisplayer = nnet.internal.cnn.util.ProgressDisplayer(columnStrategy);
    progressDisplayer.Frequency = progressDisplayerFrequency;
    reporter.add( progressDisplayer );
end

if isValidationSpecified
    % Create a validation reporter
    validationPredictStrategy = iValidationPredictStrategy( validationDispatcher, internalLayers, executionSettings.precision, executionSettings, opts.Shuffle );
    validationReporter = iValidationReporter( validationDispatcher, executionSettings, opts.ValidationFrequency, opts.ValidationPatience, opts.Shuffle, validationPredictStrategy );
    reporter.add( validationReporter );
end

if ~isempty( opts.CheckpointPath )
    checkpointSaver = nnet.internal.cnn.util.CheckpointSaver( opts.CheckpointPath );
    checkpointSaver.ConvertorFcn = @(net)iPrepareAndCreateExternalNetwork(net,...
        analyzedLayers, assembler);
    reporter.add( checkpointSaver );
end

if ~isempty( opts.OutputFcn )
    userCallbackReporter = nnet.internal.cnn.util.UserCallbackReporter( opts.OutputFcn );
    reporter.add( userCallbackReporter );
end

if strcmp( opts.Plots, 'training-progress' )
    if isdeployed
        error(message('nnet_cnn:internal:cnn:ui:trainingplot:TrainingPlotNotDeployable'))
    end
    if ~isValidationSpecified
        validationReporter = nnet.internal.cnn.util.EmptyValidationReporter();   % To be used only by the trainingPlotReporter
    end
    trainingPlotReporter = iCreateTrainingPlotReporter(isAClassificationNetwork, executionSettings, opts, internalLayers, networkInfo, trainingDispatcher, isValidationSpecified, validationReporter);
    reporter.add( trainingPlotReporter );
else
    trainingPlotReporter = nnet.internal.cnn.util.EmptyPlotReporter();
end
end

function trainingPlotReporter = iCreateTrainingPlotReporter(isAClassificationNetwork, executionSettings, opts, internalLayers, networkInfo, trainingDispatcher, isValidationSpecified, validationReporter)
hasVariableNumItersPerEpoch = iHasVariableNumItersEachEpoch(opts, internalLayers);
if hasVariableNumItersPerEpoch
    epochDisplayer = nnet.internal.cnn.ui.axes.EpochDisplayHider();
    determinateProgress = nnet.internal.cnn.ui.progress.DeterminateProgressText();
    tableDataFactory = nnet.internal.cnn.ui.info.VariableEpochSizeTextTableDataFactory();
else
    epochDisplayer = nnet.internal.cnn.ui.axes.EpochAxesDisplayer();
    determinateProgress = nnet.internal.cnn.ui.progress.DeterminateProgressBar();
    tableDataFactory = nnet.internal.cnn.ui.info.TextTableDataFactory();
end

% create the view
legendLayout = nnet.internal.cnn.ui.layout.Legend();
textLayout = nnet.internal.cnn.ui.layout.TextTable();
trainingPlotView = nnet.internal.cnn.ui.TrainingPlotViewHG(determinateProgress, legendLayout, textLayout);

% create the presenter
if isAClassificationNetwork
    axesFactory = nnet.internal.cnn.ui.factory.ClassificationAxesFactory();
    metricRowDataFactory = nnet.internal.cnn.ui.info.ClassificationMetricRowDataFactory();
else
    axesFactory = nnet.internal.cnn.ui.factory.RegressionAxesFactory();
    metricRowDataFactory = nnet.internal.cnn.ui.info.RegressionMetricRowDataFactory();
end
executionInfo = nnet.internal.cnn.ui.ExecutionInfo(executionSettings.executionEnvironment, executionSettings.useParallel, opts.LearnRateScheduleSettings.Method, opts.InitialLearnRate);
validationInfo = nnet.internal.cnn.ui.ValidationInfo(isValidationSpecified, opts.ValidationFrequency, opts.ValidationPatience);
%
watch = nnet.internal.cnn.ui.adapter.Stopwatch();
stopReasonRowDataFactory = nnet.internal.cnn.ui.info.StopReasonRowDataFactory();
preprocessingDisplayer = iCreatePreprocessingDisplayer(networkInfo);
helpLauncher = nnet.internal.cnn.ui.info.TrainingPlotHelpLauncher();
epochInfo = iCreateEpochInfo(opts, trainingDispatcher);
dialogFactory = nnet.internal.cnn.ui.DialogFactory();
trainingPlotPresenter = nnet.internal.cnn.ui.TrainingPlotPresenterWithDialog(...
    trainingPlotView, tableDataFactory, metricRowDataFactory, stopReasonRowDataFactory, preprocessingDisplayer, dialogFactory, ...
    axesFactory, epochDisplayer, helpLauncher, watch, executionInfo, validationInfo, epochInfo);

% create the reporter
summaryFactory = nnet.internal.cnn.util.SummaryFactory();
trainingPlotReporter = nnet.internal.cnn.util.TrainingPlotReporter(trainingPlotPresenter, validationReporter, summaryFactory, epochInfo);
end

function iFinalizePlot(trainingPlotReporter, errorState)
trainingPlotReporter.finalizePlot(errorState.ErrorOccurred);
end

function epochInfo = iCreateEpochInfo(opts, trainingDispatcher)
epochInfo = nnet.internal.cnn.ui.EpochInfo(opts.MaxEpochs, trainingDispatcher.NumObservations, opts.MiniBatchSize);
end

function preprocessingDisplayer = iCreatePreprocessingDisplayer(networkInfo)
if networkInfo.ShouldImageNormalizationBeComputed
    dialogFactory = nnet.internal.cnn.ui.DialogFactory();
    preprocessingDisplayer = nnet.internal.cnn.ui.PreprocessingDisplayerDialog(dialogFactory);
else
    preprocessingDisplayer = nnet.internal.cnn.ui.PreprocessingDisplayerEmpty();
end
end

function tf = iHasVariableNumItersEachEpoch(opts, internalLayers)
isRNN = isa(internalLayers{1}, 'nnet.internal.cnn.layer.SequenceInput');
hasCustomSequenceLength = isnumeric(opts.SequenceLength);
tf = (isRNN && hasCustomSequenceLength);
end

function validationDispatcher = iValidationDispatcher(opts, ...
    executionSettings, layers)
% iValidationDispatcher   Get validation data and create a dispatcher for it. Validate the
% data for the current problem and w.r.t. the current architecture.

% Return empty if no validation data was specified
if ~iIsValidationSpecified(opts)
    validationDispatcher = [];
else
    % There is no need to convert datastore into table, since validation
    % will be computed only on one worker
    [XVal, YVal] = iGetValidationDataFromOptions( opts );
    iValidateValidationDataForProblem( XVal, YVal, layers );
    % Create a validation dispatcher
    validationDispatcher = iCreateValidationDataDispatcher(XVal, YVal, ...
        opts, executionSettings.precision, executionSettings, layers);
end
end

function tf = iIsValidationSpecified(opts)
tf = ~isempty(opts.ValidationData);
end

function strategy = iValidationPredictStrategy( validationDispatcher, layers, precision, executionSettings, shuffle )
strategy = nnet.internal.cnn.util.ValidationPredictStrategyFactory.createStrategy(validationDispatcher, layers, precision, executionSettings, shuffle);
end

function validator = iValidationReporter(validationDispatcher, executionEnvironment, frequency, patience, shuffle, validationPredictStrategy)
validator = nnet.internal.cnn.util.ValidationReporter(validationDispatcher, executionEnvironment, frequency, patience, shuffle, validationPredictStrategy);
end

function trainer = iCreateTrainer( ...
    opts, precision, reporters, executionSettings )
if ~executionSettings.useParallel
    summaryFcn = @(dispatcher,maxEpochs)nnet.internal.cnn.util. ...
        MiniBatchSummary(dispatcher,maxEpochs);
    trainer = nnet.internal.cnn.Trainer( ...
        opts, precision, reporters, executionSettings, summaryFcn);
else
    summaryFcn = @(dispatcher,maxEpochs)nnet.internal.cnn.util. ...
        ParallelMiniBatchSummary(dispatcher,maxEpochs);
    trainer = nnet.internal.cnn.ParallelTrainer( ...
        opts, precision, reporters, executionSettings, summaryFcn);
end
end

function tf = iIsAnImageDatastore(x)
tf = isa(x, 'matlab.io.datastore.ImageDatastore');
end

function iAssertValidResponseNames(responseNames, tbl)
% iAssertValidResponseNames   Assert that the response names are variables
% of the table and they do not refer to the first column.
variableNames = tbl.Properties.VariableNames;
refersToFirstColumn = ismember( variableNames(1), responseNames );
responseNamesAreAllVariables = all( ismember(responseNames,variableNames) );
if refersToFirstColumn || ~responseNamesAreAllVariables
    error(message('nnet_cnn:trainNetwork:InvalidResponseNames'))
end
end

function resTbl = iSelectResponsesFromTable(tbl, responseNames)
% iSelectResponsesFromTable   Return a new table with only the first column
% (predictors) and the variables specified in responseNames.
variableNames = tbl.Properties.VariableNames;
varTF = ismember(variableNames, responseNames);
% Make sure to select predictors (first column) as well
varTF(1) = 1;
resTbl = tbl(:,varTF);
end

function tf = iIsPixelLabelDatastore(x)
tf = isa(x, 'matlab.io.datastore.PixelLabelDatastore');
end

function haveDAGNetwork = iHaveDAGNetwork(lgraph)
haveDAGNetwork = isa(lgraph,'nnet.cnn.LayerGraph');
end

function analysis = iInferParameters(layersOrGraph)
[~, analysis] = nnet.internal.cnn.layer.util.inferParameters(layersOrGraph);
end
