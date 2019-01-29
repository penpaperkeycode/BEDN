%segnetLayers Create SegNet for semantic segmentation using deep learning.
%
%   SegNet is a convolutional neural network for semantic image
%   segmentation. It uses a pixelClassificationLayer to predict the
%   categorical label for every pixel in an input image.
%
%   Use segnetLayers to create the network architecture for SegNet. This
%   network must be trained using trainNetwork from Deep Learning Toolbox
%   before it can be used for semantic segmentation.
%
%   lgraph = segnetLayers(imageSize, numClasses, model) returns SegNet
%   network layers pre-initialized with layers and weights from a
%   pre-trained network model. Valid values for model are 'vgg16' or
%   'vgg19'. The imageSize and numClasses inputs are described below. The
%   output lgraph is a LayerGraph object representing the SegNet network
%   architecture.
%
%   [...] = segnetLayers(imageSize, numClasses, encoderDepth) returns
%   uninitialized SegNet layers configured using the following inputs:
%
%   Inputs
%   ------
%   imageSize    - size of the network input image specified as a vector
%                  [H W] or [H W C], where H and W are the image height and
%                  width, and C is the number of image channels.
%
%   numClasses   - number of classes the network should be configured to
%                  classify.
%
%   encoderDepth - SegNet is composed of an encoder sub-network and a
%                  corresponding decoder sub-network. Specify the depth of
%                  these networks as a scalar D. The depth of these
%                  networks determines the number of times an input image
%                  is downsampled or upsampled as it is processed. The
%                  encoder network downsamples the input image by a factor
%                  of 2^D. The decoder network performs the opposite
%                  operation and upsamples the encoder network output by a
%                  factor of 2^D.
%
%                  <a href="matlab:doc('segnetLayers')">Learn more about SegNet.</a>
%
%   [...] = segnetLayers(imageSize, numClasses, encoderDepth, Name, Value)
%   specifies additional name-value pair arguments described below:
%
%   'NumConvolutionLayers'   The encoder and decoder networks are divided
%                            into D sections of N convolutional layers,
%                            where D is the depth of the encoder/decoder
%                            network. Specify the number of convolutional
%                            layers, N, as a scalar or a vector. When N is
%                            a vector, N(k) is the number of convolution
%                            layers in the k-th encoder/decoder section.
%                            Typical values are between 1 and 3.
%
%                            Default: 2
%
%   'NumOutputChannels'      Specify the number of output channels for each
%                            of the D sections in the SegNet encoder
%                            network as a scalar or vector M. When M is a
%                            vector, M(k) is the number of output channels
%                            of the k-th section of the encoder network.
%                            When M is a scalar, the same number of output
%                            channels is used for all encoder sections. The
%                            number of output channels in the decoder
%                            sections is automatically set to match the
%                            corresponding encoder section.
%
%                            Default: 64
%
%   'FilterSize'             Specify the height and width used for all
%                            convolutional layer filters as a scalar or
%                            vector [H W]. When the size is a scalar, the
%                            same value is used for all layers. Typical
%                            values are between 3 and 7.
%
%                            Default: 3
%
% Notes
% -----
% - The sections within the SegNet encoder and decoder sub-networks are
%   made up of convolutional, batch normalization, and ReLU layers.
%
% - All convolutional layers are configured such that the bias term is
%   fixed to zero.
%
% - The depth of vgg16 and vgg19 networks is 5.
%
% - Convolution layer weights in the encoder and decoder sub-networks are
%   initialized using the 'MSRA' weight initialization method. When 'vgg16'
%   or 'vgg19' is specified as the model, only the decoder sub-network is
%   initialized using MSRA.
%
% Example 1 - Create SegNet initialized using VGG-16 weights.
% -----------------------------------------------------------
% imageSize = [480 640 3];
% numClasses = 5;
% lgraph = segnetLayers(imageSize, numClasses, 'vgg16')
%
% % Display network.
% figure
% plot(lgraph)
%
% Example 2 - Create SegNet with custom encoder/decoder depth.
% ------------------------------------------------------------
% % Create SegNet layers with an encoder/decoder depth of 4. 
% imageSize = [480 640 3];
% numClasses = 5;
% encoderDepth = 4;
% lgraph = segnetLayers(imageSize, numClasses, encoderDepth)
% 
% % Display network.
% figure
% plot(lgraph)
%
% Example 3 - Train SegNet.
% -------------------------
% % Load training images and pixel labels.
% dataSetDir = fullfile(toolboxdir('vision'),'visiondata','triangleImages');
% imageDir = fullfile(dataSetDir, 'trainingImages');
% labelDir = fullfile(dataSetDir, 'trainingLabels');
%
% % Create an imageDatastore holding the training images.
% imds = imageDatastore(imageDir);
%
% % Define the class names and their associated label IDs.
% classNames = ["triangle", "background"];
% labelIDs   = [255 0];
%
% % Create a pixelLabelDatastore holding the ground truth pixel labels for
% % the training images.
% pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);
%  
% % Create SegNet.
% imageSize = [32 32];
% numClasses = 2;
% lgraph = segnetLayers(imageSize, numClasses, 2)
% 
% % Create data source for training a semantic segmentation network.
% datasource = pixelLabelImageSource(imds,pxds);
% 
% % Setup training options. 
% options = trainingOptions('sgdm', 'InitialLearnRate', 1e-3, ...
%     'MaxEpochs', 20, 'VerboseFrequency', 10);
% 
% % Train network.
% net = trainNetwork(datasource, lgraph, options)
%
% See also fcnLayers, vgg16, vgg19, unetLayers, pixelClassificationLayer,
%          LayerGraph, trainNetwork, DAGNetwork, semanticseg,
%          pixelLabelImageDatastore.

% References 
% ----------
% [1] Badrinarayanan, Vijay, Alex Kendall, and Roberto Cipolla. "Segnet: A
% deep convolutional encoder-decoder architecture for image segmentation."
% arXiv preprint arXiv:1511.00561 (2015).
%
% [2] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
% human-level performance on imagenet classification." Proceedings of the
% IEEE international conference on computer vision. 2015.

% Copyright 2017-2018 The MathWorks, Inc.

function lgraph = BinarizedsegnetLayersMaker(net,imageSize, numClasses, depth, varargin)

vision.internal.requiresNeuralToolbox(mfilename);

args = iParseInputs(imageSize, numClasses, depth, varargin{:});

if ischar(args.depth)
            
    iCheckImageSizeHasThreeElementsForVGG(args.imageSize);
    iCheckImageSizeBigEnoughForDepth(args.imageSize, 5);
    
    switch args.depth
        case 'vgg16'
            
            iCheckIfVGG16AddOnIsAvailable();
            
            net = vgg16();
            lastMaxPoolingLayer = 32;
            numChannelsLastMaxPoolingLayer = 512;
            
            args.NumConvolutionLayers = [2 2 3 3 3];
            args.NumOutputChannels = [64 128 256 512 512];
            args.FilterSize = [3 3];
        case 'vgg19'
            
            iCheckIfVGG19AddOnIsAvailable()
            
            net = vgg19();
            lastMaxPoolingLayer = 38;
            numChannelsLastMaxPoolingLayer = 512;
            
            args.NumConvolutionLayers = [2 2 4 4 4];
            args.NumOutputChannels = [64 128 256 512 512];
            args.FilterSize = [3 3];
    end      
    % both vgg16 and 19 have depth of 5.
    args.depth = 5;
    
    layers = net.Layers;
    
    layers(1) = imageInputLayer(args.imageSize, 'Name', 'inputImage');
    
    layers(lastMaxPoolingLayer+1:end) = [];
    
    layers = iInsertBatchNormalization(layers);
    
    layers = iDisableConvBias(layers);
    
    layers = iAddDecoderLayers(layers, numChannelsLastMaxPoolingLayer, args);

    layers = [
        layers'
        iFinalLayers()
        ];
    
    maxPoolLayerIndices = iFindMaxPoolingLayer(layers);
    maxPoolLayerID = iGetMaxPoolingLayerNames(layers);
    unpoolLayerID  = flip(iGetUnpoolingLayerNames(layers));
    assert(numel(maxPoolLayerID) == numel(unpoolLayerID));
    
    % Change max pooling layers to have HasUnpoolingOutputs true
    for i = 1:numel(maxPoolLayerID)
        currentMaxPooling = layers(maxPoolLayerIndices(i));
        poolSize = currentMaxPooling.PoolSize;
        stride = currentMaxPooling.Stride;
        name = currentMaxPooling.Name;
        padding = currentMaxPooling.PaddingSize;
        layers(maxPoolLayerIndices(i)) = maxPooling2dLayer(poolSize, 'Stride', stride, 'Name', name, 'Padding', padding, 'HasUnpoolingOutputs', true);
    end
    
    % Create layer graph
    lgraph = layerGraph(layers);
    
    % Connect all max pool outputs to unpooling layers
    for i = 1:numel(maxPoolLayerID)
        lgraph = connectLayers(lgraph, [maxPoolLayerID{i} '/indices'], [unpoolLayerID{i} '/indices']);
        lgraph = connectLayers(lgraph, [maxPoolLayerID{i} '/size'], [unpoolLayerID{i} '/size']);
    end
    
    return
end

filterSize = args.FilterSize;
 
iCheckImageSizeBigEnoughForDepth(args.imageSize, args.depth);
 
layers = imageInputLayer(args.imageSize, 'Name', 'inputImage');
layerIdx = 1;

if numel(args.imageSize) == 3
    numChannels = args.imageSize(3);
else
    numChannels = 1;
end

for i = 1:args.depth
    
    numFilters = args.NumOutputChannels(i);
    
    % Stack encoders
    for j = 1:args.NumConvolutionLayers(i)
        layers(end+1) = iCreateConvLayer(filterSize, numFilters, numChannels, iConvEncoderName(i,j));
        
        % next numChannels is previous layer's numFilters.
        numChannels = numFilters;
               
        layers(end+1) = batchNormalizationLayer('Name', iBNLayerName('encoder', i,j));
        layers(end+1) = reluLayer('Name', iReluLayerName('encoder', i,j));
         
        layerIdx = layerIdx + 3;
    end
    
    maxPoolName = iMaxPoolName(i);
    layers(end+1) = maxPooling2dLayer([2 2], 'Stride', 2, 'Name', maxPoolName, 'HasUnpoolingOutputs', true);
    
    layerIdx = layerIdx + 1;
    
    % ID of max pooling layer. to be used to connect.
    maxPoolID{i} = maxPoolName;
    
end

decoderLastConvNumFilters = [numClasses (args.NumOutputChannels(1:end-1))];

for i = args.depth:-1:1
    
    % choose one before current decoder.
    numFilters = args.NumOutputChannels(i);
    layerIdx = layerIdx + 1;
    layers(end+1) = maxUnpooling2dLayer('Name', iMaxUnpoolName(i)); %#ok<*AGROW>
    
    maxUnpoolID{i} = iMaxUnpoolName(i);
    
    % Stack decoders, build backwards
    for j = args.NumConvolutionLayers(i):-1:2
        layers(end+1) = iCreateConvLayer(filterSize, numFilters, numChannels, iConvDecoderName(i,j));
        
        % next numChannels is previous layer's numFilters.
        numChannels = numFilters;
        
        layers(end+1) = batchNormalizationLayer('Name', iBNLayerName('decoder', i,j));
        layers(end+1) = reluLayer('Name', iReluLayerName('decoder', i,j));
        
        layerIdx = layerIdx + 3;
    end
    
    % last decoder conv layer uses different set of num output filters
    layers(end+1) = iCreateConvLayer(filterSize, decoderLastConvNumFilters(i),  numChannels, iConvDecoderName(i,j-1));
    
    numChannels = decoderLastConvNumFilters(i);
          
    layers(end+1) = batchNormalizationLayer('Name', iBNLayerName('decoder', i,j-1));
    layers(end+1) = reluLayer('Name', iReluLayerName('decoder', i,j-1));
    
    layerIdx = layerIdx + 3;
end


layers = [
    layers'
    iFinalLayers()
    ];

% Create layer graph
lgraph = layerGraph(layers);

% Connect all max pool outputs to unpooling layers
for i = 1:numel(maxPoolID)
    lgraph = connectLayers(lgraph, [maxPoolID{i} '/indices'], [maxUnpoolID{i} '/indices']);
    lgraph = connectLayers(lgraph, [maxPoolID{i} '/size'], [maxUnpoolID{i} '/size']);
end

%--------------------------------------------------------------------------
function finalLayers = iFinalLayers()

finalLayers = [
    softmaxLayer('Name', 'softmax')
    pixelClassificationLayer('Name', 'pixelLabels')
    ];

function iCheckImageSizeBigEnoughForDepth(imageSize, depth)
% the encoder sub-network downsamples image size by 2^D. Ensure the image
% size doesn't get reduce to less than 1.

if any( imageSize(1:2) ./ (2^depth)  < 1 )
    sz = imageSize;
    sz(1:2) = repelem(2^depth,1,2);
    error(message('vision:semanticseg:imageTooSmallForDepth', mat2str(sz)));
end

%--------------------------------------------------------------------------
function name = iConvDecoderName(encoderIdx, idx)
name = sprintf('decoder%d_conv%d', encoderIdx, idx);

%--------------------------------------------------------------------------
function name = iConvEncoderName(encoderIdx, idx)
name = sprintf('encoder%d_conv%d', encoderIdx, idx);

%--------------------------------------------------------------------------
function name = iBNLayerName(prefix, encoderIdx, idx)
name = sprintf('%s%d_bn_%d',prefix, encoderIdx, idx);

%--------------------------------------------------------------------------
function name = iReluLayerName(prefix, encoderIdx, idx)
name = sprintf('%s%d_relu_%d',prefix, encoderIdx, idx);

%--------------------------------------------------------------------------
function name = iMaxPoolName(encoderIdx)
name = sprintf('encoder%d_maxpool',encoderIdx);

%--------------------------------------------------------------------------
function name = iMaxUnpoolName(encoderIdx)
name = sprintf('decoder%d_unpool',encoderIdx);

%--------------------------------------------------------------------------
function args = iParseInputs(varargin)

p = inputParser;
p.addRequired('imageSize', @iCheckImageSize);
p.addRequired('numClasses', @iCheckNumClasses);
p.addRequired('depth');
p.addParameter('FilterSize', [3 3], @iCheckFilterSize);
p.addParameter('NumConvolutionLayers', 2, @iCheckNumConvLayers);
p.addParameter('NumOutputChannels', 64, @iCheckNumOutputChannels);

p.parse(varargin{:});

userInput = p.Results;
depthOrName = iCheckDepthOrName(userInput.depth);

args.depth      = depthOrName;
args.imageSize  = double(userInput.imageSize);
args.numClasses = double(userInput.numClasses);

usingPretrainedNetwork = ischar(args.depth);

if usingPretrainedNetwork
    wasSpecified = @(x)~ismember(x,p.UsingDefaults);
    
    if wasSpecified('FilterSize') || ...
            wasSpecified('NumConvolutionLayers') ||...
            wasSpecified('NumOutputChannels')
        
        error(message('vision:semanticseg:notSupportedWithPretrained'));
    end
else
    args.FilterSize = double(userInput.FilterSize);
    
    args.NumConvolutionLayers = iCheckVectorOrExpandScalar(...
        double(userInput.NumConvolutionLayers), ...
        args.depth, ...
        'NumConvolutionLayers');
    
    args.NumOutputChannels = iCheckVectorOrExpandScalar(...
        double(userInput.NumOutputChannels), ...
        args.depth, ...
        'NumOutputChannels');
end

%--------------------------------------------------------------------------
function layer = iCreateConvLayer(filterSize, numFilters, numChannels, name)

layer = convolution2dLayer(filterSize, numFilters, 'NumChannels', numChannels, ...
    'Padding', iSamePadding(filterSize), 'Name', name);

% Effectively disable bias by setting learning rates to 0 and initializing
% the bias to zero.
layer.BiasLearnRateFactor = 0;
layer.BiasL2Factor = 0;
layer.Bias = zeros(1,1,numFilters, 'single');

% Initialize weights using MSRA weight initialization:
%   He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
%   human-level performance on imagenet classification." Proceedings of the
%   IEEE international conference on computer vision. 2015.
shape = [layer.FilterSize layer.NumChannels layer.NumFilters];
n = prod([layer.FilterSize layer.NumChannels]);
layer.Weights = sqrt(2/n) * randn(shape, 'single');

%--------------------------------------------------------------------------
function out = iCheckVectorOrExpandScalar(value, N, name)
if isscalar(value)
    out = repelem(value, 1, N);
else   
    validateattributes(value,{'numeric'},{'numel', N}, mfilename, name);
    out = value;
end

%--------------------------------------------------------------------------
function iCheckImageSizeHasThreeElementsForVGG(imageSize)
% VGG only support RGB images. Check that image size has 3 elements.
if ~(numel(imageSize) == 3 && imageSize(3) == 3)
    error(message('vision:semanticseg:vggNeedsRGB'));
end

%--------------------------------------------------------------------------
function p = iSamePadding(FilterSize)
p = floor(FilterSize / 2);

%--------------------------------------------------------------------------
function x = iCheckDepthOrName(x)

if isnumeric(x)
    validateattributes(x,{'numeric'}, ...
        {'scalar', 'real', 'finite', 'integer', 'nonsparse', 'positive'}, ...
        mfilename, 'depth');
else
    validateattributes(x,{'char','string'},{'scalartext'}, mfilename, 'name')
    x = validatestring(x,{'vgg16','vgg19'},mfilename,'name');
end
    
%--------------------------------------------------------------------------
function iCheckImageSize(x)
validateattributes(x, {'numeric'}, ...
    {'vector', 'real', 'finite', 'integer', 'nonsparse', 'positive'}, ...
    mfilename, 'imageSize');
    
N = numel(x);
if ~(N == 2 || N == 3)
    error(message('vision:semanticseg:imageSizeIncorrect'));
end
    
%--------------------------------------------------------------------------
function iCheckFilterSize(x)
% require odd filter sizes to facilitate "same" output size padding. In the
% future this can be relaxed with asymmetric padding.
if isscalar(x)
    validateattributes(x, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'integer', 'nonsparse', 'positive', 'odd'}, ...
        mfilename, 'FilterSize');
else
    validateattributes(x, {'numeric'}, ...
        {'vector', 'real', 'finite', 'integer', 'nonsparse', 'positive', 'odd'}, ...
        mfilename, 'FilterSize');
end

%--------------------------------------------------------------------------
function iCheckNumClasses(x)
validateattributes(x, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', 'nonsparse', '>', 1}, ...
    mfilename, 'numClasses');

%--------------------------------------------------------------------------
function iCheckNumConvLayers(x)
if isscalar(x)
    validateattributes(x, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'integer', 'nonsparse', 'positive'}, ...
        mfilename, 'NumConvolutionLayers');
else
    validateattributes(x, {'numeric'}, ...
        {'vector', 'real', 'finite', 'integer', 'nonsparse', 'positive'}, ...
        mfilename, 'NumConvolutionLayers');
end

%--------------------------------------------------------------------------
function iCheckNumOutputChannels(x)
if isscalar(x)
    validateattributes(x, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'integer', 'nonsparse', 'positive'}, ...
        mfilename, 'NumOutputChannels');
else
    validateattributes(x, {'numeric'}, ...
        {'vector', 'real', 'finite', 'integer', 'nonsparse', 'positive'}, ...
        mfilename, 'NumOutputChannels');
end

%--------------------------------------------------------------------------
function layers = iAddDecoderLayers(layers, numChannels, args)
decoderLastConvNumFilters = [args.numClasses (args.NumOutputChannels(1:end-1))];
filterSize = args.FilterSize;
for i = args.depth:-1:1
    
    % choose one before current decoder.
    numFilters = args.NumOutputChannels(i);
    
    layers(end+1) = maxUnpooling2dLayer('Name', iMaxUnpoolName(i)); %#ok<*AGROW>
    
    % Stack decoders, build backwards
    for j = args.NumConvolutionLayers(i):-1:2
        layers(end+1) = iCreateConvLayer(filterSize, numFilters, numChannels, iConvDecoderName(i,j));
        
        % next numChannels is previous layer's numFilters.
        numChannels = numFilters; 
        
        layers(end+1) = batchNormalizationLayer('Name', iBNLayerName('decoder', i,j));        
        layers(end+1) = reluLayer('Name', iReluLayerName('decoder', i,j));
        
    end
    
    % last decoder conv layer uses different set of num output filters
    layers(end+1) = iCreateConvLayer(filterSize, decoderLastConvNumFilters(i), numChannels, iConvDecoderName(i,j-1));
     
    numChannels = decoderLastConvNumFilters(i);
    
    layers(end+1) = batchNormalizationLayer('Name', iBNLayerName('decoder', i,j-1));    
    layers(end+1) = reluLayer('Name', iReluLayerName('decoder', i,j-1));
    
    
end

%--------------------------------------------------------------------------
function idx = iFindLayer(layers, type)
results = arrayfun(@(x)isa(x,type),layers,'UniformOutput', true);
idx = find(results);

%--------------------------------------------------------------------------
function idx = iFindMaxPoolingLayer(layers) 
idx = iFindLayer(layers, 'nnet.cnn.layer.MaxPooling2DLayer');

%--------------------------------------------------------------------------
function layers = iDisableConvBias(layers)
idx = iFindLayer(layers, 'nnet.cnn.layer.Convolution2DLayer');
for i = 1:numel(idx)
    layers(idx(i)).BiasLearnRateFactor = 0;
    layers(idx(i)).BiasL2Factor = 0;
    layers(idx(i)).Bias = zeros(1,1,layers(idx(i)).NumFilters);
end

%--------------------------------------------------------------------------
function layerNames = iGetLayerNames(layers, type)
isOfType = arrayfun(@(x)isa(x,type),layers,'UniformOutput', true);
layerNames = {layers(isOfType).Name}';

%--------------------------------------------------------------------------
function layerNames = iGetUnpoolingLayerNames(layers)
layerNames = iGetLayerNames(layers, 'nnet.cnn.layer.MaxUnpooling2DLayer');

%--------------------------------------------------------------------------
function layerNames = iGetMaxPoolingLayerNames(layers)
layerNames = iGetLayerNames(layers, 'nnet.cnn.layer.MaxPooling2DLayer');

%--------------------------------------------------------------------------
function newLayers = iInsertBatchNormalization(layers)
newLayers(1) = layers(1);
for i = 2:numel(layers)
    newLayers(end+1) = layers(i);
    if isa(layers(i),'nnet.cnn.layer.Convolution2DLayer')
        % Place batch norm after conv layer.
        newLayers(end+1) = batchNormalizationLayer('Name', sprintf('bn_%s',layers(i).Name));
    end
end

%--------------------------------------------------------------------------
function iCheckIfVGG16AddOnIsAvailable()
breadcrumbFile = 'nnet.internal.cnn.supportpackages.IsVGG16Installed';
fullpath = which(breadcrumbFile);

if isempty(fullpath)
    name = 'Deep Learning Toolbox Model for VGG-16 Network';
    error(message('vision:semanticseg:missingVGGAddon',name));
end

%--------------------------------------------------------------------------
function iCheckIfVGG19AddOnIsAvailable()
breadcrumbFile = 'nnet.internal.cnn.supportpackages.IsVGG16Installed';
fullpath = which(breadcrumbFile);
if isempty(fullpath)
    name = 'Deep Learning Toolbox Model for VGG-19 Network';
    error(message('vision:semanticseg:missingVGGAddon',name));
end
