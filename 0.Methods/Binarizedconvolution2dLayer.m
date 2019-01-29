function layer = Binarizedconvolution2dLayer( varargin )
% Binarziedconvolution2dLayer   2D convolution layer for Convolutional Neural Networks
%
%   layer = Binarziedconvolution2dLayer(filterSize, numFilters) creates a layer
%   for 2D convolution. filterSize specifies the height and width of the
%   filters. It can be a scalar, in which case the filters will have the
%   same height and width, or a vector [h w] where h specifies the height
%   for the filters, and w specifies the width. numFilters specifies the
%   number of filters.
% 
%   layer = Binarziedconvolution2dLayer(filterSize, numFilters, 'PARAM1', VAL1, 'PARAM2', VAL2, ...) 
%   specifies optional parameter name/value pairs for creating the layer:
%
%       'Stride'                  - The step size for traversing the input
%                                   vertically and horizontally. This can
%                                   be a scalar, in which case the same
%                                   value is used for both dimensions, or
%                                   it can be a vector [u v] where u is the
%                                   vertical stride, and v is the
%                                   horizontal stride. The default is 
%                                   [1 1].
%       'DilationFactor'          - The up-sampling factor of the filter.
%                                   It corresponds to an effective filter
%                                   size of filterSize + (filterSize-1) .*
%                                   (dilationFactor-1). This can be a
%                                   scalar, in which case the same value is
%                                   used for both dimensions, or it can be
%                                   a vector [dHeight dWidth] where dHeight
%                                   is the vertical dilation, and dWidth is
%                                   the horizontal dilation. The default is
%                                   [1 1].
%       'Padding'                 - The padding applied to the input
%                                   along the edges. This can be:
%                                     - the character array 'same'. Padding
%                                       is set so that the output size 
%                                       is the same as the input size 
%                                       when the stride is 1. More 
%                                       generally, the output size is 
%                                       ceil(inputSize/stride), where 
%                                       inputSize is the height and width 
%                                       of the input.
%                                     - a scalar, in which case the same
%                                       padding is applied vertically and
%                                       horizontally.
%                                     - a vector [a b] where a is the 
%                                       padding applied to the top and 
%                                       bottom of the input, and b is the
%                                       padding applied to the left and 
%                                       right.
%                                     - a vector [t b l r] where t is the
%                                       padding applied to the top, b is
%                                       the padding applied to the bottom,
%                                       l is the padding applied to the 
%                                       left, and r is the padding applied 
%                                       to the right.
%                                   Note that the padding dimensions must
%                                   be less than the pooling region
%                                   dimensions. The default is 0.
%       'NumChannels'             - The number of channels for each filter.
%                                   If a value of 'auto' is passed in, the
%                                   correct value for this parameter will
%                                   be inferred at training time. The
%                                   default is 'auto'.
%       'WeightLearnRateFactor'   - A number that specifies multiplier for
%                                   the learning rate of the weights. The
%                                   default is 1.
%       'BiasLearnRateFactor'     - A number that specifies a multiplier
%                                   for the learning rate for the biases. 
%                                   The default is 1.
%       'WeightL2Factor'          - A number that specifies a multiplier
%                                   for the L2 weight regulariser for the
%                                   weights. The default is 1.
%       'BiasL2Factor'            - A number that specifies a multiplier
%                                   for the L2 weight regulariser for the
%                                   biases. The default is 0.
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%
%   Example 1:
%       Create a convolutional layer with 96 filters that have a height and
%       width of 11, and use a stride of 4 in the horizontal and vertical 
%       directions.
%
%       layer = Binarziedconvolution2dLayer(11, 96, 'Stride', 4);
%
%   Example 2:
%       Create a convolutional layer with 32 filters that have a height and
%       width of 5. Pad the input image with 2 pixels along its border. Set
%       the learning rate factor for the bias to 2. Manually initialize the
%       weights from a Gaussian with standard deviation 0.0001.
%
%       layer = Binarziedconvolution2dLayer(5, 32, 'Padding', 2, 'BiasLearnRateFactor', 2);
%       layer.Weights = randn([5 5 3 32])*0.0001;
%
%   Example 3:
%       Create a convolutional layer with 32 filters that have a height and
%       width of 3. Set the dilation factor to 12 in both the horizontal 
%       and vertical direction.
%
%       layer = Binarziedconvolution2dLayer(3, 32, 'DilationFactor', 12);
%    
%
%   See also nnet.cnn.layer.Binarziedconvolution2dLayer, maxPooling2dLayer, 
%   averagePooling2dLayer.

%   Copyright 2015-2018 The MathWorks, Inc.

% Parse the input arguments.
args = iParseInputArguments(varargin{:});

% Create an internal representation of a convolutional layer.
internalLayer = BinarizedConvolution2D(args.Name, ...
                                                      args.FilterSize, ...
                                                      args.NumChannels, ...
                                                      args.NumFilters, ...
                                                      args.Stride, ...
                                                      args.DilationFactor, ...
                                                      args.PaddingMode, ...
                                                      args.PaddingSize);

internalLayer.Weights.L2Factor = args.WeightL2Factor;
internalLayer.Weights.LearnRateFactor = args.WeightLearnRateFactor;

internalLayer.Bias.L2Factor = args.BiasL2Factor;
internalLayer.Bias.LearnRateFactor = args.BiasLearnRateFactor;

% Pass the internal layer to a function to construct a user visible
% convolutional layer.
layer = BinarizedConvolution2DLayerFixer(internalLayer);

end

function inputArguments = iParseInputArguments(varargin)
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = iConvertToCanonicalForm(parser.Results);
end

function p = iCreateParser()
p = inputParser;
defaultStride = 1;
defaultDilationFactor = 1;
defaultPadding = 0;
defaultNumChannels = 'auto';
defaultWeightLearnRateFactor = 1;
defaultBiasLearnRateFactor = 1;
defaultWeightL2Factor = 1;
defaultBiasL2Factor = 0;
defaultName = '';

p.addParameter('Name', defaultName, @iAssertValidLayerName);
p.addRequired('FilterSize',@iAssertValidFilterSize);
p.addRequired('NumFilters',@iAssertValidNumFilters);
p.addParameter('Stride', defaultStride, @iAssertValidStride);
p.addParameter('DilationFactor', defaultDilationFactor, @iAssertValidDilationFactor);
p.addParameter('Padding', defaultPadding, @iAssertValidPadding);
p.addParameter('NumChannels', defaultNumChannels, @iAssertValidNumChannels);
p.addParameter('WeightLearnRateFactor', defaultWeightLearnRateFactor, @iAssertValidFactor);
p.addParameter('BiasLearnRateFactor', defaultBiasLearnRateFactor, @iAssertValidFactor);
p.addParameter('WeightL2Factor', defaultWeightL2Factor, @iAssertValidFactor);
p.addParameter('BiasL2Factor', defaultBiasL2Factor, @iAssertValidFactor);
end

function iAssertValidFilterSize(value)
nnet.internal.cnn.layer.paramvalidation.validateSizeParameter(value, 'FilterSize');
end

function iAssertValidNumFilters(value)
validateattributes(value, {'numeric'}, ...
    {'scalar','integer','positive'});
end

function iAssertValidStride(value)
nnet.internal.cnn.layer.paramvalidation.validateSizeParameter(value, 'Stride');
end

function iAssertValidDilationFactor(value)
nnet.internal.cnn.layer.paramvalidation.validateSizeParameter(value, 'DilationFactor');
end

function iAssertValidPadding(value)
nnet.internal.cnn.layer.paramvalidation.validatePadding(value);
end

function iAssertValidNumChannels(value)
if(ischar(value))
    validatestring(value,{'auto'});
else
    validateattributes(value, {'numeric'}, ...
        {'scalar','integer','positive'});
end
end

function iAssertValidFactor(value)
nnet.internal.cnn.layer.paramvalidation.validateLearnFactor(value);
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name)
end

function inputArguments = iConvertToCanonicalForm(params)
% Make sure integral values are converted to double and strings to char vectors
inputArguments = struct;
inputArguments.FilterSize = double( iMakeIntoRowVectorOfTwo(params.FilterSize) );
inputArguments.NumFilters = double( params.NumFilters );
inputArguments.Stride = double( iMakeIntoRowVectorOfTwo(params.Stride) );
inputArguments.DilationFactor = double( iMakeIntoRowVectorOfTwo(params.DilationFactor) );
inputArguments.PaddingMode = iCalculatePaddingMode(params.Padding);
inputArguments.PaddingSize = double( iCalculatePaddingSize(params.Padding) );
inputArguments.NumChannels = double( iConvertToEmptyIfAuto(params.NumChannels) );
inputArguments.WeightLearnRateFactor = params.WeightLearnRateFactor;
inputArguments.BiasLearnRateFactor = params.BiasLearnRateFactor;
inputArguments.WeightL2Factor = params.WeightL2Factor;
inputArguments.BiasL2Factor = params.BiasL2Factor;
inputArguments.Name = char(params.Name);
end

function tf = iIsRowVectorOfTwo(x)
tf = isrow(x) && numel(x)==2;
end

function rowVectorOfTwo = iMakeIntoRowVectorOfTwo(scalarOrRowVectorOfTwo)
if(iIsRowVectorOfTwo(scalarOrRowVectorOfTwo))
    rowVectorOfTwo = scalarOrRowVectorOfTwo;
else
    rowVectorOfTwo = [scalarOrRowVectorOfTwo scalarOrRowVectorOfTwo];
end
end

function paddingMode = iCalculatePaddingMode(padding)
paddingMode = nnet.internal.cnn.layer.padding.calculatePaddingMode(padding);
end

function paddingSize = iCalculatePaddingSize(padding)
paddingSize = nnet.internal.cnn.layer.padding.calculatePaddingSize(padding);
end

function y = iConvertToEmptyIfAuto(x)
if(iIsAutoString(x))
    y = [];
else
    y = x;
end
end

function tf = iIsAutoString(x)
tf = strcmp(x, 'auto');
end
