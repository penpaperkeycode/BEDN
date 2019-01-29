function layer = BinarizedtransposedConv2dLayer( varargin )
% transposedConv2dLayer 2D transposed convolution layer
%
%   layer = transposedConv2dLayer(filterSize, numFilters) creates a
%   transposed 2D convolution layer. This layer is used to upsample feature
%   maps. filterSize specifies the height and width of the filters. It can
%   be a scalar, in which case the filters will have the same height and
%   width, or a vector [h w] where h specifies the height for the filters,
%   and w specifies the width. numFilters specifies the number of filters,
%   which determines the number of channels in the output feature map.
%
%   layer = transposedConv2dLayer(filterSize, numFilters, 'PARAM1', VAL1, ...)
%   specifies optional parameter name/value pairs for creating the layer:
%
%       'Stride'                  - The stride used to upsample the input
%                                   vertically and horizontally. This can
%                                   be a scalar, in which case the same
%                                   value is used for both dimensions, or
%                                   it can be a vector [u v] where u is the
%                                   vertical stride, and v is the
%                                   horizontal stride. The default is
%                                   [1 1] and no upsampling is performed.
%
%       'Cropping'                - Trim the vertical and horizontal
%                                   edges of the full transposed
%                                   convolution by the specified value. Use
%                                   this parameter when you need to reduce
%                                   the output size of the layer. The
%                                   default is 0.
%
%                                   Valid 'Cropping' values are:
%                                     - a scalar, in which case the same
%                                     amount of data is trimmed from the
%                                     all vertical and horizontal edges.
%                                     - a vector [a b] where a is the
%                                     amount to trim from the top and
%                                     bottom, and b is the amount to
%                                     trim from the left and right.
%
%       'NumChannels'             - The number of channels for each filter.
%                                   If a value of 'auto' is passed in, the
%                                   correct value for this parameter will
%                                   be inferred at training time. The
%                                   default is 'auto'.
%
%       'WeightLearnRateFactor'   - A number that specifies multiplier for
%                                   the learning rate of the weights. The
%                                   default is 1.
%
%       'BiasLearnRateFactor'     - A number that specifies a multiplier
%                                   for the learning rate for the biases.
%                                   The default is 1.
%
%       'WeightL2Factor'          - A number that specifies a multiplier
%                                   for the L2 weight regulariser for the
%                                   weights. The default is 1.
%
%       'BiasL2Factor'            - A number that specifies a multiplier
%                                   for the L2 weight regulariser for the
%                                   biases. The default is 0.
%
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%
%   Example 1:
%       Create a transposed convolutional layer with 96 filters that have a
%       height and width of 11, and use a stride of 4 in the horizontal and
%       vertical directions.
%
%       layer = transposedConv2dLayer(11, 96, 'Stride', 4);
%
%   See also nnet.cnn.layer.TransposedConvolution2DLayer, convolution2dLayer.

%   Copyright 2017 The MathWorks, Inc.

% Parse the input arguments.

args = iParseInputArguments(varargin{:});

% Create an internal representation of a convolutional layer.
internalLayer = BinarizedTransposedConvolution2D(args.Name, ...
    args.FilterSize, ...
    args.NumChannels, ...
    args.NumFilters, ...
    args.Stride, ...
    args.Cropping);

internalLayer.Weights.L2Factor = args.WeightL2Factor;
internalLayer.Weights.LearnRateFactor = args.WeightLearnRateFactor;

internalLayer.Bias.L2Factor = args.BiasL2Factor;
internalLayer.Bias.LearnRateFactor = args.BiasLearnRateFactor;

% Pass the internal layer to a function to construct a user visible
% convolutional layer.
layer = BinarizedTransposedConvolution2DLayerFixer(internalLayer);

end

function inputArguments = iParseInputArguments(varargin)
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = iConvertToCanonicalForm(parser);
end

function p = iCreateParser()
p = inputParser;

defaultStride = 1;
defaultCropping = 0;
defaultNumChannels = 'auto';
defaultWeightLearnRateFactor = 1;
defaultBiasLearnRateFactor = 1;
defaultWeightL2Factor = 1;
defaultBiasL2Factor = 0;
defaultName = '';

p.addRequired('FilterSize', @iAssertValidFilterSize);
p.addRequired('NumFilters', @iAssertValidNumFilters);
p.addParameter('Stride', defaultStride, @iAssertValidStride);
p.addParameter('Cropping', defaultCropping, @iAssertValidCropping);
p.addParameter('NumChannels', defaultNumChannels, @iAssertValidNumChannels);
p.addParameter('WeightLearnRateFactor', defaultWeightLearnRateFactor, @iAssertValidFactor);
p.addParameter('BiasLearnRateFactor', defaultBiasLearnRateFactor, @iAssertValidFactor);
p.addParameter('WeightL2Factor', defaultWeightL2Factor, @iAssertValidFactor);
p.addParameter('BiasL2Factor', defaultBiasL2Factor, @iAssertValidFactor);
p.addParameter('Name', defaultName, @iAssertValidLayerName);
end

function inputArguments = iConvertToCanonicalForm(p)
% Make sure integral values are converted to double and strings to char vectors
inputArguments = struct;
inputArguments.FilterSize = double( iMakeIntoRowVectorOfTwo(p.Results.FilterSize) );
inputArguments.NumFilters = double( p.Results.NumFilters );
inputArguments.Stride = double( iMakeIntoRowVectorOfTwo(p.Results.Stride) );
inputArguments.Cropping = double( iMakeIntoRowVectorOfTwo(p.Results.Cropping) );
inputArguments.NumChannels = double( iConvertToEmptyIfAuto(p.Results.NumChannels) );
inputArguments.WeightLearnRateFactor = p.Results.WeightLearnRateFactor;
inputArguments.BiasLearnRateFactor = p.Results.BiasLearnRateFactor;
inputArguments.WeightL2Factor = p.Results.WeightL2Factor;
inputArguments.BiasL2Factor = p.Results.BiasL2Factor;
inputArguments.Name = char(p.Results.Name); 
end

function iAssertValidFilterSize(value)
validateattributes(value, {'numeric'}, ...
    {'positive', 'real', 'integer', 'nonempty'});
iAssertScalarOrRowVectorOfTwo(value,'FilterSize');
end

function iAssertValidNumFilters(value)
validateattributes(value, {'numeric'}, ...
    {'scalar','integer','positive'});
end

function iAssertValidStride(value)
nnet.internal.cnn.layer.paramvalidation.validateSizeParameter(value, 'Stride');
end

function iAssertValidCropping(value)
validateattributes(value, {'numeric'}, ...
    {'nonnegative', 'real', 'integer', 'nonempty'});
iAssertScalarOrRowVectorOfTwo(value,'Cropping');
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

function rowVectorOfTwo = iMakeIntoRowVectorOfTwo(scalarOrRowVectorOfTwo)
if(iIsRowVectorOfTwo(scalarOrRowVectorOfTwo))
    rowVectorOfTwo = scalarOrRowVectorOfTwo;
else
    rowVectorOfTwo = [scalarOrRowVectorOfTwo scalarOrRowVectorOfTwo];
end
end

function iAssertScalarOrRowVectorOfTwo(value,name)
if ~(isscalar(value) || iIsRowVectorOfTwo(value))
    exception = MException(message('nnet_cnn:layer:Layer:ParamMustBeScalarOrPair',name));
    throwAsCaller(exception);
end
end

function tf = iIsRowVectorOfTwo(x)
tf = isrow(x) && numel(x)==2;
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