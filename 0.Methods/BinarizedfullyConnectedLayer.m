function layer = BinarizedfullyConnectedLayer( varargin )
% BinarizedfullyConnectedLayer   Fully connected layer
%
%   layer = BinarizedfullyConnectedLayer(outputSize) creates a fully connected
%   layer. outputSize specifies the size of the output for the layer. A
%   fully connected layer will multiply the input by a matrix and then add
%   a bias vector.
%
%   layer = BinarizedfullyConnectedLayer(outputSize, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
%   specifies optional parameter name/value pairs for creating the layer:
%
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
%   Example 1:
%       Create a fully connected layer with an output size of 10, and an
%       input size that will be determined at training time.
%
%       layer = BinarizedfullyConnectedLayer(10);
%
%   See also nnet.cnn.layer.BinarizedfullyConnectedLayer, convolution2dLayer,
%   reluLayer.

%   Copyright 2015-2017 The MathWorks, Inc.

% Parse the input arguments.
args = iParseInputArguments(varargin{:});

% Create an internal representation of a fully connected layer.
internalLayer = BinarizedFullyConnected( ...
    args.Name, ...
    args.InputSize, ...
    args.OutputSize);

internalLayer.Weights.L2Factor = args.WeightL2Factor;
internalLayer.Weights.LearnRateFactor = args.WeightLearnRateFactor;

internalLayer.Bias.L2Factor = args.BiasL2Factor;
internalLayer.Bias.LearnRateFactor = args.BiasLearnRateFactor;

% Pass the internal layer to a function to construct a user visible
% fully connected layer.
layer = BinarizedFullyConnectedLayerFixer(internalLayer);
end

function inputArguments = iParseInputArguments(varargin)
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = iConvertToCanonicalForm(parser);
end

function p = iCreateParser()
p = inputParser;

defaultWeightLearnRateFactor = 1;
defaultBiasLearnRateFactor = 1;
defaultWeightL2Factor = 1;
defaultBiasL2Factor = 0;
defaultName = '';

p.addRequired('OutputSize', @iAssertValidOutputSize);
p.addParameter('WeightLearnRateFactor', defaultWeightLearnRateFactor, @iAssertValidFactor);
p.addParameter('BiasLearnRateFactor', defaultBiasLearnRateFactor, @iAssertValidFactor);
p.addParameter('WeightL2Factor', defaultWeightL2Factor, @iAssertValidFactor);
p.addParameter('BiasL2Factor', defaultBiasL2Factor, @iAssertValidFactor);
p.addParameter('Name', defaultName, @iAssertValidLayerName);
end

function iAssertValidFactor(value)
nnet.internal.cnn.layer.paramvalidation.validateLearnFactor(value);
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name)
end

function inputArguments = iConvertToCanonicalForm(p)
% Make sure integral values are converted to double and strings to char vectors
inputArguments = struct;
inputArguments.InputSize = [];
inputArguments.OutputSize = double( p.Results.OutputSize );
inputArguments.WeightLearnRateFactor = p.Results.WeightLearnRateFactor;
inputArguments.BiasLearnRateFactor = p.Results.BiasLearnRateFactor;
inputArguments.WeightL2Factor = p.Results.WeightL2Factor;
inputArguments.BiasL2Factor = p.Results.BiasL2Factor;
inputArguments.Name = char(p.Results.Name); 
end

function iAssertValidOutputSize(value)
validateattributes(value, {'numeric'}, ...
    {'nonempty', 'scalar', 'integer', 'positive'});
end