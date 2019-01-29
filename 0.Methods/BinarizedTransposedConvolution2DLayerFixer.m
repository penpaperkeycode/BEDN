classdef BinarizedTransposedConvolution2DLayerFixer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % TransposedConvolution2DLayer   2-D transposed convolution layer
    %
    %   To create a transposed convolution layer, use transposedConv2dLayer.
    %
    %   TransposedConvolution2DLayer properties:
    %       Name                        - A name for the layer.
    %       FilterSize                  - The height and width of the
    %                                     filters.
    %       NumChannels                 - The number of channels for each
    %                                     filter.
    %       NumFilters                  - The number of filters.
    %       Stride                      - The step size for traversing the
    %                                     input vertically and
    %                                     horizontally.
    %       Cropping                    - The amount to trim from the
    %                                     vertical and horizontal edges of
    %                                     the full transposed convolution
    %                                     size.
    %       Weights                     - Weights of the layer.
    %       Bias                        - Biases of the layer.
    %       WeightLearnRateFactor       - A number that specifies
    %                                     multiplier for the learning rate
    %                                     of the weights.
    %       BiasLearnRateFactor         - A number that specifies a
    %                                     multiplier for the learning rate
    %                                     for the biases.
    %       WeightL2Factor              - A number that specifies a
    %                                     multiplier for the L2 weight
    %                                     regulariser for the weights.
    %       BiasL2Factor                - A number that specifies a
    %                                     multiplier for the L2 weight
    %                                     regulariser for the biases.
    %
    %   Example:
    %       Create a transposed convolution layer with 5 filters of size 10-by-10.
    %
    %       layer = transposedConv2dLayer(10, 5);
    %
    %   See also transposedConv2dLayer, convolution2dLayer.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name         
    end
        
    properties(SetAccess = private, Dependent)
        % FilterSize   The height and width of the filters
        %   The height and width of the filters. This is a row vector [h w]
        %   where h is the filter height and w is the filter width.
        FilterSize
        
        % NumChannels   The number of channels in the input
        %   The number of channels in the input. This can be set to 'auto',
        %   in which case the correct value will be determined at training
        %   time.
        NumChannels
        
        % NumFilters   The number of filters
        %   The number of filters for this layer. This also determines how
        %   many maps there will be in the output.
        NumFilters
        
        % Stride   The vertical and horizontal stride
        %   The step size for traversing the input vertically and
        %   horizontally. This is a row vector [u v] where u is the
        %   vertical stride, and v is the horizontal stride.
        Stride
        
        % Cropping   The vertical and horizontal output cropping
        %   The amount to trim the full transposed convolution output along
        %   the vertical and horizontal edges. The amount to crop is a row
        %   vector [a b] where a is the amount removed from the top and
        %   bottom, and b is the amount removed from the left and right.
        Cropping
    end
    
    properties(Dependent)
        % Weights   The weights for the layer
        %   The filters for the convolutional layer. An array with size
        %   FilterSize(1)-by-FilterSize(2)-by-NumFilters-by-NumChannels.
        Weights
        
        % Bias   The bias vector for the layer
        %   The bias for the convolutional layer. The size will be
        %   1-by-1-by-NumFilters.
        Bias
        
        % WeightLearnRateFactor   The learning rate factor for the weights
        %   The learning rate factor for the weights. This factor is
        %   multiplied with the global learning rate to determine the
        %   learning rate for the weights in this layer. For example, if it
        %   is set to 2, then the learning rate for the weights in this
        %   layer will be twice the current global learning rate.
        WeightLearnRateFactor
        
        % WeightL2Factor   The L2 regularization factor for the weights
        %   The L2 regularization factor for the weights. This factor is
        %   multiplied with the global L2 regularization setting to
        %   determine the L2 regularization setting for the weights in this
        %   layer. For example, if it is set to 2, then the L2
        %   regularization for the weights in this layer will be twice the
        %   global L2 regularization setting.
        WeightL2Factor
        
        % BiasLearnRateFactor   The learning rate factor for the biases
        %   The learning rate factor for the bias. This factor is
        %   multiplied with the global learning rate to determine the
        %   learning rate for the bias in this layer. For example, if it
        %   is set to 2, then the learning rate for the bias in this layer
        %   will be twice the current global learning rate.
        BiasLearnRateFactor
        
        % BiasL2Factor   The L2 regularization factor for the biases
        %   The L2 regularization factor for the biases. This factor is
        %   multiplied with the global L2 regularization setting to
        %   determine the L2 regularization setting for the biases in this
        %   layer. For example, if it is set to 2, then the L2
        %   regularization for the biases in this layer will be twice the
        %   global L2 regularization setting.
        BiasL2Factor
    end
    
    methods
        function this = BinarizedTransposedConvolution2DLayerFixer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            privateLayer = this.PrivateLayer;
            out.Version = 1.0;
            out.Name = privateLayer.Name;
            out.FilterSize = privateLayer.FilterSize;
            out.NumChannels = privateLayer.NumChannels;
            out.NumFilters = privateLayer.NumFilters;
            out.Stride = privateLayer.Stride;
            out.Cropping = privateLayer.Padding;
            out.Weights = toStruct(privateLayer.Weights);
            out.Bias = toStruct(privateLayer.Bias);
        end

        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function this = set.Name(this, val)
            iAssertValidLayerName(val);
            this.PrivateLayer.Name = char(val);
        end
        
        function val = get.Weights(this)
            val = this.PrivateLayer.Weights.HostValue;
        end
        
        function this = set.Weights(this, value)
            classes = {'single', 'double', 'gpuArray'};
            
            expectedNumChannels = iExpectedNumChannels(this.NumChannels);
            
            attributes = {'size', [this.FilterSize this.NumFilters expectedNumChannels], 'nonempty', 'real'};
            validateattributes(value, classes, attributes);
            
            % Call inferSize to determine the size of the layer. In
            % transposed convolution weights, 4 dim stores numChannels.
            inputChannels = size(value,4);
            
            this.PrivateLayer = this.PrivateLayer.inferSize( [NaN NaN inputChannels] );
            this.PrivateLayer.Weights.Value = gather(value);
        end
        
        function val = get.Bias(this)
            val = this.PrivateLayer.Bias.HostValue;
        end
        
        function this = set.Bias(this, value)
            classes = {'single', 'double', 'gpuArray'};
            attributes = {'size', [1 1 this.NumFilters], 'nonempty', 'real'};
            validateattributes(value, classes, attributes);
            
            this.PrivateLayer.Bias.Value = gather(value);
        end
        
        function val = get.FilterSize(this)
            val = this.PrivateLayer.FilterSize;
        end
        
        function val = get.NumChannels(this)
            val = this.PrivateLayer.NumChannels;
            if isempty(val)
                val = 'auto';
            end
        end
        
        function val = get.NumFilters(this)
            val = this.PrivateLayer.NumFilters;
        end
        
        function val = get.Stride(this)
            val = this.PrivateLayer.Stride;
        end
        
        function val = get.Cropping(this)
            val = this.PrivateLayer.Padding;
        end
        
        function val = get.WeightLearnRateFactor(this)
            val = this.PrivateLayer.Weights.LearnRateFactor;
        end
        
        function this = set.WeightLearnRateFactor(this, value)
            iAssertValidFactor(value,'WeightLearnRateFactor');
            this.PrivateLayer.Weights.LearnRateFactor = value;
        end
        
        function val = get.BiasLearnRateFactor(this)
            val = this.PrivateLayer.Bias.LearnRateFactor;
        end
        
        function this = set.BiasLearnRateFactor(this, value)
            iAssertValidFactor(value,'BiasLearnRateFactor');
            this.PrivateLayer.Bias.LearnRateFactor = value;
        end
        
        function val = get.WeightL2Factor(this)
            val = this.PrivateLayer.Weights.L2Factor;
        end
        
        function this = set.WeightL2Factor(this, value)
            iAssertValidFactor(value,'WeightL2Factor');
            this.PrivateLayer.Weights.L2Factor = value;
        end
        
        function val = get.BiasL2Factor(this)
            val = this.PrivateLayer.Bias.L2Factor;
        end
        
        function this = set.BiasL2Factor(this, value)
            iAssertValidFactor(value,'BiasL2Factor');
            this.PrivateLayer.Bias.L2Factor = value;
        end
    end
    
    methods(Static)
        function this = loadobj(in)
            this = iLoadLayerFromCurrentVersion(in);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(this)
            numFiltersString = int2str( this.NumFilters );
            filterSizeString = i2DSizeToString( this.FilterSize );
            if ~isequal(this.NumChannels, 'auto')
                numChannelsString = ['x' int2str( this.NumChannels )];
            else
                numChannelsString = '';
            end
            strideString = int2str( this.Stride );
            croppingString = int2str( this.Cropping );

            description = iGetMessageString( ...
                'nnet_cnn:layer:TransposedConvolution2DLayer:oneLineDisplay', ...
                numFiltersString, ...
                filterSizeString, ...
                numChannelsString, ...
                strideString, ...
                croppingString );
            
            type = iGetMessageString( 'nnet_cnn:layer:TransposedConvolution2DLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            hyperparameters = {
                'FilterSize'
                'NumChannels'
                'NumFilters'
                'Stride'
                'Cropping'
                };
            
            learnableParameters = {'Weights', 'Bias'};
            
            groups = [
                this.propertyGroupGeneral( {'Name'} )
                this.propertyGroupHyperparameters( hyperparameters )
                this.propertyGroupLearnableParameters( learnableParameters )
                ];
        end
        
        function footer = getFooter( this )
            variableName = inputname(1);
            footer = this.createShowAllPropertiesFooter( variableName );
        end
        
    end
end

function messageString = iGetMessageString( varargin )
messageString = getString( message( varargin{:} ) );
end

function sizeString = i2DSizeToString( sizeVector )
% i2DSizeToString   Convert a 2-D size stored in a vector of 2 elements
% into a string separated by 'x'.
sizeString = [ ...
    int2str( sizeVector(1) ) ...
    'x' ...
    int2str( sizeVector(2) ) ];
end

function obj = iLoadLayerFromCurrentVersion(in)
internalLayer = BinarizedTransposedConvolution2D( ...
    in.Name, in.FilterSize, in.NumChannels, ...
    in.NumFilters, in.Stride, in.Cropping);
internalLayer.Weights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.Weights);
internalLayer.Bias = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.Bias);
            
obj = BinarizedTransposedConvolution2DLayerFixer(internalLayer);
end

function iAssertValidFactor(value,factorName)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateLearnFactor(value,factorName));
end

function expectedNumChannels = iExpectedNumChannels(NumChannels)
expectedNumChannels = NumChannels;
if isequal(expectedNumChannels, 'auto')
    expectedNumChannels = NaN;
end
end

function iAssertValidLayerName(name)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateLayerName(name));
end

function iEvalAndThrow(func)
% Omit the stack containing internal functions by throwing as caller
try
    func();
catch exception
    throwAsCaller(exception)
end
end