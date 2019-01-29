classdef BinarizedConvolution2DLayerFixer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % BinarizedConvolution2DLayerFixer   2-D convolution layer
    %
    %   To create a convolution layer, use BinarizedConvolution2DLayerFixer
    %
    %   BinarizedConvolution2DLayerFixer properties:
    %       Name                        - A name for the layer.
    %       FilterSize                  - The height and width of the
    %                                     filters.
    %       NumChannels                 - The number of channels for each
    %                                     filter.
    %       NumFilters                  - The number of filters.
    %       Stride                      - The step size for traversing the
    %                                     input vertically and
    %                                     horizontally.
    %       DilationFactor              - The vertical and horizontal 
    %                                     up-sampling factor of the filter.
    %       PaddingMode                 - The mode used to determine the
    %                                     padding.
    %       PaddingSize                 - The padding applied to the input 
    %                                     along the edges.
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
    %       Create a convolution layer with 5 filters of size 10-by-10.
    %
    %       layer = BinarizedConvolution2DLayerFixer(10, 5);
    %
    %   See also BinarizedConvolution2DLayerFixer.
    
    %   Copyright 2015-2018 The MathWorks, Inc.
    
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
        
        % PaddingMode   The mode used to determine the padding
        %   The mode used to calculate the PaddingSize property. This can
        %   be:
        %       'manual'    - PaddingSize is specified manually. 
        %       'same'      - PaddingSize is calculated so that the output
        %                     is the same size as the input when the stride
        %                     is 1. More generally, the output size will be
        %                     ceil(inputSize/stride), where inputSize is 
        %                     the height and width of the input.
        PaddingMode
    end
    
    properties(SetAccess = private, Dependent, Hidden)
        % Padding   The vertical and horizontal padding
        %   Padding property will be removed in a future release. Use 
        %   PaddingSize instead.
        %
        %   The padding that is applied to the input vertically and
        %   horizontally. This a row vector [a b] where a is the padding
        %   applied to the top and bottom of the input, and b is the
        %   padding applied to the left and right of the image.
        Padding
    end
    
    properties(Dependent)
        % Stride   The vertical and horizontal stride
        %   The step size for traversing the input vertically and
        %   horizontally. This is a row vector [u v] where u is the
        %   vertical stride, and v is the horizontal stride.
        Stride
        
        % DilationFactor   The up-sampling factor of the filter.  
        % It correponds to an effective filter size of 
        % filterSize + (filterSize-1) * (DilationFactor-1), but the size
        % of the weights does not depend on the dilation factor.
        % This can be a scalar, in which case the same value is used for 
        % both dimensions, or it can be a vector [d_h d_w] where d_h is 
        % the vertical dilation, and d_w is the horizontal dilation.
        DilationFactor
                
        % PaddingSize   The padding applied to the input along the edges
        %   The padding that is applied along the edges. This is a row 
        %   vector [t b l r] where t is the padding to the top, b is the 
        %   padding applied to the bottom, l is the padding applied to the 
        %   left, and r is the padding applied to the right.
        PaddingSize
        
        % Weights   The weights for the layer
        %   The filters for the convolutional layer. An array with size
        %   FilterSize(1)-by-FilterSize(2)-by-NumChannels-by-NumFilters.
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
        function this = BinarizedConvolution2DLayerFixer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            privateLayer = this.PrivateLayer;
            out.Version = 4.0;
            out.Name = privateLayer.Name;
            out.FilterSize = privateLayer.FilterSize;
            out.NumChannels = privateLayer.NumChannels;
            out.NumFilters = privateLayer.NumFilters;
            out.Stride = privateLayer.Stride;
            out.DilationFactor = privateLayer.DilationFactor;
            out.PaddingMode = privateLayer.PaddingMode;
            out.PaddingSize = privateLayer.PaddingSize;
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
        
        function this = set.Stride(this, value)
            iAssertValidStride(value);
            % Convert to canonical form
            this.PrivateLayer.Stride = double(iMakeIntoRowVectorOfTwo(value));
        end
        
        function this = set.DilationFactor(this, value)
            iAssertValidDilationFactor(value);
            % Convert to canonical form
            this.PrivateLayer.DilationFactor = double(iMakeIntoRowVectorOfTwo(value));
        end
        
        function this = set.PaddingSize(this, value)
            if isequal(this.PaddingMode,'same')
                error(message('nnet_cnn:layer:Layer:PaddingSizeCanOnlyBeSetInManualMode'))
            end
            iAssertValidPaddingSize(value);
            % Convert to canonical form
            this.PrivateLayer.PaddingSize = double(iCalculatePaddingSize(value));
        end
        
        function val = get.Weights(this)
            val = this.PrivateLayer.Weights.HostValue;
        end
        
        function this = set.Weights(this, value)
            classes = {'single', 'double', 'gpuArray'};
            if(this.filterGroupsAreUsed())
                expectedNumChannels = iExpectedNumChannels(this.NumChannels(1));
            else
                expectedNumChannels = iExpectedNumChannels(this.NumChannels);
            end
            attributes = {'size', [this.FilterSize expectedNumChannels sum(this.NumFilters)], 'nonempty', 'real'};
            validateattributes(value, classes, attributes);
            
            % Call inferSize to determine the size of the layer
            if(this.filterGroupsAreUsed())
                inputChannels = size(value,3)*2;
            else
                inputChannels = size(value,3);
            end
            this.PrivateLayer = this.PrivateLayer.inferSize( [NaN NaN inputChannels] );
            this.PrivateLayer.Weights.Value = gather(value);
        end
        
        function val = get.Bias(this)
            val = this.PrivateLayer.Bias.HostValue;
        end
        
        function this = set.Bias(this, value)
            classes = {'single', 'double', 'gpuArray'};
            attributes = {'size', [1 1 sum(this.NumFilters)], 'nonempty', 'real'};
            validateattributes(value, classes, attributes);
            
            this.PrivateLayer.Bias.Value = gather(value);
        end
        
        function val = get.FilterSize(this)
            val = this.PrivateLayer.FilterSize;
        end
        
        function val = get.NumChannels(this)
            val = this.PrivateLayer.NumChannels;
            if(this.filterGroupsAreUsed())
                val = [val val];
            end
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

        function val = get.DilationFactor(this)
            val = this.PrivateLayer.DilationFactor;
        end
        
        function val = get.PaddingMode(this)
            val = this.PrivateLayer.PaddingMode;
        end
        
        function val = get.PaddingSize(this)
            val = this.PrivateLayer.PaddingSize;
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
    
    methods
        function val = get.Padding(this)
            % This is required for backward compatibility.
            iValidatePaddingCanBeExpressedAs1By2Vector( ...
                this.PrivateLayer.PaddingSize);
            val = [this.PrivateLayer.PaddingSize(1) ...
                this.PrivateLayer.PaddingSize(3)];
            warning(message('nnet_cnn:layer:Convolution2DLayer:PaddingObsolete'));
        end
    end
    
    methods(Static)
        function this = loadobj(in)
            if in.Version <= 1
                in = iUpgradeVersionOneToVersionTwo(in);
            end
            if in.Version <= 2
                in = iUpgradeVersionTwoToVersionThree(in);
            end
            if in.Version <= 3
                in = iUpgradeVersionThreeToVersionFour(in);
            end
            this = iLoadConvolution2DLayerFromCurrentVersion(in);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(this)
            numFiltersString = int2str( sum(this.NumFilters) );
            filterSizeString = i2DSizeToString( this.FilterSize );
            if ~isequal(this.NumChannels, 'auto')
                % When using filter groups, the number of channels is
                % replicated to match NumFilters. For display, only show
                % the first element.
                numChannelsString = ['x' int2str( this.NumChannels(1) )];
            else
                numChannelsString = '';
            end
            strideString = "["+int2str( this.Stride )+"]";
            dilationFactorString = "["+int2str( this.DilationFactor )+"]";
            
            if this.PaddingMode ~= "manual"
                paddingSizeString = "'"+this.PaddingMode+"'";
            else
                paddingSizeString = "["+int2str( this.PaddingSize )+"]";
            end
            
            if isequal(this.DilationFactor, [1 1])
                description = iGetMessageString( ...
                    'nnet_cnn:layer:Convolution2DLayer:oneLineDisplayNoDilation', ...
                    numFiltersString, ...
                    filterSizeString, ...
                    numChannelsString, ...
                    strideString, ...
                    paddingSizeString );
            else
                description = iGetMessageString( ...
                    'nnet_cnn:layer:Convolution2DLayer:oneLineDisplay', ...
                    numFiltersString, ...
                    filterSizeString, ...
                    numChannelsString, ...
                    strideString, ...
                    dilationFactorString, ...
                    paddingSizeString );
            end
            type = iGetMessageString( 'nnet_cnn:layer:Convolution2DLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            hyperparameters = {
                'FilterSize'
                'NumChannels'
                'NumFilters'
                'Stride'
                'DilationFactor'
                'PaddingMode'
                'PaddingSize'
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
        
        function tf = filterGroupsAreUsed(this)
            tf = numel(this.NumFilters) ~= 1;
        end
    end
end

function iValidatePaddingCanBeExpressedAs1By2Vector(paddingSize)
if(iPaddingCanBeExpressedAs1By2Vector(paddingSize))
else
    error(message('nnet_cnn:layer:Convolution2DLayer:PaddingCannotBeAsymmetric'));
end
end

function tf = iPaddingCanBeExpressedAs1By2Vector(paddingSize)
tf = (paddingSize(1) == paddingSize(2)) && (paddingSize(3) == paddingSize(4));
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

function iAssertValidFactor(value,factorName)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateLearnFactor(value,factorName));
end

function S = iUpgradeVersionOneToVersionTwo(S)
% iUpgradeVersionOneToVersionTwo   Upgrade a v1 (2016a) saved struct to a v2 saved struct
%   This means gathering the bias and weights from the GPU and putting them
%   on the host.

S.Version = 2;
try
    S.Weights.Value = gather(S.Weights.Value);
    S.Bias.Value = gather(S.Bias.Value);
catch e
    % Only throw the error we want to throw.
    e = MException( message('nnet_cnn:layer:Convolution2DLayer:MustHaveGPUToLoadFrom2016a'));
    throwAsCaller(e);
end
end

function S = iUpgradeVersionTwoToVersionThree(S)
S.Version = 3;
S.PaddingMode = 'manual';
S.PaddingSize = iCalculatePaddingSize(S.Padding);
end

function S = iUpgradeVersionThreeToVersionFour(S)
S.Version = 4;
% Set the dilation factor to its default value in canonical form
S.DilationFactor = [1 1];
end

function obj = iLoadConvolution2DLayerFromCurrentVersion(in)
internalLayer = BinarizedConvolution2D( ...
    in.Name, in.FilterSize, in.NumChannels, ...
    in.NumFilters, in.Stride, in.DilationFactor, in.PaddingMode, in.PaddingSize);
internalLayer.Weights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.Weights);
internalLayer.Bias = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.Bias);

obj = BinarizedConvolution2DLayerFixer(internalLayer);
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

function iAssertValidStride(value)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateSizeParameter(value, 'Stride'));
end

function iAssertValidDilationFactor(value)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateSizeParameter(value, 'DilationFactor'));
end

function iAssertValidPaddingSize(value)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validatePaddingSize(value));
end

function rowVectorOfTwo = iMakeIntoRowVectorOfTwo(scalarOrRowVectorOfTwo)
rowVectorOfTwo = ...
    nnet.internal.cnn.layer.paramvalidation.makeIntoRowVectorOfTwo(scalarOrRowVectorOfTwo);
end

function paddingSize = iCalculatePaddingSize(padding)
paddingSize = nnet.internal.cnn.layer.padding.calculatePaddingSize(padding);
end

function iEvalAndThrow(func)
% Omit the stack containing internal functions by throwing as caller
try
    func();
catch exception
    throwAsCaller(exception)
end
end
