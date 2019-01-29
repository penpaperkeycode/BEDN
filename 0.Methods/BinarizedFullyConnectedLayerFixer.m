classdef BinarizedFullyConnectedLayerFixer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % BinarizedFullyConnectedLayerFixer   Fully connected layer
    %
    %   To create a fully connected layer, use BinarizedFullyConnectedLayerFixer
    %
    %   A fully connected layer. This layer has weight and bias parameters
    %   that are learned during training.
    %
    %   BinarizedFullyConnectedLayerFixer properties:
    %       Name                        - A name for the layer.
    %       InputSize                   - The input size of the fully
    %                                     connected layer.
    %       OutputSize                  - The output size of the fully
    %                                     connected layer.
    %       Weights                     - The weight matrix.
    %       Bias                        - The bias vector.
    %       WeightLearnRateFactor       - The learning rate factor for the
    %                                     weights.
    %       WeightL2Factor              - The L2 regularization factor for
    %                                     the weights.
    %       BiasLearnRateFactor         - The learning rate factor for the
    %                                     bias.
    %       BiasL2Factor                - The L2 regularization factor for
    %                                     the bias.
    %
    %   Example:
    %       Create a fully connected layer with an output size of 10, and an
    %       input size that will be determined at training time.
    %
    %       layer = BinarizedFullyConnectedLayerFixer(10);
    %
    %   See also BinarizedFullyConnectedLayerFixer
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name
    end
    
    properties(SetAccess = private, Dependent)
        % InputSize   The input size for the layer
        %   The input size for the fully connected layer. If this is set to
        %   'auto', then the input size will be automatically set at
        %   training time.
        InputSize
        
        % OutputSize   The output size for the layer
        %   The output size for the fully connected layer.
        OutputSize
    end
    
    properties(Dependent)
        % Weights   The weights for the layer
        %   The weight matrix for the fully connected layer. This matrix
        %   will have size OutputSize-by-InputSize.
        Weights
        
        % Bias   The biases for the layer
        %   The bias vector for the fully connected layer. This vector will
        %   have size OutputSize-by-1.
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
        function this = BinarizedFullyConnectedLayerFixer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            privateLayer = this.PrivateLayer;
            out.Version = 2.0;
            out.Name = privateLayer.Name;
            out.InputSize = privateLayer.InputSize;
            out.OutputSize = [1 1 privateLayer.NumNeurons];
            out.Weights = toStruct(privateLayer.Weights);
            out.Bias = toStruct(privateLayer.Bias);
        end
        
        function val = get.InputSize(this)
            if ~isempty(this.PrivateLayer.InputSize)
                % Get the input size from the internal 4-D input size.
                val = prod(this.PrivateLayer.InputSize);
            elseif ~isempty(this.PrivateLayer.Weights.Value)
                % If the weights have been set externally as 2-D matrix
                % the user visible size is available. The internal size
                % will be determined when the weights will be reshaped
                % to 4-D.
                val = size(this.PrivateLayer.Weights.Value, 2);
            else
                val = 'auto';
            end
        end
        
        function val = get.OutputSize(this)
            val = this.PrivateLayer.NumNeurons;
        end
        
        function weights = get.Weights(this)
            privateWeights = this.PrivateLayer.Weights.HostValue;
            
            if isempty(privateWeights)
                % If no weights have been defined, return "empty" for
                % weights
                weights = [];
                
            elseif ismatrix(privateWeights)
                % If the weights are in a 2d matrix, then they can just be
                % returned as is
                weights = privateWeights;
                
            else % Default case: 4d array
                % In case the internal weights are 4-D we need to reshape
                % them to 2-D.
                weights = reshape(privateWeights, [], this.OutputSize);
                weights = weights';
            end
        end
        
        function this = set.Weights(this, value)
            classes = {'single', 'double', 'gpuArray'};
            if ~isequal(this.InputSize, 'auto')
                expectedInputSize = prod(this.InputSize);
            else
                expectedInputSize = NaN;
            end
            attributes = {'size', [this.OutputSize expectedInputSize], 'nonempty', 'real', 'nonsparse'};
            validateattributes(value, classes, attributes);
            
            this.PrivateLayer.Weights.Value = gather(value);
        end
        
        function val = get.Bias(this)
            val = this.PrivateLayer.Bias.HostValue;
            if(~isempty(val))
                val = reshape(val, this.OutputSize, 1);
            end
        end
        
        function this = set.Bias(this, value)
            classes = {'single', 'double', 'gpuArray'};
            attributes = {'column', 'nonempty', 'real', 'nonsparse', 'nrows', this.OutputSize};
            validateattributes(value, classes, attributes);
            
            this.PrivateLayer.Bias.Value = gather(value);
        end
        
        function val = get.WeightLearnRateFactor(this)
            val = this.PrivateLayer.Weights.LearnRateFactor;
        end
        
        function this = set.WeightLearnRateFactor(this, value)
            iAssertValidFactor(value,'WeightLearnRateFactor');
            this.PrivateLayer.Weights.LearnRateFactor = value;
        end
        
        function val = get.WeightL2Factor(this)
            val = this.PrivateLayer.Weights.L2Factor;
        end
        
        function this = set.WeightL2Factor(this, value)
            iAssertValidFactor(value,'WeightL2Factor');
            this.PrivateLayer.Weights.L2Factor = value;
        end
        
        function val = get.BiasLearnRateFactor(this)
            val = this.PrivateLayer.Bias.LearnRateFactor;
        end
        
        function this = set.BiasLearnRateFactor(this, value)
            iAssertValidFactor(value,'BiasLearnRateFactor');
            this.PrivateLayer.Bias.LearnRateFactor = value;
        end
        
        function val = get.BiasL2Factor(this)
            val = this.PrivateLayer.Bias.L2Factor;
        end
        
        function this = set.BiasL2Factor(this, value)
            iAssertValidFactor(value,'BiasL2Factor');
            this.PrivateLayer.Bias.L2Factor = value;
        end
        
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function this = set.Name(this, val)
            iAssertValidLayerName(val);
            this.PrivateLayer.Name = char(val);
        end
    end
    
    methods(Hidden, Static)
        function this = loadobj(in)
            if in.Version <= 1
                in = iUpgradeVersionOneToVersionTwo(in);
            end
            this = iLoadFullyConnectedLayerFromCurrentVersion(in);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(this)
            outputSizeString = int2str( this.OutputSize );
            
            description = iGetMessageString(  ...
                'nnet_cnn:layer:FullyConnectedLayer:oneLineDisplay', ...
                outputSizeString );
            
            type = iGetMessageString( 'nnet_cnn:layer:FullyConnectedLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            hyperparameters = {
                'InputSize'
                'OutputSize'
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

function S = iUpgradeVersionOneToVersionTwo(S)
% iUpgradeVersionOneToVersionTwo   Upgrade a v1 (2016a) saved struct to a v2 saved struct
%   This means gathering the bias and weights from the GPU and putting them
%   on the host.

S.Version = 2;
try
    S.Bias.Value = gather(S.Bias.Value);
    S.Weights.Value = gather(S.Weights.Value);
catch e
    % Only throw the error we want to throw.
    e = MException( ...
        'nnet_cnn:layer:FullyConnectedLayer:MustHaveGPUToLoadFrom2016a', ...
        getString(message('nnet_cnn:layer:FullyConnectedLayer:MustHaveGPUToLoadFrom2016a')));
    throwAsCaller(e);
end
end

function obj = iLoadFullyConnectedLayerFromCurrentVersion(in)
if ~isempty(in.OutputSize)
    % Remove the first two singleton dimensions of the Outputsize to construct the internal layer.
    in.OutputSize = in.OutputSize(3);
end
internalLayer = BinarizedFullyConnected( ...
    in.Name, in.InputSize, in.OutputSize);
internalLayer.Weights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.Weights);
internalLayer.Bias = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.Bias);

obj = BinarizedFullyConnectedLayerFixer(internalLayer);
end

function iAssertValidFactor(value,factorName)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateLearnFactor(value,factorName));
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