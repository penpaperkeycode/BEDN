classdef BinarizedFullyConnected < nnet.internal.cnn.layer.Layer
    % BinarizedFullyConnected   Implementation of the fully connected layer
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties
        % LearnableParameters   The learnable parameters for this layer
        %   This layer has two learnable parameters, which are the weights
        %   and the bias.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty()
        
        % Name (char array)   A name for the layer
        Name
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'fc'
    end
    
    properties(SetAccess = private)
        % InputNames   This layer has a single input
        InputNames = {'in'}
        
        % OutputNames   This layer has a single output
        OutputNames = {'out'}
        
        % InputSize   Input size of the layer
        %   The input size of the fully connected layer. Note that the
        %   internal layers deal with 3D observations, and so this input
        %   size will be 3D. This will be empty if the input size has not
        %   been set yet.
        InputSize
        
        % NumNeurons  (scalar integer)   Number of neurons for the layer
        NumNeurons
        
        % Execution stategy   Execution stategy of the layer
        %   The execution strategy determines where (host/GPU) and how
        %   forward and backward operations are performed.
        ExecutionStrategy
    end
    
    properties (Dependent)
        % Weights   The weights for the layer
        Weights
        
        % Bias   The bias vector for the layer
        Bias
    end
    
    properties (Dependent, SetAccess = private)
        % HasSizeDetermined   Specifies if all size parameters are set
        %   If the input size has not been determined, then this will be
        %   set to false, otherwise it will be true.
        HasSizeDetermined
    end
    
    properties (Constant, Access = private)
        % WeightsIndex   Index of the Weights in the LearnableParameter vector
        WeightsIndex = 1;
        
        % BiasIndex   Index of the Bias in the LearnableParameter vector
        BiasIndex = 2;
    end
    
    methods
        function this = BinarizedFullyConnected(name, inputSize, numNeurons)
            this.Name = name;
            
            % Set hyperparameters
            this.NumNeurons = numNeurons;
            this.InputSize = inputSize;
            
            % Set learnable parameters
            this.Weights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            this.Bias = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            
            % Set execution strategy
            this.ExecutionStrategy = this.getHostStrategy();
        end
        
        function Z = predict(this, X)
            Z = this.ExecutionStrategy.forward( ...
                X, this.Weights.Value, this.Bias.Value);
        end
        
        function varargout = backward(this, X, ~, dZ, ~)
            [varargout{1:nargout}] = this.ExecutionStrategy.backward(X, this.Weights.Value, dZ);
        end
        
        function this = inferSize(this, inputSize)
            if ~this.HasSizeDetermined
                this.InputSize = inputSize;
                % Match weights and bias to layer size
                weights = this.matchWeightsToLayerSize(this.Weights);
                this.LearnableParameters(this.WeightsIndex) = weights;
                bias = this.matchBiasToLayerSize(this.Bias);
                this.LearnableParameters(this.BiasIndex) = bias;
                % Set execution strategy
                this.ExecutionStrategy = this.getHostStrategy();
            end
        end
        
        function outputSize = forwardPropagateSize(this, inputSize)
            if ~this.HasSizeDetermined
                error(message('nnet_cnn:internal:cnn:layer:FullyConnected:ForwardPropagateSizeWithNoInputSize'));
            else
                if numel(this.InputSize) ~= 1
                    filterSize = this.InputSize(1:2);
                    outputHeightAndWidth = floor(inputSize(1:2) - filterSize) + 1;
                    outputSize = [ outputHeightAndWidth this.NumNeurons ];
                else
                    outputSize = this.NumNeurons;
                end
            end
        end
        
        function tf = isValidInputSize(this, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size
            tf = ( ~this.HasSizeDetermined && (numel(inputSize) == 3 || numel(inputSize) == 1) ...
                || isequal(this.InputSize, inputSize) );
        end
        
        function this = initializeLearnableParameters(this, precision)
            % Initialize weights
            if isempty(this.Weights.Value)
                % Initialize only if it is empty
                weightsSize = [this.InputSize this.NumNeurons];
                this.Weights.Value = iInitializeGaussian( weightsSize, precision );
            else
                % Cast to desired precision
                this.Weights.Value = precision.cast(this.Weights.Value);
            end
            
            % Initialize bias
            if isempty(this.Bias.Value)
                % Initialize only if it is empty
                biasSize = [1 1 this.NumNeurons];
                this.Bias.Value = iInitializeConstant( biasSize, precision );
            else
                % Cast to desired precision
                this.Bias.Value = precision.cast(this.Bias.Value);
            end
        end
        
        function this = prepareForTraining(this)
            % prepareForTraining   Prepare this layer for training
            %   Before this layer can be used for training, we need to
            %   convert the learnable parameters to use the class
            %   TrainingLearnableParameter.
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.convert2training(this.LearnableParameters);
        end
        
        function this = prepareForPrediction(this)
            % prepareForPrediction   Prepare this layer for prediction
            %   Before this layer can be used for prediction, we need to
            %   convert the learnable parameters to use the class
            %   PredictionLearnableParameter.
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.convert2prediction(this.LearnableParameters);
        end
        
        function this = setupForHostPrediction(this)
            this.ExecutionStrategy = this.getHostStrategy();
            this.LearnableParameters(1).UseGPU = false;
            this.LearnableParameters(2).UseGPU = false;
        end
        
        function this = setupForGPUPrediction(this)
            this.ExecutionStrategy = this.getGPUStrategy();
            this.LearnableParameters(1).UseGPU = true;
            this.LearnableParameters(2).UseGPU = true;
        end
        
        function this = setupForHostTraining(this)
            this.ExecutionStrategy = this.getHostStrategy();
        end
        
        function this = setupForGPUTraining(this)
            this.ExecutionStrategy = this.getGPUStrategy();
        end
        
        function weights = get.Weights(this)
            weights = this.LearnableParameters(this.WeightsIndex);
            weights.Value=max(-1,min(1,weights.Value)); %%%%%%%%%%%%%%%%%%%%%%%%%
        end
        
        function this = set.Weights(this, weights)
            if this.HasSizeDetermined
                weights = this.matchWeightsToLayerSize(weights);
            end
            this.LearnableParameters(this.WeightsIndex) = weights;
        end
        
        function bias = get.Bias(this)
            bias = this.LearnableParameters(this.BiasIndex);
        end
        
        function this = set.Bias(this, bias)
            if this.HasSizeDetermined
                bias = this.matchBiasToLayerSize(bias);
            end
            this.LearnableParameters(this.BiasIndex) = bias;
        end
        
        function tf = get.HasSizeDetermined(this)
            tf = ~isempty( this.InputSize );
        end
    end
    
    methods (Access = private)
        function executionStrategy = getHostStrategy(this)
            if ~isscalar(this.InputSize)
                executionStrategy = BinarizedFullyConnectedHostImageStrategy();
            else
                executionStrategy = nnet.internal.cnn.layer.util.FullyConnectedHostVectorStrategy();
            end
        end
        
        function executionStrategy = getGPUStrategy(this)
            if ~isscalar(this.InputSize)
                executionStrategy = BinarizedFullyConnectedGPUImageStrategy();
            else
                executionStrategy = nnet.internal.cnn.layer.util.FullyConnectedGPUVectorStrategy();
            end
        end
        
        function weightsParameter = matchWeightsToLayerSize(this, weightsParameter)
            % matchWeightsToLayerSize    Reshape weights from a matrix into
            % a 4-D array.
            weights = weightsParameter.Value;
            if numel(this.InputSize) == 3
                requiredSize = [this.InputSize this.NumNeurons];
                if isequal( i4DSize( weights ), requiredSize )
                    % Weights are the right size -- nothing to do
                elseif isempty( weights )
                    % Weights are empty -- nothing we can do
                elseif ismatrix( weights )
                    % Weights are 2D -- need to transpose and reshape to 4D
                    % Transpose is needed since the user would set
                    % it as [output input] instead of [input output].
                    weights = reshape( weights', requiredSize );
                else
                    % There are three possibilities and this is a fourth state
                    % therefore something has gone wrong
                    warning( message('nnet_cnn:internal:cnn:layer:FullyConnected:InvalidState') );
                end
            elseif isscalar(this.InputSize)
                requiredSize = [this.NumNeurons this.InputSize];
                currentSize = size(weights);
                if isequal( currentSize, requiredSize )
                    % Weights are the right size -- nothing to do
                elseif isempty( weights )
                    % Weights are empty -- nothing we can do
                elseif isequal( currentSize, fliplr(requiredSize) )
                    % Weights need to be transposed
                    weights = weights';
                elseif isequal( [currentSize(4) prod(currentSize(1:3))] , requiredSize )
                    % Weights are 4D -- need to reshape to 2D, then
                    % transpose
                    weights = reshape( weights, [prod(currentSize(1:3)) currentSize(4)] );
                    weights = weights';
                else
                    % Weights are incorrect size
                    warning( message('nnet_cnn:internal:cnn:layer:FullyConnected:InvalidState') );
                end
            end
            weightsParameter.Value = weights;
        end
        
        function biasParameter = matchBiasToLayerSize(this, biasParameter)
            % matchBiasToLayerSize   Reshape biases from a matrix into a
            % 3-D array.
            bias = biasParameter.Value;
            if numel(this.InputSize) == 3
                requiredSize = [1 1 this.NumNeurons];
                if isequal( i3DSize( bias ), requiredSize )
                    % Biases are the right size -- nothing to do
                elseif isempty( bias )
                    % Biases are empty -- nothing we can do
                elseif ismatrix( bias )
                    % Biases are 2d -- need to reshape to 3d
                    bias = reshape(bias, requiredSize);
                end
            elseif isscalar(this.InputSize)
                requiredSize = [this.NumNeurons 1];
                if isequal( size(bias), requiredSize )
                    % Biases are the right size -- nothing to do
                elseif isempty( bias )
                    % Biases are empty -- nothing we can do
                elseif isequal ( size(bias), fliplr(requiredSize) )
                    % Transpose the bias
                    bias = bias';
                elseif isequal( size(bias), [1 1 this.NumNeurons] )
                    % Biases are 3d -- need to reshape to 2d
                    bias = reshape(bias, requiredSize);
                end
            end
            biasParameter.Value = bias;
        end
    end
end

function parameter = iInitializeGaussian(parameterSize, precision)
parameter = precision.cast( ...
    iNormRnd(0, 0.000000001, parameterSize) );%%%%%
end

function parameter = iInitializeConstant(parameterSize, precision)
parameter = precision.zeros(parameterSize);
end

function out = iNormRnd(mu, sigma, outputSize)
% iNormRnd  Returns an array of size 'outputSize' chosen from a
% normal distribution with mean 'mu' and standard deviation 'sigma'
out = randn(outputSize) .* sigma + mu;
end

function sz = i4DSize(X)
[sz(1), sz(2), sz(3), sz(4)] = size(X);
end

function sz = i3DSize(X)
[sz(1), sz(2), sz(3)] = size(X);
end

