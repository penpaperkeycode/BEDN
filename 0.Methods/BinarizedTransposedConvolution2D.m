classdef BinarizedTransposedConvolution2D < nnet.internal.cnn.layer.Layer
    % TransposedConvolution2D   Implementation of the 2D transposed convolution layer
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        % (Vector of nnet.internal.cnn.layer.learnable.PredictionLearnableParameter)
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        
        % Name (char array)   A name for the layer
        Name
        
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'transposed-conv'
    end
    
    properties (SetAccess = private)
        % InputNames   This layer has a single input
        InputNames = {'in'}
        
        % OutputNames   This layer has a single output
        OutputNames = {'out'}
        
        % HasSizeDetermined   True for layers with size determined.
        HasSizeDetermined
        
        % Hyperparameters
        
        % FilterSize  (1x2 int vector)  Size of each filter expressed in
        % height x width
        FilterSize
        
        % NumChannels (int)   The number of channels that the input to the
        % layer will have. [] if it has to be inferred later
        NumChannels
        
        % NumFilters (int)  The number of filters in the layer
        NumFilters
        
        % Stride (vector of int) Stride for each dimension
        Stride
        
        % Padding (vector of int) Padding for each dimension
        Padding
    end
    
    properties(Access = private)
        ExecutionStrategy
    end
    
    properties (Dependent)
        % Learnable Parameters (nnet.internal.cnn.layer.LearnableParameter)
        Weights
        Bias
    end
    
    properties (Constant, Access = private)
        % WeightsIndex  Index of the Weights into the LearnableParameter
        %               vector
        WeightsIndex = 1;
        
        % BiasIndex     Index of the Bias into the LearnableParameter
        %               vector
        BiasIndex = 2;
    end
    
    methods
        function this = BinarizedTransposedConvolution2D(name, filterSize, numChannels, ...
                numFilters, stride, padding)
            % TransposedConvolution2D   Constructor for a TransposedConvolution2D layer
            %
            %   Create a 2D transposed convolutional layer with the
            %   following compulsory parameters:
            %
            %       name            - Name for the layer
            %       filterSize      - Size of the filters [height x width]
            %       numChannels     - The number of channels that the input
            %                       to the layer will have. [] if it has to
            %                       be determined later
            %       numFilters      - The number of filters in the layer
            %       stride          - A vector specifying the stride for
            %                       each dimension [height x width]
            %       padding         - A vector specifying the padding for
            %                       each dimension [height x width]
            
            this.Name = name;
            
            % Set Hyperparameters
            this.FilterSize = filterSize;
            this.NumChannels = numChannels;
            this.HasSizeDetermined = ~isempty( numChannels );
            this.NumFilters = numFilters;
            this.Stride = stride;
            this.Padding = padding;
            
            % Set weights and bias to be LearnableParameter
            this.Weights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            this.Bias = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            
            this.ExecutionStrategy = BinarizedTransposedConvolution2DHostStrategy();
        end
        
        function Z = predict( this, X )
            % TODO g1530578 add asymmetric padding support.
            Z = this.ExecutionStrategy.forward(X, ...
                this.Weights.Value, ...
                this.Bias.Value, ...
                this.Padding(1), this.Padding(2), ...
                this.Stride(1), this.Stride(2));
        end
        
        function varargout = backward( this, X, ~, dZ, ~ )
            % backward    Back propagate the derivative of the loss function
            % thru the layer
            
            % TODO g1530578 asymmetric padding.
            [varargout{1:nargout}] = this.ExecutionStrategy.backward( ...
                X, this.Weights.Value, dZ, ...
                this.Padding(1), this.Padding(2), ...
                this.Stride(1), this.Stride(2) ...
                );
            
        end
        
        function this = inferSize(this, inputSize)
            % inferSize     Infer the number of channels based on the input size
            numChannels = iNumChannelsFromInputSize(inputSize);
            
            this.NumChannels = numChannels;
            
            this.HasSizeDetermined = true;
        end
        
        function tf = isValidInputSize(this, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size.
            
            tf = ~this.HasSizeDetermined || inputSize(3) == this.NumChannels;
            
            % The input size must produce an output size that is at least
            % as big as the filter size (otherwise backward will fail).
            outputSize = this.forwardPropagateSize(inputSize);
            
            tf = tf && all(outputSize(1:2) >= this.FilterSize);
            
        end
        
        function outputSize = forwardPropagateSize(this, inputSize)
            % forwardPropagateSize    Output the size of the layer based on
            % the input size
            
            outputHeightAndWidth = this.Stride .* (inputSize(1:2) - 1) + this.FilterSize - 2*this.Padding;
            
            if(this.HasSizeDetermined)
                outputSize = [outputHeightAndWidth this.NumFilters];
            else
                outputSize = [outputHeightAndWidth NaN];
            end
        end
        
        function this = initializeLearnableParameters(this, precision)
            % initializeLearnableParameters    Initialize learnable
            % parameters using their initializer
            
            if isempty(this.Weights.Value)
                % Initialize only if it is empty
                % Conv transpose requires to swap channels with num filters.
                weightsSize = [this.FilterSize, this.NumFilters, this.NumChannels];
                this.LearnableParameters(this.WeightsIndex).Value = iInitializeGaussian( weightsSize, precision );
            else
                % Cast to desired precision
                this.Weights.Value = precision.cast(this.Weights.Value);
            end
            
            if isempty(this.Bias.Value)
                % Initialize only if it is empty
                biasSize = [1, 1, this.NumFilters];
                this.LearnableParameters(this.BiasIndex).Value = iInitializeConstant( biasSize, precision );
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
            this.ExecutionStrategy = BinarizedTransposedConvolution2DHostStrategy();
            this.LearnableParameters(1).UseGPU = false;
            this.LearnableParameters(2).UseGPU = false;
        end
        
        function this = setupForGPUPrediction(this)
            this.ExecutionStrategy = BinarizedTransposedConvolution2DGPUStrategy();
            this.LearnableParameters(1).UseGPU = true;
            this.LearnableParameters(2).UseGPU = true;
        end
        
        function this = setupForHostTraining(this)
            this.ExecutionStrategy = BinarizedTransposedConvolution2DHostStrategy();
        end
        
        function this = setupForGPUTraining(this)
            this.ExecutionStrategy = BinarizedTransposedConvolution2DGPUStrategy();
        end
        
        % Setter and getter for Weights and Bias
        % These make easier to address into the vector of LearnableParameters
        % giving a name to each index of the vector
        function weights = get.Weights(this)
            weights = this.LearnableParameters(this.WeightsIndex);
            weights.Value=max(-1,min(1,weights.Value)); %%%%%%%%%%%%%%%%%%%%%%%%%
        end
        
        function this = set.Weights(this, weights)
            this.LearnableParameters(this.WeightsIndex) = weights;
        end
        
        function bias = get.Bias(this)
            bias = this.LearnableParameters(this.BiasIndex);
        end
        
        function this = set.Bias(this, bias)
            this.LearnableParameters(this.BiasIndex) = bias;
        end
    end
    
    methods(Static)
        function sz = outputSize(X, weights, ...
                verticalPad, horizontalPad, ...
                verticalStride, horizontalStride)
            
            % Return a 4-D array size for the output of transposed conv.
            FH = size(weights,1);
            FW = size(weights,2);
            
            H = verticalStride   * (size(X,1) - 1) + FH - 2*verticalPad;
            W = horizontalStride * (size(X,2) - 1) + FW - 2*horizontalPad;
            C = size(weights,3);
            N = size(X,4);
            
            sz = [H W C N];
            
        end
    end
end

function parameter = iInitializeGaussian(parameterSize, precision)
parameter = precision.cast( ...
    iNormRnd(0, 0.0000001, parameterSize) );   %%%%%
end

function parameter = iInitializeConstant(parameterSize, precision)
parameter = precision.cast( ...
    zeros(parameterSize) );
end

function out = iNormRnd(mu, sigma, outputSize)
% iNormRnd  Returns an array of size 'outputSize' chosen from a
% normal distribution with mean 'mu' and standard deviation 'sigma'
out = randn(outputSize) .* sigma + mu;
end

function numChannels = iNumChannelsFromInputSize(inputSize)
% iNumChannelsFromInputSize   The number of channels is the third element
% in inputSize. If inputSize doesn't have three elements, then it is
% implicitly 1.
if numel(inputSize)<3
    numChannels = 1;
else
    numChannels = inputSize(3);
end
end
