classdef BTransposedConvolution2D < nnet.internal.cnn.layer.FunctionalLayer
    % TransposedConvolution2D   Implementation of the 2D transposed convolution layer
    
    %   Copyright 2017-2019 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        % (Vector of nnet.internal.cnn.layer.learnable.PredictionLearnableParameter)
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        
        % Name (char array)   A name for the layer
        Name
        
        % OutputSizeOffset (1x2 int vector)    A number in [0,stride-1]
        % corresponding to the offset to the output size.
        OutputSizeOffset
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
        
        % CroppingMode (char) Cropping mode: 'manual' or 'same'
        CroppingMode
        
        % CroppingSize (vector of int) Cropping for each dimension
        CroppingSize
    end
    
    properties(Access = private)
        ExecutionStrategy
    end
    
    properties (Dependent)
        % Learnable Parameters (nnet.internal.cnn.layer.LearnableParameter)
        Weights
        Bias
        
        % Learnables    Cell array of learnable parameter values
        Learnables
    end
    
    properties(SetAccess=protected, GetAccess=?nnet.internal.cnn.dlnetwork)
        LearnablesNames = ["Weights" "Bias"]
    end
    
    properties (Constant, Access = private)
        % WeightsIndex  Index of the Weights into the LearnableParameter
        %               vector
        WeightsIndex = 1;
        
        % BiasIndex     Index of the Bias into the LearnableParameter
        %               vector
        BiasIndex = 2;
    end
    
    properties(Dependent, SetAccess=private)
        % Expected Weights size
        ExtWeightsSize
        
        % Expected Bias size
        ExtBiasSize
    end
    
    methods
        function this = BTransposedConvolution2D(params)
            % TransposedConvolution2D   Constructor for
            % TransposedConvolution2D layer
            %
            %   Create a 2D transposed convolutional layer. params has the
            %   following compulsory fields:
            %
            %       Name             - Name for the layer
            %       FilterSize       - Size of the filters [height x width]
            %       NumChannels      - The number of channels that the input
            %                          to the layer will have. [] if it has
            %                          to be determined later
            %       NumFilters       - The number of filters in the layer
            %       Stride           - A vector specifying the stride for
            %                          each dimension [height x width]
            %       CroppinMode      - Cropping mode: 'manual' or 'same'
            %       CroppingSize     - A vector specifying the cropping for
            %                          each dimension [top bottom left
            %                          right]
            %   Optional:
            %       OutputSizeOffset - The offset of the output size.
            %                          Default: [].
            
            if ~isfield(params, 'OutputSizeOffset')
                params.OutputSizeOffset = [];
            end
            
            this.Name = params.Name;
            this.NeedsZForBackward = false;
            
            % Set Hyperparameters
            this.FilterSize = params.FilterSize;
            this.NumChannels = params.NumChannels;
            this.NumFilters = params.NumFilters;
            this.Stride = params.Stride;
            this.CroppingMode = params.CroppingMode;
            if iIsTheStringSame(this.CroppingMode)
                this = this.setCroppingAndOutputSizeOffsetIfSame();
            else
                this.OutputSizeOffset = params.OutputSizeOffset;
                this.CroppingSize = params.CroppingSize;
            end
            this.HasSizeDetermined = ~isempty(this.NumChannels) && ...
                ~isempty(this.OutputSizeOffset);
            
            % Set weights and bias to be LearnableParameter
            this.Weights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            this.Bias = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            % Set default initializers. The external layer constructor
            % overwrites these values, which are set only for internal code
            % that bypasses the casual API.
            this.Weights.Initializer = iInternalInitializer('narrow-normal');
            this.Bias.Initializer = iInternalInitializer('zeros');
            
            this.ExecutionStrategy = BTransposedConvolution2DHostStrategy();
        end
        
        function Z = predict( this, X )
            if isa(this.ExecutionStrategy, ...
                    'nnet.internal.cnn.layer.util.FunctionalStrategy')
                % TODO: This needs to be removed when the we replace the
                % dltranspconv with the internal API that accepts cropping
                % size and output size offset only
                if isequal(this.CroppingMode, 'same')
                    Z = this.ExecutionStrategy.forward(X, ...
                        this.Weights.Value, ...
                        this.Bias.Value, ...
                        this.CroppingMode, ...
                        this.Stride(1), this.Stride(2), ...
                        this.OutputSizeOffset(1), this.OutputSizeOffset(2));
                else
                    Z = this.ExecutionStrategy.forward(X, ...
                        this.Weights.Value, ...
                        this.Bias.Value, ...
                        [this.CroppingSize(1) this.CroppingSize(3); ...     % t l
                        this.CroppingSize(2) this.CroppingSize(4)], ... % b r
                        this.Stride(1), this.Stride(2), ...
                        this.OutputSizeOffset(1), this.OutputSizeOffset(2));
                end
            else
                Z = this.ExecutionStrategy.forward(X, ...
                    this.Weights.Value, ...
                    this.Bias.Value, ...
                    this.CroppingSize(1), this.CroppingSize(3), ... % t l
                    this.CroppingSize(2), this.CroppingSize(4), ... % b r
                    this.Stride(1), this.Stride(2), ...
                    this.OutputSizeOffset(1), this.OutputSizeOffset(2));
            end
        end
        
        function varargout = backward( this, X, ~, dZ, ~ )
            [varargout{1:nargout}] = this.ExecutionStrategy.backward( ...
                X, this.Weights.Value, dZ, ...
                this.CroppingSize(1), this.CroppingSize(3), ... % t l
                this.CroppingSize(2), this.CroppingSize(4), ... % b r
                this.Stride(1), this.Stride(2));
        end
        
        function this = inferNumChannels(this, inputSize)
            numChannels = iNumChannelsFromInputSize(inputSize);
            this.NumChannels = numChannels;
        end
        
        function this = inferSize(this, inputSize)
            % Assume inputSize is 3d array of positive integers.
            this = this.inferNumChannels(inputSize);
            
            if ~iIsTheStringSame(this.CroppingMode)
                totCropping = [this.CroppingSize(1)+this.CroppingSize(2), ...
                    this.CroppingSize(3)+this.CroppingSize(4)];
                % If cropping not same, compute dynamically
                % outputSizeOffset, as the minimum allowed value.
                % This is the minimum value such that
                % 1) 0 <= outputSizeOffset < stride (from definition)
                % 2) The output size of the layer is positive:
                %    stride .* (inputSize(1:2) - 1) + this.FilterSize - ...
                %       totCropping + outputSizeOffset >= 1
                %    We rewrite this as
                %    outputSizeOffset >= minOutputSizeOffset, with:
                minOutputSizeOffset = 1 - this.Stride .* (inputSize(1:2) - 1) - ...
                    this.FilterSize + totCropping;
                if any(minOutputSizeOffset >= this.Stride)
                    % If such value does not exist, error.
                    error(message('nnet_cnn:layer:TransposedConvolution2DLayer:InvalidInputSize'));
                else
                    this.OutputSizeOffset = max(0, minOutputSizeOffset);
                end
            end
            
            this.HasSizeDetermined = true;
        end
        
        function tf = isValidInputSize(this, inputSize)
            % Assumes inferSize has been called. There we computed
            % dynamically the outputSizeOffset so that input of a certain
            % size is valid and we threw error if not. Here check
            % numChannels and recheck valid output size.
            
            cropping = [this.CroppingSize(1)+this.CroppingSize(2), ...
                this.CroppingSize(3)+this.CroppingSize(4)];
            outputHeightAndWidth = iOutputSize(inputSize(1:2), ...
                this.FilterSize, this.Stride, cropping, ...
                this.OutputSizeOffset);
            
            tf = numel(inputSize) == 3 && inputSize(3) == this.NumChannels && all(outputHeightAndWidth > 0);
        end
        
        function outputSize = forwardPropagateSize(this, inputSize)
            % forwardPropagateSize    Output the size of the layer based on
            % the input size. Assumes inputSize is valid.
            
            cropping = [this.CroppingSize(1)+this.CroppingSize(2), ...
                this.CroppingSize(3)+this.CroppingSize(4)];
            outputHeightAndWidth = iOutputSize(inputSize(1:2), ...
                this.FilterSize, this.Stride, cropping, ...
                this.OutputSizeOffset);
            
            outputSize = [outputHeightAndWidth this.NumFilters];
        end
        
        function this = initializeLearnableParameters(this, precision)
            % initializeLearnableParameters    Initialize learnable
            % parameters using their initializer
            
            if isempty(this.Weights.Value)
                % Initialize only if it is empty
                % Conv transpose requires to swap channels with num filters.
                weightsSize = [this.FilterSize, this.NumFilters, this.NumChannels];
                weights = this.Weights.Initializer.initialize(weightsSize, 'Weights');
                this.Weights.Value = precision.cast(weights);
            else
                % Cast to desired precision
                this.Weights.Value = precision.cast(this.Weights.Value);
            end
            
            if isempty(this.Bias.Value)
                % Initialize only if it is empty
                biasSize = [1, 1, this.NumFilters];
                bias = this.Bias.Initializer.initialize(biasSize, 'Bias');
                this.Bias.Value = precision.cast(bias);
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
            this.ExecutionStrategy = BTransposedConvolution2DHostStrategy();
            this.LearnableParameters(1).UseGPU = false;
            this.LearnableParameters(2).UseGPU = false;
        end
        
        function this = setupForGPUPrediction(this)
            this.ExecutionStrategy = BTransposedConvolution2DGPUStrategy();
            this.LearnableParameters(1).UseGPU = true;
            this.LearnableParameters(2).UseGPU = true;
        end
        
        function this = setupForHostTraining(this)
            this.ExecutionStrategy = BTransposedConvolution2DHostStrategy();
        end
        
        function this = setupForGPUTraining(this)
            this.ExecutionStrategy = BTransposedConvolution2DGPUStrategy();
        end
        
        % Setter and getter for Weights and Bias
        % These make easier to address into the vector of LearnableParameters
        % giving a name to each index of the vector
        function weights = get.Weights(this)
            weights = this.LearnableParameters(this.WeightsIndex);
            weights.Value(weights.Value>1)=1; %%%%%%%%%%%%%%%%%%%%%%%%%
            weights.Value(weights.Value<-1)=-1; %%%%%%%%%%%%%%%%%%%%%%%%%
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
        
        function learnables = get.Learnables(this)
            % Assume setupForFunctional has been called
            w = this.Weights.Value;
            b = this.Bias.Value;
            learnables = {w, b};
        end
        
        function this = set.Learnables(this, learnables)
            % Assume setupForFunctional has been called
            nnet.internal.cnn.layer.paramvalidation.assertValidLearnables(learnables{1}, this.ExtWeightsSize);
            nnet.internal.cnn.layer.paramvalidation.assertValidLearnables(learnables{2}, this.ExtBiasSize);
            
            this.LearnableParameters(this.WeightsIndex).Value = learnables{1};
            this.LearnableParameters(this.BiasIndex).Value = learnables{2};
        end
        
        function sz = get.ExtWeightsSize(this)
            expectedNumChannels = iExpectedNumChannels(this.NumChannels);
            sz = [this.FilterSize this.NumFilters expectedNumChannels];
        end
        
        function sz = get.ExtBiasSize(this)
            sz = [1 1 this.NumFilters];
        end
    end
    
    methods(Access=protected)
        function this = setFunctionalStrategy(this)
            this.ExecutionStrategy = ...
                BTransposedConvolution2DFunctionalStrategy();
        end
    end
    
    methods(Access = private)
        function this = setCroppingAndOutputSizeOffsetIfSame(this)
            % Set cropping size and outputSizeOffset given stride and
            % filter size in such a way that
            % 1) The outputSize is stride*inputSize
            % 2) Cropping is minimum
            totalCropping = this.FilterSize - this.Stride;
            this.OutputSizeOffset = max(0, -totalCropping);
            totalCropping = totalCropping + this.OutputSizeOffset;
            
            % Height
            t = floor(1/2*(totalCropping(1)));
            b = ceil(1/2*(totalCropping(1)));
            
            % Width
            l = floor(1/2*(totalCropping(2)));
            r = ceil(1/2*(totalCropping(2)));
            
            this.CroppingSize = [t b l r];
        end
    end
    
    methods(Static)
        function sz = outputSize(X, weights, ...
                topPad, leftPad, bottomPad, rightPad, ...
                verticalStride, horizontalStride, ...
                verticalOutputSizeOffset, horizontalOutputSizeOffset)
            
            % Return a 4-D array size for the output of transposed conv.
            FH = size(weights,1);
            FW = size(weights,2);
            
            H = iOutputSize(size(X,1), FH, verticalStride, ...
                topPad+bottomPad, verticalOutputSizeOffset);
            W = iOutputSize(size(X,2), FW, horizontalStride, ...
                leftPad+rightPad, horizontalOutputSizeOffset);
            C = size(weights,3);
            N = size(X,4);
            
            sz = [H W C N];
            
        end
    end
end

function outputSize = iOutputSize(inputSize, filterSize, stride, ...
    totalCropping, outputSizeOffset)
% Formula for the output size. Works for arguments arrays of arbitrary dim.
assert(~isempty(totalCropping) && ~isempty(outputSizeOffset));
outputSize = stride .* (inputSize - 1) + filterSize - totalCropping + ...
    outputSizeOffset;
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

function tf = iIsTheStringSame(x)
tf = nnet.internal.cnn.layer.padding.isTheStringSame(x);
end

function initializer = iInternalInitializer(name)
initializer = nnet.internal.cnn.layer.learnable.initializer...
    .initializerFactory(name);
end

function expectedNumChannels = iExpectedNumChannels(NumChannels)
expectedNumChannels = NumChannels;
if isempty(expectedNumChannels)
    expectedNumChannels = NaN;
end
end
