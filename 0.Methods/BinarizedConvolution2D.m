classdef BinarizedConvolution2D < nnet.internal.cnn.layer.FunctionalLayer ...
        & nnet.internal.cnn.layer.CPUFusableLayer
    % Convolution2D   Implementation of the 2D convolution layer
    
    %   Copyright 2015-2019 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        % (Vector of nnet.internal.cnn.layer.learnable.PredictionLearnableParameter)
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        
        % Name (char array)   A name for the layer
        Name
        
        % Stride (vector of int) Stride for each dimension
        Stride
        
        % DilationFactor (vector of int) Dilation factor for each dimension
        DilationFactor
        
        % PaddingSize   The padding applied to the input along the edges
        %   The padding that is applied along the edges. This is a row
        %   vector [t b l r] where t is the padding to the top, b is the
        %   padding applied to the bottom, l is the padding applied to the
        %   left, and r is the padding applied to the right.
        PaddingSize
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'conv'
    end
    
    properties (SetAccess = private)
        % InputNames   This layer has a single input
        InputNames = {'in'}
        
        % OutputNames   This layer has a single output
        OutputNames = {'out'}
        
        % HasSizeDetermined   True for layers with size determined.
        HasSizeDetermined
        
        % Hyper-parameters
        
        % FilterSize  (1x2 int vector)  Size of each filter expressed in
        % height x width
        FilterSize
        
        % NumChannels (int)   The number of channels that the input to the
        % layer will have. [] if it has to be inferred later
        NumChannels
        
        % NumFilters (int)  The number of filters in the layer
        NumFilters
        
        % PaddingMode   The mode used to determine the padding
        %   The mode used to calculate the PaddingSize property. This can
        %   be:
        %       'manual'    - PaddingSize is specified manually.
        %       'same'      - PaddingSize will be calculated so that the
        %                     output size is the same size as the input
        %                     when the stride is 1. More generally, the
        %                     output size will be ceil(inputSize/stride),
        %                     where inputSize is the height and width of
        %                     the input.
        PaddingMode
    end
    
    properties(Access = public)
        
        % QuantizationMethod Quantization method for learnable parameters
        QuantizationMethod
    end
    
    properties(Access = private)
        ExecutionStrategy
        CacheHandle
        IsTraining
    end
    
    properties (Dependent)
        % Learnable Parameters (nnet.internal.cnn.layer.LearnableParameter)
        Weights
        Bias
        Learnables        
    end
    
    properties(SetAccess=protected, GetAccess=?nnet.internal.cnn.dlnetwork)
        LearnablesNames = ["Weights" "Bias"]
    end
    
    properties (Dependent, SetAccess = private)
        % Effective filter size which takes into account dilation
        EffectiveFilterSize
        
        % Expected Weights size
        ExtWeightsSize
        
        % Expected Bias size
        ExtBiasSize
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
        function this = BinarizedConvolution2D( ...
                name, filterSize, numChannels, numFilters, stride, dilationFactor, paddingMode, paddingSize)
            % Convolution2D   Constructor for a Convolution2D layer
            %
            %   Create a 2D convolutional layer with the following
            %   compulsory parameters:
            %
            %       name            - Name for the layer
            %       filterSize      - Size of the filters [height x width]
            %       numChannels     - The number of channels that the input
            %                       to the layer will have. [] if it has to
            %                       be determined later
            %       numFilters      - The number of filters in the layer
            %       dilationFactor  - A vector specifying the dilation factor for
            %                       each dimension [height width]
            %       stride          - A vector specifying the stride for
            %                       each dimension [height width]
            %       paddingMode     - A string, 'manual' or 'same'.
            %       paddingSize     - A vector specifying the padding for
            %                       each dimension [top bottom left right]
            
            this.Name = name;
            
            % Set Hyper-parameters
            this.FilterSize = filterSize;
            this.NumChannels = numChannels;
            this.HasSizeDetermined = ~isempty( numChannels ) && ~iIsTheStringSame(paddingMode);
            this.NumFilters = numFilters;
            this.Stride = stride;
            this.DilationFactor = dilationFactor;
            this.PaddingMode = paddingMode;
            this.PaddingSize = paddingSize;
            
            % Set weights and bias to be LearnableParameter
            this.CacheHandle = nnet.internal.cnn.layer.learnable.CacheHandle();
            this.Weights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            this.Bias = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            % Set default initializers. convolution2dLayer overwrites these
            % values, which are set only for internal code that bypasses
            % convolution2dLayer.
            this.Weights.Initializer = iInternalInitializer('narrow-normal');
            this.Bias.Initializer = iInternalInitializer('zeros');
            
            this = this.setHostStrategy();
            this.IsTraining = false;
            
            % Initialize QuantizationMethod
            this.QuantizationMethod = nnet.internal.cnn.layer.NoQuantization;
            
            % Convolution2D layer needs X but not Z for the backward pass
            this.NeedsXForBackward = true;
            this.NeedsZForBackward = false;
        end
        
        function Z = predict( this, X )
            % predict   Forward input data through the layer and output the result
            if(this.usingFilterGroups())
                Z = this.predictTwoFilterGroupsWithCaching( X );
            else
                Z = this.predictNormal( X );
            end
        end
        
        function [Z, memory] = forward( this, X )
            % forward   Forward propagate data during training
            memory = [];
            if(this.usingFilterGroups())
                Z = this.forwardTwoFilterGroups( X );
            else
                Z = this.forwardNormal( X );
            end
        end
        
        function varargout = backward( this, X, ~, dZ, ~ )
            % backward    Back propagate the derivative of the loss function
            % through the layer
            if(this.usingFilterGroups())
                [varargout{1:nargout}] = this.backwardTwoFilterGroups(X, [], dZ, []);
            else
                [varargout{1:nargout}] = this.backwardNormal(X, [], dZ, []);
            end
        end
        
        function this = inferSize(this, inputSize)
            % inferSize     Infer the number of channels and padding if
            % same based on the input size
            if isempty(this.Weights.Value)
                % Reset NumChannels only if empty weights. This assumes
                % that if weights are set, NumChannels is set correctly.
                numChannels = iNumChannelsFromInputSize(inputSize);
                if this.usingFilterGroups()
                    % For filter groups, the number of channels is half the
                    % size
                    numChannels = numChannels/2;
                end
                this.NumChannels = numChannels;
            end
            
            if iIsTheStringSame(this.PaddingMode)
                this.PaddingSize = iCalculateSamePadding( ...
                    this.EffectiveFilterSize, this.Stride, inputSize(1:2));
                
                % If the padding is set to 'same', the size will always
                % need to be determined again because we will need to
                % recalculate the padding.
                this.HasSizeDetermined = false;
            else
                this.HasSizeDetermined = true;
            end
        end
        
        function tf = isValidInputSize(this, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size
            tf = numel(inputSize)==3 && ...
                this.isFilterSizeSmallerThanImage( inputSize ) && ...
                this.numFilterChannelsMatchesNumImageChannels( inputSize );
        end
        
        function outputSize = forwardPropagateSize(this, inputSize)
            % forwardPropagateSize    Output the size of the layer based on
            % the input size
            paddingSize = iCalculatePaddingSizeFromInputSize( ...
                this.PaddingMode, this.PaddingSize, this.EffectiveFilterSize, ...
                this.Stride, inputSize(1:2));
            
            heightAndWidthPadding = iCalculateHeightAndWidthPadding(paddingSize);
            outputHeightAndWidth = floor((inputSize(1:2) + ...
                heightAndWidthPadding - this.EffectiveFilterSize)./this.Stride) + 1;
            outputSize = [outputHeightAndWidth sum(this.NumFilters)];
        end
        
        function this = initializeLearnableParameters(this, precision)
            % initializeLearnableParameters    Initialize learnable
            % parameters using their initializer
            
            if isempty(this.Weights.Value)
                % Initialize only if it is empty
                weightsSize = [this.FilterSize, this.NumChannels, sum(this.NumFilters)];
                weights = this.Weights.Initializer.initialize(weightsSize, 'Weights');
                this.Weights.Value = precision.cast(weights);
            else
                % Cast to desired precision
                this.Weights.Value = precision.cast(this.Weights.Value);
            end
            
            if isempty(this.Bias.Value)
                % Initialize only if it is empty
                biasSize = [1, 1, sum(this.NumFilters)];
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
            this.IsTraining = true;
        end
        
        function this = prepareForPrediction(this)
            % prepareForPrediction   Prepare this layer for prediction
            %   Before this layer can be used for prediction, we need to
            %   convert the learnable parameters to use the class
            %   PredictionLearnableParameter.
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.convert2prediction(this.LearnableParameters);
            this.IsTraining = false;
        end
        
        function this = setupForHostPrediction(this)
            this = this.setHostStrategy();
            this.LearnableParameters(1).UseGPU = false;
            this.LearnableParameters(2).UseGPU = false;
            this.CacheHandle.clearCache();
        end
        
        function this = setupForGPUPrediction(this)
            this = this.setGPUStrategy();
            this.LearnableParameters(1).UseGPU = true;
            this.LearnableParameters(2).UseGPU = true;
            this.CacheHandle.clearCache();
        end
        
        function this = setupForHostTraining(this)
            this = this.setHostStrategy();
        end
        
        function this = setupForGPUTraining(this)
            this = this.setGPUStrategy();
        end
        
        % Setter and getter for Weights and Bias
        % These make easier to address into the vector of LearnableParameters
        % giving a name to each index of the vector
        function weights = get.Weights(this)
            weights = this.LearnableParameters(this.WeightsIndex);
%             weights.Value=sign(weights.Value); %%%%%%%%%%%%%%%%%%%%%%%%%  Binarize Weights When training is finished
%             weights.Value(weights.Value==0)=1; %%%%%%%%%%%%%%%%%%%%%%%%%  Binarize Weights When training is finished
            weights.Value(weights.Value>1)=1; %%%%%%%%%%%%%%%%%%%%%%%%% 
            weights.Value(weights.Value<-1)=-1; %%%%%%%%%%%%%%%%%%%%%%%%%  
        end
        
        function this = set.Weights(this, weights)
            this.LearnableParameters(this.WeightsIndex) = weights;
            if ~this.CacheHandle.isEmpty
                this.CacheHandle.clearCache;
            end
        end
        
        function bias = get.Bias(this)
            bias = this.LearnableParameters(this.BiasIndex);
        end
        
        function this = set.Bias(this, bias)
            this.LearnableParameters(this.BiasIndex) = bias;
            if ~this.CacheHandle.isEmpty
                this.CacheHandle.clearCache;
            end
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
            if ~this.CacheHandle.isEmpty
                this.CacheHandle.clearCache;
            end
        end
        
        function dilatedFilterSize = get.EffectiveFilterSize(this)
            % Dilation is equivalent to adding extra zeros in between the
            % elements of the filter so that it leads to the following
            % effective filter size:
            % dilatedFilterSize = filterSize +
            % (filterSize - 1) * (dilationFactor - 1)
            % or, simplifying:
            dilatedFilterSize = (this.FilterSize - 1) .* this.DilationFactor + 1;
        end
        
        function sz = get.ExtWeightsSize(this)
            if(this.usingFilterGroups)
                expectedNumChannels = iExpectedNumChannels(this.NumChannels(1));
            else
                expectedNumChannels = iExpectedNumChannels(this.NumChannels);
            end
            sz = [this.FilterSize expectedNumChannels sum(this.NumFilters)];
        end
        
        function sz = get.ExtBiasSize(this)
            sz = [1 1 sum(this.NumFilters)];
        end
        
        function this = accept(this, visitor)
            % accept  Accept a layer visitor and call its visit method
            %
            %   layer = accept(layer, visitor) requests that the layer accepts a
            %   LayerVisitor object.  Convolution2D calls the visitConvolution2D method
            %   on the visitor.
            
            this = visitConvolution2D(visitor, this);
        end
        
        function this = clearPredictionCache(this)
            % clearPredictionCache  Reset caches that depend on learnable values 
            %
            %   clearPredictionCache(layer) ensures that caches that depend on
            %   learnable parameters are emptied.
            
            this.CacheHandle = nnet.internal.cnn.layer.learnable.CacheHandle();
        end 
    end
    
    methods(Access = private)
        function tf = usingFilterGroups(this)
            tf = numel(this.NumFilters) ~= 1;
        end
        
        function [Z, memory] = forwardNormal( this, X )
            Z = this.doForward(X,this.Weights.Value,this.Bias.Value);
            memory = [];
        end
        
        function [Z, memory] = predictNormal( this, X )
            [weights0,bias0] = this.QuantizationMethod.remapped(this.Weights.Value,this.Bias.Value);
            Z = this.doForward(X,weights0,bias0);
            memory = [];
        end
        
        function [Z, memory] = predictTwoFilterGroupsWithCaching( this, X )
            % predictTwoFilterGroupsWithCaching   Forward propagation for
            % two filter groups with caching
            
            if this.IsTraining
                % This code branch is followed during validation. We do not
                % cache learnable parameters during validation, as the
                % parameters are still being learned and are not fixed.
                [Z, memory] = forwardTwoFilterGroups( this, X );
            else
                % This code branch is followed during inference. The cache
                % is used here to avoid splitting the fixed learnable
                % parameters during repeated calls to predict().
                [X1,X2] = iSplitDataAlongThirdDimension(X, this.NumChannels);
                
                if this.CacheHandle.isEmpty
                    [weights0,bias0] = this.QuantizationMethod.remapped(this.Weights.Value,this.Bias.Value);
                    [weights1, weights2] = iSplitWeightsAlongFourthDimension(weights0, this.NumFilters);
                    [bias1, bias2] = iSplitBiasAlongThirdDimension(bias0, this.NumFilters);
                    split.weights1 = weights1;
                    split.weights2 = weights2;
                    split.bias1 = bias1;
                    split.bias2 = bias2;
                    this.CacheHandle.fillCache(split);
                else
                    split = this.CacheHandle.Value;
                    weights1 = split.weights1;
                    weights2 = split.weights2;
                    bias1 = split.bias1;
                    bias2 = split.bias2;
                end
                % Do forward propagation in two parallel branches
                Z1 = this.doForward(X1,weights1,bias1);
                Z2 = this.doForward(X2,weights2,bias2);
                
                % Stack the results back together again
                Z = cat(3, Z1, Z2);
                memory = [];
            end
        end
        
        function [Z, memory] = forwardTwoFilterGroups( this, X )
            % forwardTwoFilterGroups   Forward propagation for two filter
            % groups
            
            [X1,X2] = iSplitDataAlongThirdDimension(X, this.NumChannels);
            [weights1, weights2] = iSplitWeightsAlongFourthDimension(this.Weights.Value, this.NumFilters);
            [bias1, bias2] = iSplitBiasAlongThirdDimension(this.Bias.Value, this.NumFilters);
            
            % Do forward propagation in two parallel branches
            Z1 = this.doForward(X1,weights1,bias1);
            Z2 = this.doForward(X2,weights2,bias2);
            
            % Stack the results back together again
            Z = cat(3, Z1, Z2);
            memory = [];
        end
        
        function Z = doForward(this, X, weights, bias)
            % Note that padding is stored as [top bottom left right] but
            % the function expects [top left bottom right].
            inputSize = [size(X,1) size(X,2)];
            paddingSize = iCalculatePaddingSizeFromInputSize( ...
                this.PaddingMode, this.PaddingSize, this.EffectiveFilterSize, ...
                this.Stride, inputSize );
            Z = this.ExecutionStrategy.forward(X, weights, bias, ...
                paddingSize(1), paddingSize(3), ...
                paddingSize(2), paddingSize(4), ...
                this.Stride(1), this.Stride(2), ...
                this.DilationFactor(1), this.DilationFactor(2));
        end
        
        function varargout = backwardNormal(this, X, ~, dZ, ~)
            [varargout{1:nargout}] = this.doBackward(X, this.Weights.Value, dZ);
        end
        
        function [dX,dW] = backwardTwoFilterGroups(this, X, ~, dZ, ~)
            % backwardFilterGroup   Backpropagation for two filter groups
            
            [X1, X2] = iSplitDataAlongThirdDimension(X, this.NumChannels);
            
            [weights1, weights2] = iSplitWeightsAlongFourthDimension( ...
                this.Weights.Value, this.NumFilters);
            
            [dZ1, dZ2] = iSplitDerivativeAlongThirdDimension(dZ, this.NumFilters);
            
            % Do backward propagation in two parallel branches
            [d1{1:nargout}] = this.doBackward(X1, weights1, dZ1);
            [d2{1:nargout}] = this.doBackward(X2, weights2, dZ2);
            
            dX = cat(3, d1{1}, d2{1});
            if nargout == 2
                dW = {cat(4, d1{2}{1}, d2{2}{1}), cat(3, d1{2}{2}, d2{2}{2})};
            end
        end
        
        function varargout = doBackward(this, X, weights, dZ)
            % Note that padding is stored as [top bottom left right] but
            % the function expects [top left bottom right].
            inputSize = [size(X,1) size(X,2)];
            paddingSize = iCalculatePaddingSizeFromInputSize( ...
                this.PaddingMode, this.PaddingSize, this.EffectiveFilterSize, ...
                this.Stride, inputSize );
            [varargout{1:nargout}] = this.ExecutionStrategy.backward( ...
                X, weights, dZ, ...
                paddingSize(1), paddingSize(3), ...
                paddingSize(2), paddingSize(4), ...
                this.Stride(1), this.Stride(2), ...
                this.DilationFactor(1), this.DilationFactor(2));
        end
        
        function tf = isFilterSizeSmallerThanImage( this, inputSize )
            % The size of the image is given by the first two dimensions of the input size
            imageSize = inputSize(1:2);
            
            % Need to take padding as well as dilation factor into account when comparing
            % image size and filter size
            heightAndWidthPadding = iCalculateHeightAndWidthPadding(this.PaddingSize);
            tf = all( this.EffectiveFilterSize <= imageSize + heightAndWidthPadding );
        end
        
        function tf = numFilterChannelsMatchesNumImageChannels( this, inputSize )
            numImageChannels = inputSize(3);
            % The total number of channels for the filters must take into
            % account whether filter groups are used
            numGroups = numel(this.NumFilters);
            totalNumFilterChannels = numGroups*this.NumChannels;
            tf = isempty(this.NumChannels) || totalNumFilterChannels == numImageChannels;
        end
        
        function this = setHostStrategy(this)
            % setHostStrategy   Use Mkldnn only if Mkldnn is featured on and no dilation
            noDilation = isequal(this.DilationFactor, [1 1]);
            if nnet.internal.cnnhost.useMKLDNN && noDilation
                this.ExecutionStrategy = BinarizedConvolution2DHostMkldnnStrategy();
            else
                this.ExecutionStrategy = BinarizedConvolution2DHostStridedConvStrategy();
            end
        end
        
        function this = setGPUStrategy(this)
            this.ExecutionStrategy = BinarizedConvolution2DGPUStrategy();
        end
    end
    
    methods(Access=protected)
        function this = setFunctionalStrategy(this)
            this.ExecutionStrategy = BinarizedConvolution2DFunctionalStrategy();
        end
    end
    
    methods (Hidden)
        function layerArgs = getFusedArguments(layer)
            % getFusedArguments  Returned the arguments needed to call the
            % layer in a fused network.
            import nnet.internal.cnn.layer.*
            grps = layer.NumFilters;
            padding = CPUGenericFusedLayer.shufflePadding(layer.PaddingSize);
            b = layer.Bias.Value;
            if ~any(b) && all(isfinite(b))
                b = [];
            end
            layerArgs = { 'conv', layer.Weights.Value, ...
                padding, layer.Stride, layer.DilationFactor, b, grps };
        end
        
        function tf = isFusable(layer, precision, numDataDimensions)
            % isFusable  Indicates if the layer is fusable in a given network.
            tf = (numDataDimensions == 2) && (class(layer.Weights.Value) == precision);
        end
    end
end

function [data1, data2] = iSplitDataAlongThirdDimension(data, numChannels)
data1 = data(:,:,1:numChannels,:);
data2 = data(:,:,numChannels + 1:2*numChannels,:);
end

function [weights1, weights2] = iSplitWeightsAlongFourthDimension(weights, numFilters)
weights1 = weights(:,:,:,1:numFilters(1));
weights2 = weights(:,:,:,numFilters(1)+1:numFilters(1)+numFilters(2));
end

function [bias1, bias2] = iSplitBiasAlongThirdDimension(bias, numFilters)
bias1 = bias(:,:,1:numFilters(1));
bias2 = bias(:,:,numFilters(1)+1:numFilters(1)+numFilters(2));
end

function [dZ1, dZ2] = iSplitDerivativeAlongThirdDimension(dZ, numFilters)
dZ1 = dZ(:,:,1:numFilters(1),:);
dZ2 = dZ(:,:,numFilters(1)+1:numFilters(1)+numFilters(2),:);
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

function expectedNumChannels = iExpectedNumChannels(NumChannels)
expectedNumChannels = NumChannels;
if isempty(expectedNumChannels)
    expectedNumChannels = NaN;
end
end

function tf = iIsTheStringSame(x)
tf = nnet.internal.cnn.layer.padding.isTheStringSame(x);
end

function paddingSize = iCalculatePaddingSizeFromInputSize( ...
    paddingMode, paddingSize, filterOrPoolSize, stride, spatialInputSize)
paddingSize = nnet.internal.cnn.layer.padding.calculatePaddingSizeFromInputSize( ...
    paddingMode, paddingSize, filterOrPoolSize, stride, spatialInputSize);
end

function heightAndWidthPadding = iCalculateHeightAndWidthPadding(paddingSize)
heightAndWidthPadding = nnet.internal.cnn.layer.padding.calculateHeightAndWidthPadding(paddingSize);
end

function paddingSize = iCalculateSamePadding(filterSize, stride, inputSize)
paddingSize = nnet.internal.cnn.layer.padding.calculateSamePadding(filterSize, stride, inputSize);
end

function initializer = iInternalInitializer(name)
initializer = nnet.internal.cnn.layer.learnable.initializer...
    .initializerFactory(name);
end

% LocalWords:  Learnable nnet cnn learnable convolutional Backpropagation
% LocalWords:  Mkldnn Rnd dlarray
