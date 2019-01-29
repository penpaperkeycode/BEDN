classdef LayerAnalyzer < handle
    % LayerAnalyzer     This class is intended to be used by the class
    %                   NetworkAnalyzer as a wrapper for layers in the
    %                   package nnet.cnn.layer.*.
    %                   It provides access to some protected methods, and
    %                   encapsulate some functionalities to query
    %                   information from the layer.
    %
    % Properties:
    %   ExternalLayer   External layer being analyzed.
    %   InternalLayer   Internal layer corresponding to the external layer.
    %
    %   Name            Name of the layer being analyzed.
    %   OriginalName    Name of the layer when the analysis started.
    %   DefaultName     Default name used for layers of the same type as
    %                   the one being analyzed.
    %
    %   Type            String identifying the type of layer being
    %                   analyzed.
    %   Description     Description of the layer being analyzed.
    %
    %   Is*Layer        Logical properties indicating whether the layer is
    %                   of certain type.
    %   
    %   Inputs          Table with the source and size of each input to the
    %                   layer.
    %   Outputs         Table with the destination and size of each output
    %                   from the layer.
    %   Learnables      Table with the size of each learnable parameter.
    %   
    %   Precision       Precision object used to initialize the layer.
    %
    % Methods:
    %   this = LayerAnalyzer(input)
    %       Constructor of LayerAnalyzer. Takes as input an external layer.
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    properties ( Dependent )
        
        Name(1,1) string;
        DefaultName(1,1) string;
        
        Type(1,1) string;
        Description(1,1) string;
        
        IsInputLayer(1,1) logical;
        IsImageInputLayer(1,1) logical;
        IsBatchNormalizationLayer(1,1) logical;
        IsOutputLayer(1,1) logical;
        IsClassificationLayer(1,1) logical;
        IsRegressionLayer(1,1) logical;
        IsSoftmaxLayer(1,1) logical;
        IsCustomLayer(1,1) logical;
        IsLSTMLayer(1,1) logical;
        IsRNNLayer(1,1) logical;
        
        IsImageSpecificLayer(1,1) logical;
        IsSequenceSpecificLayer(1,1) logical;
        
    end
    
    properties ( Hidden, Dependent )
        
        IsRenamed(1,1) logical;
        DisplayName(1,1) string;
        
    end
    
    properties ( SetAccess = ...
            { ?nnet.internal.cnn.analyzer.NetworkAnalyzer } )
        
        OriginalName(1,1) string;
        OriginalIndex(1,1) double;
        
    end
    
    properties ( SetAccess = ...
            { ?nnet.internal.cnn.analyzer.NetworkAnalyzer, ...
              ?nnet.internal.cnn.analyzer.util.LayerAnalyzer } )
        
        Inputs table;
        Outputs table;
        Learnables table;
        Dynamics table;
        Hypers table;
        
        Properties table;
        
    end
    
    properties ( SetAccess = private )
        
        ExternalLayer(:,1) nnet.cnn.layer.Layer {mustBeScalarOrEmpty};
        
    end
    
    properties ( Dependent, SetAccess = private )
        
        InternalLayer(1,1) nnet.internal.cnn.layer.Layer;
        IsLayerInputValid(1,1) logical;
        
    end
    
    properties ( Constant, Hidden )
        
        Precision = nnet.internal.cnn.util.Precision('gpuArray');
        
    end
    
    properties ( Access = ...
            { ?nnet.internal.cnn.analyzer.NetworkAnalyzer, ...
              ?nnet.internal.cnn.analyzer.util.LayerAnalyzer } )
        
        HasValidInputSizes(1,1) logical = false;
        HasValidOutputSizes(1,1) logical = false;
        InternalInputSizes;
        InternalOutputSizes;
        
        HasPropagatedOutputSize(1,1) logical = false;
        HasPropagatedLearnableSize(1,1) logical = false;
        HasPropagatedDynamicSize(1,1) logical = false;
        
        HasPopulatedHyperValues(1,1) logical = false;
        HasPopulatedPropertyValues(1,1) logical = false;
        
    end
    
    methods
        
        function this = LayerAnalyzer(layer)
            % Constructor of LayerAnalyzer. Takes as input an external
            % layer.
            
            this.ExternalLayer = layer;

            % Obtain the proxy object to the external layer.
            % This proxy is used to obtain some information (Type
            % and Description, Internal Layer), and stored locally.
            proxy = iLayerProxy(this.ExternalLayer);
            
            % Store the original name of the layer.
            this.OriginalName = this.Name;
            
            % Generate list of parameters
            this.Learnables = iMakeTable('Parameter', proxy.LearnableParameters, ...
                                         'Size', [], ...
                                         'Initialized', false);
            
            this.Dynamics   = iMakeTable('Parameter', proxy.DynamicParameters, ...
                                         'Size', [], ...
                                         'Initialized', false);
            
            this.Hypers     = iMakeTable('Parameter', proxy.HyperParameters, ...
                                         'Value', []);
            this.Properties = iMakeTable('Property', proxy.HyperParameters, ...
                                         'Value', []);
            
            % Generate list of inputs / outputs
            this.Outputs    = iMakeTable('Port', proxy.OutputPorts, ...
                                         'Destination', string.empty, ...
                                         'Size', []);
            
            this.Inputs     = iMakeTable('Port', proxy.InputPorts, ...
                                         'Source', string.empty, ...
                                         'Size', []);
        end
        
    end
    
    methods ( Hidden )
        
        function varargout = subsref(this,s)
            if s(1).type == "()" && iscell(s(1).subs) && isstring(s(1).subs{1})
                [~, ind] = ismember(s(1).subs{1}, string({this.Name}));
                s(1).subs = {ind};
            end
            try
                [varargout{1:nargout}] = builtin('subsref',this,s);
            catch e
                throwAsCaller(e);
            end
        end
        
        function n = numArgumentsFromSubscript(this,s,varargin)
            if s(1).type == "()" && iscell(s(1).subs) && isstring(s(1).subs{1})
                [~, ind] = ismember(s(1).subs{1}, string({this.Name}));
                s(1).subs = {ind};
            end
            n = builtin('numArgumentsFromSubscript',this,s,varargin{:});
        end
        
    end
    
    methods ( Hidden, Access = protected )
        
        function validateInputs(this)
            % Takes the sizes of the inputs and populates the sizes of the
            % outputs as well as the sizes of the learnable parameters.
            %
            % If all inputs are missing, this function is a no-op.
            %
            % If there is any invalid input, a best effort is made to
            % estimate the output size regardless.
            %
            % The size of the learnable parameters is only computed if all
            % the inputs are valid.
            
            % Forward propagate the size of the inputs.
            
            assert(isscalar(this));
            
            layer = this.InternalLayer;

            if isprop(layer, 'InputSize') && isempty(this.Inputs)
                if iscell(layer.InputSize)
                    inputSizes = layer.InputSize;
                else
                    inputSizes = {layer.InputSize};
                end
            else
                inputSizes = this.Inputs.Size;
            end
            
            validInputs = ~any( cellfun(@isempty, inputSizes) );
            
            if iscell(inputSizes) && numel(inputSizes) == 1
                inputSizes = inputSizes{1};
            end
            
            % Infer the size of input dependent parameters and check 
            % whether the input size is valid. Finally check that the input
            % size can be propagated.
            if validInputs
                try
                    if ~layer.HasSizeDetermined
                        layer = layer.inferSize(inputSizes);
                    end
                    validInputs = layer.isValidInputSize(inputSizes);
                    layer.forwardPropagateSize(inputSizes);
                catch
                    validInputs = false;
                end
            end
            
            this.InternalLayer = layer;
            this.HasValidInputSizes = validInputs;
            
            % If we could not infer valid input sizes, but the layer
            % defines input sizes, try using those as the internal input
            % sizes that will be used to propagate (regardless of the layer
            % having no input ports)
            if ~validInputs && isprop(layer, 'InputSize')
                inputSizes = layer.InputSize;
            end
            this.InternalInputSizes = inputSizes;
        end
        
        function populateOutputs(this)
            % Takes the sizes of the inputs and populates the sizes of the
            % outputs.
            %
            % If there is any invalid input, a best effort is made to
            % estimate the output size regardless.
            
            % Forward propagate the size of the inputs.
            
            assert(isscalar(this));
            
            layer = this.InternalLayer;
            inputSizes = this.InternalInputSizes;
            validInputs = this.HasValidInputSizes;

            % Now try to propagate the size, even if inferSize failed. We might
            % still succeed.
            outputSizes = cell(1, size(this.Outputs,1));
            outputSizes(:) = {[nan nan nan]};
            try
                outputSizes = layer.forwardPropagateSize(inputSizes);
            catch
            end

            this.InternalOutputSizes = outputSizes;
            if ~iscell(outputSizes)
                outputSizes = {outputSizes};
            else
                outputSizes = outputSizes(:);
            end

            % Double check that the output size is valid:
            %   finite, positive, and integer
            validOutputs = validInputs & iIsValidOutputSizes(outputSizes);

            % If the output size is invalid, replace it with nan to flag it
            % as invalid.
            for i=find(~validOutputs(:)')
                outputSizes{i} = nan(size(outputSizes{i}));
            end

            % Assign the outputs
            if ~isempty(outputSizes)
                this.Outputs.Size = outputSizes(1:size(this.Outputs,1),1);
            end
            
            this.HasValidOutputSizes = all(validOutputs);
        end
        
        function populateLearnables(this)
            % Takes the sizes of the inputs/outputs and populates the sizes
            % of the learnable parameters.
            %
            % The size of the learnable parameters is only computed if all
            % the inputs and outputs are valid.
            
            % Detect whether the learnable parameter is initialized
            external = this.ExternalLayer;
            initialized = false(size(this.Learnables,1), 1);
            for i=1:size(this.Learnables,1)
                p = this.Learnables.Parameter{i};
                initialized(i) = ~isempty(external.(p));
                this.Learnables.Initialized{i} = initialized(i);
                if initialized(i)
                    this.Learnables.Size{i} = size(external.(p));
                end
            end
            
            % If everything is initialized, we don't need to infer anything
            if all(initialized)
                return;
            end

            % Obtain the size of the learnable parameters.
            if ~this.HasValidInputSizes || ~this.HasValidOutputSizes
                for i=1:size(this.Learnables,1)
                    this.Learnables.Size{i} = nan;
                end
                return;
            end
            
            % Initialize parameters. Currently this is the only way we
            % have to obtain their sizes
            layer = this.InternalLayer;
            layer = layer.setupForHostPrediction();
            layer = layer.initializeLearnableParameters(this.Precision);
            external = iInternalToExternal(this.ExternalLayer, layer);

            for i=1:size(this.Learnables,1)
                p = this.Learnables.Parameter{i};
                this.Learnables.Size{i} = size(external.(p));
            end
        end
        
        function populateDynamic(this)
            % Takes the sizes of the inputs/outputs and populates the sizes
            % of the dynamic parameters.
            %
            % The size of the dynamic parameters is only computed if all
            % the inputs are valid.
            
            % Detect whether the learnable parameter is initialized
            external = this.ExternalLayer;
            initialized = false(size(this.Dynamics,1), 1);
            for i=1:size(this.Dynamics,1)
                p = this.Dynamics.Parameter{i};
                initialized(i) = ~isempty(external.(p));
                this.Dynamics.Initialized{i} = initialized(i);
                if initialized(i)
                    this.Dynamics.Size{i} = size(external.(p));
                end
            end
            
            % If everything is initialized, we don't need to infer anything
            if all(initialized)
                return;
            end
            
            % Obtain the size of the dynamic parameters.
            if ~this.HasValidInputSizes || ~this.HasValidOutputSizes
                for i=1:size(this.Dynamics,1)
                    this.Dynamics.Size{i} = nan;
                end
                return;
            end
            
            if ~this.IsLSTMLayer
                return;
            end
            
            % Initialize parameters. Currently this is the only way we
            % have to obtain their sizes
            layer = this.InternalLayer;
            layer = layer.setupForHostPrediction();
            layer = layer.initializeDynamicParameters(this.Precision);
            external = iInternalToExternal(this.ExternalLayer, layer);

            for i=1:size(this.Dynamics,1)
                p = this.Dynamics.Parameter{i};
                this.Dynamics.Size{i} = size(external.(p));
            end
        end
        
        function populateHypers(this)
            % Initialize parameters. Currently this is the only way we
            % have to obtain their sizes
            external = this.ExternalLayer;
            for i=1:size(this.Hypers,1)
                p = this.Hypers.Parameter{i};
                this.Hypers.Value{i} = external.(p);
            end
        end
        
        function populateProperties(this)
            % Initialize parameters. Currently this is the only way we
            % have to obtain their sizes
            external = this.ExternalLayer;
            for i=1:size(this.Properties,1)
                p = this.Properties.Property{i};
                this.Properties.Value{i} = external.(p);
            end
        end
        
    end
    
    methods
        
        function set.Inputs(this, in)
            % Make sure we don't store invalid sizes
            for i=1:size(in,1)
                in.Size{i}( ~iIsNatural(in.Size{i}) ) = 0;
            end
            this.Inputs = in;
            
            this.HasPropagatedOutputSize = false; %#ok<MCSUP>
            this.HasPropagatedLearnableSize = false; %#ok<MCSUP>
            
            this.validateInputs();
        end
        
        function v = get.Outputs(this)
            if ~this.HasPropagatedOutputSize
                this.HasPropagatedOutputSize = true;
                this.populateOutputs();
            end
            v = this.Outputs;
        end
        function v = get.Learnables(this)
            if ~this.HasPropagatedLearnableSize
                this.HasPropagatedLearnableSize = true;
                this.populateLearnables();
            end
            v = this.Learnables;
        end
        function v = get.Dynamics(this)
            if ~this.HasPropagatedDynamicSize
                this.HasPropagatedDynamicSize = true;
                this.populateDynamic();
            end
            v = this.Dynamics;
        end
        function v = get.Hypers(this)
            if ~this.HasPopulatedHyperValues
                this.HasPopulatedHyperValues = true;
                this.populateHypers();
            end
            v = this.Hypers;
        end
        function v = get.Properties(this)
            if ~this.HasPopulatedPropertyValues
                this.HasPopulatedPropertyValues = true;
                this.populateProperties();
            end
            v = this.Properties;
        end
        
        function v = get.InternalLayer(this)
            proxy = iLayerProxy(this.ExternalLayer);
            v = proxy.InternalLayer;
        end
        function set.InternalLayer(this, v)
            this.ExternalLayer = iInternalToExternal(this.ExternalLayer, v);
        end
        
        function v = get.Type(this)
            proxy = iLayerProxy(this.ExternalLayer);
            v = string(proxy.Type);
        end
        function v = get.Description(this)
            proxy = iLayerProxy(this.ExternalLayer);
            v = string(proxy.Description);
        end
        
        function v = get.Name(this)
            v = string(this.ExternalLayer.Name);
        end
        function v = get.DefaultName(this)
            v = string(this.InternalLayer.DefaultName);
            if isempty(v) || v == ""
                v = "layer";
            end
        end
        function set.Name(this,v)
            this.InternalLayer.Name = v;
            this.ExternalLayer.Name = v;
        end
        
        function tf = get.IsRenamed(this)
            tf = ( this.Name ~= this.OriginalName );
        end
        function v = get.DisplayName(this)
            if this.IsRenamed
                v = string(iMessage('NetworkAnalyzer:UnnamedLayer', ...
                                    this.OriginalIndex));
            else
                v = string(iMessage('NetworkAnalyzer:NamedLayer', ...
                                    this.Name));
            end
        end
        
        function tf = get.IsImageInputLayer(this)
            tf = isa(this.InternalLayer, 'nnet.internal.cnn.layer.ImageInput');
        end
        function tf = get.IsBatchNormalizationLayer(this)
            tf = isa(this.InternalLayer, 'nnet.internal.cnn.layer.BatchNormalization');
        end
        function tf = get.IsInputLayer(this)
            tf = isa(this.InternalLayer, 'nnet.internal.cnn.layer.InputLayer');
        end
        function tf = get.IsOutputLayer(this)
            tf = isa(this.InternalLayer, 'nnet.internal.cnn.layer.OutputLayer');
        end
        function tf = get.IsClassificationLayer(this)
            tf = isa(this.InternalLayer, 'nnet.internal.cnn.layer.ClassificationLayer');
        end
        function tf = get.IsRegressionLayer(this)
            tf = isa(this.InternalLayer, 'nnet.internal.cnn.layer.RegressionLayer');
        end
        function tf = get.IsSoftmaxLayer(this)
            tf = isa(this.InternalLayer, 'nnet.internal.cnn.layer.Softmax');
        end
        function tf = get.IsCustomLayer(this)
            tf = isa(this.InternalLayer, 'nnet.internal.cnn.layer.CustomLayer') ...
                || isa(this.InternalLayer, 'nnet.internal.cnn.layer.CustomClassificationLayer') ...
                || isa(this.InternalLayer, 'nnet.internal.cnn.layer.CustomRegressionLayer');
        end
        
        function tf = get.IsLSTMLayer(this)
            tf = isa(this.InternalLayer, 'nnet.internal.cnn.layer.LSTM');
        end
        function tf = get.IsRNNLayer(this)
            tf = isa(this.InternalLayer, 'nnet.internal.cnn.layer.Recurrent');
        end
        
        function tf = get.IsSequenceSpecificLayer(this)
            tf = this.IsRNNLayer ...
                || isa(this.InternalLayer, 'nnet.internal.cnn.layer.SequenceInput');
        end
        function tf = get.IsImageSpecificLayer(this)
            imageSpecific = "nnet.internal.cnn.layer." + [
                    "AveragePooling2D"
                    "BatchNormalization"
                    "Convolution2D"
                    "LocalMapNorm2D"
                    "ImageInput"
                    "MaxPooling2D"
                    "MaxUnpooling2D"
                    "TransposedConvolution2D"];
            tf = cellfun(@(class) isa(this.InternalLayer, class), imageSpecific);
            tf = any(tf);
        end
        function tf = get.IsLayerInputValid(this)
            tf = this.HasValidInputSizes;
        end
    end
        
end

function t = iMakeTable(rowName, rows, varargin)
    % Fast constructor for tables, since cell2table and struct2table would
    % have a negative impact in performance.
    % tbl = iMakeTable('Rows', ["row1", "row2", ...], ...
    %                  'Column1', DefaultValueOfCol1, ...
    %                  'Column2', DefaultValueOfCol2 );
    % * rowName     The name given to the "row" dimension of the table
    % * rows        The names given to each row. The table will have one
    %               row per element in "rows"
    % * ColName,ColValue
    %               A name value pair of column names and default column
    %               value. Each row will contain the default value for that
    %               column
    
    nRows = numel(rows);
    val = arrayfun(@(x) repmat(x,nRows,1), varargin(2:2:end), 'UniformOutput', false);
    vars = varargin(1:2:end);
    rows = cellstr(rows(:));
    t = table.init(val(:)',numel(rows),rows,numel(vars),vars);
    t.Properties.DimensionNames = {char(rowName), 'Variables'};
end

function o = iLayerProxy(varargin)
    o = nnet.internal.cnn.analyzer.util.LayerProxy(varargin{:});
end

function tf = iIsNatural(v)
    % Returns true if a number belongs to the group of natural numbers
    tf = isfinite(v) & ( v > 0 ) & ( v == round(v) );
end

function externalLayer = iInternalToExternal(externalClass, internalLayer)
    map = nnet.internal.cnn.layer.util.InternalExternalMap(externalClass);
    externalLayer = map.externalLayers({internalLayer});
end

function tf = iIsValidOutputSizes(outputSizes)
    if ~iscell(outputSizes)
        outputSizes = {outputSizes};
    end
    tf = false(size(outputSizes));
    for i=1:numel(outputSizes)
        tf(i) = ~isempty(outputSizes{i}) ...
                && all(iIsNatural(outputSizes{i}));
    end
end

function tf = mustBeScalarOrEmpty(in)
    tf = isscalar(in) || isempty(in);
end

function msg = iMessage(id, varargin)
    id = "nnet_cnn:internal:cnn:analyzer:" + id;
    msg = message(id{1}, varargin{:});
end
