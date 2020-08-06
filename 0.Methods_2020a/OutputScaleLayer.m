classdef OutputScaleLayer < nnet.layer.Layer
    % Hardware Tanh layer
    % perform the min, max saturation on input value
    % over the range of [-1 1] will be saturated,
    % in the range of [-1,1] will remain still
    % Simple script
    %  Warning! This function does not support GPU acceleration
    %
    %  Custom layer designed by Jeonghoon Kim
    properties (Learnable)
        % Layer learnable parameters
        
        % Scaling coefficient
        Scale
    end
    
    methods
        function layer = OutputScaleLayer(scale,name)
            % Activation  Constructor for the layer
            layer.Name = name;
            %layer.Scale=scale;
%             scale_mod=mod(scale,0.00390625); %7bit= 0.0078125 %8bit = 0.00390625
%             Xvalue=scale-scale_mod;
%             Xvalue(Xvalue==1) = Xvalue - 0.0078125;
            layer.Scale=single(scale); %24bit Quantization
        end
        
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result
            % Inputs:
            %         layer    -    Layer to forward propagate through
            %         X        -    Input data
            % Output:
            %         Z        -    Output of layer forward function
            %             Z=round(max(-1,min(1,X)));
            Z=layer.Scale.*X;
            
        end
        
    end
end