classdef SignumActivation < nnet.layer.Layer
    % Hardware Tanh layer
    % perform the min, max saturation on input value
    % over the range of [-1 1] will be saturated,
    % in the range of [-1,1] will remain still
    % Simple script
    %  Warning! This function does not support GPU acceleration
    %
    %  Custom layer designed by Jeonghoon Kim
    
    methods
        function layer = SignumActivation(name)
            % Activation  Constructor for the layer
            layer.Name = name;
        end
        
        function [Z,memory] = foward(~, X)
            % Forward input data through the layer at prediction time and
            % output the result
            % Inputs:
            %         layer    -    Layer to forward propagate through
            %         X        -    Input data
            % Output:
            %         Z        -    Output of layer forward function
            %             Z=round(max(-1,min(1,X)));
            %             X = gpuArray(X); % No-op if already on GPU
            
            %             if ~isa(X,'gpuArray')% No-op if already on GPU
            %                 X = gpuArray(single(X));
            %             end
            
            
            
            Z=sign(X);
            Z(Z==0)=1;
            memory=[];
            
            %             if existsOnGPU(X)% No-op if already on GPU
            %                 Z = gpuArray(single(Z));
            %             end
            
            
        end
        
        function Z = predict(~, X)
            % Forward input data through the layer at prediction time and
            % output the result
            % Inputs:
            %         layer    -    Layer to forward propagate through
            %         X        -    Input data
            % Output:
            %         Z        -    Output of layer forward function
            %             Z=round(max(-1,min(1,X)));
            
            
            %                                     if ~isa(X,'gpuArray')% No-op if already on GPU
            %                                         X = gpuArray(single(X));
            %                                     end
            
            
            %             Z=X./abs(X);
            Z=sign(X);
            Z(Z==0)=1;
            %             if X==0
            %                 Z=1;
            %             else
            %                 Z=X ./ ABS(X);
            %             end
            
            %                         if existsOnGPU(X)% No-op if already on GPU
            %                             Z = gpuArray(single(Z));
            %                         end
            
            
        end
        
        function dLdX = backward(~, X, ~, dLdZ, ~)
            % Backward propagate the derivative of the loss function through
            % the layer
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X                 - Input data
            %         Z                 - Output of layer forward function
            %         dLdZ              - Gradient propagated from the deeper layer
            %         memory            - Memory value which can be used in
            %                             backward propagation
            % Output:
            %         dLdX              - Derivative of the loss with respect to the
            %                             input data
            
            %  dLdX = 3*(X.^2).*(dLdZ) .*(X>-1 & X<1);
            %             dLdX = (dLdZ) .*(X>-1 & X<1) + (X>=1) -(X<=-1) ;
            
            
            %                                     if existsOnGPU(dLdZ) || existsOnGPU(X)% No-op if already on GPU
            %                                         dLdZ = gpuArray(single(dLdZ));
            %                                     end
            %                         if ~isa(X,'gpuArray')% No-op if already on GPU
            %                             X = gpuArray(single(X));
            %                         end
            
            dLdX = 2.5*sech(X).^2.*(dLdZ); %2 or 3 is good
            
            %             dLdX = (1 - X.^2) .* dLdZ;
            
            %                         if existsOnGPU(dLdZ) || existsOnGPU(X)% No-op if already on GPU
            %                             dLdX = gpuArray(single(dLdX));
            %                         end
            
            %              dLdX = (dLdZ) .*(X>-1 & X<1) ;
            
            
        end
    end
end