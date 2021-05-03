classdef TernaryActivation < nnet.layer.Layer
    
    properties (Learnable)
        % Layer learnable parameters
        
        % Scaling coefficient
        thP
        thN
        
    end
    
    methods
        function layer = TernaryActivation(thP,thN,name)
            % Activation  Constructor for the layer
            layer.thP = thP;%rand([1 1 numChannels]);
            layer.thN = thN;%-rand([1 1 numChannels]);
            layer.Name = name;

        end
        
        function [Z,memory] = foward(~, X)

            
            tmp1=X>layer.thP;
            tmp2=X<layer.thN;
            Z=single(tmp1+(-1*tmp2));
            
            %             Z=sign(X);
            %             Z(Z==0)=1;
            memory=[];
            
            %             if existsOnGPU(X)% No-op if already on GPU
            %                 Z = gpuArray(single(Z));
            %             end
            
            
        end
        
        function Z = predict(layer, X)

            tmp1=X>layer.thP;
            tmp2=X<layer.thN;
            Z=single(tmp1+(-1*tmp2));
        end
        
        function [dLdX,dLdPositive,dLdNegative] = backward(layer, X, ~, dLdZ, ~)
            dLdX=dLdZ;
            dLdPositive=max(layer.thP,X) .* dLdZ;
            dLdPositive = sum(dLdPositive,[1 2]);
            dLdPositive = sum(dLdPositive,4);
            
            dLdNegative=min(layer.thN,X) .*dLdZ;
            dLdNegative = sum(dLdNegative,[1 2]);
            dLdNegative = sum(dLdNegative,4);
        end
    end
end