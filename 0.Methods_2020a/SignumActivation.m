classdef SignumActivation < nnet.layer.Layer
    methods
        function layer = SignumActivation(name)
            % Activation  Constructor for the layer
            layer.Name = name;
            
        end
        
        function [Z,memory] = foward(~, X)
            Z=sign(X);
            Z(Z==0)=1;
            memory=[];
            
        end
        
        function Z = predict(~, X)   
            Z=sign(X);
            Z(Z==0)=1;
            
        end
        
        function dLdX = backward(~, X, ~, dLdZ, ~)
            dLdX = 2.5*sech(X).^2.*(dLdZ); 
             %              dLdX = (dLdZ) .*(X>-1 & X<1) ;
            
        end
    end
end