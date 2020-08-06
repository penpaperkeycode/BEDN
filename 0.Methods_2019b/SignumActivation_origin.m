classdef SignumActivation_origin < nnet.layer.Layer
    methods
        function layer = SignumActivation(name)
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
            dLdX = (dLdZ) .*(X>-1 & X<1) ;
        end
    end
end