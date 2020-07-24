classdef BinarizedTransposedConvolution2DFunctionalStrategy < ...
        nnet.internal.cnn.layer.util.FunctionalStrategy
    % TransposedConvolution2DFunctionalStrategy    dlarray method strategy
    
    %   Copyright 2019 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X, ...
                weights, bias, ...
                cropping, ...
                verticalStride, horizontalStride, ...
                ~, ~)
            
                        
            weights= sign(weights);  %%%%%%%%%%%%%%%%%%%%%
            weights(weights==0)=1;  %%%%%%%%%%%%%%%%%%%%%
            
            assert( isa(X, 'dlarray') && ~isempty(dims(X)) )
            
            % TODO: use internal API            
            Z = dltranspconv(X, weights, bias, ...
                'Stride', [verticalStride, horizontalStride], ...
                'Cropping', cropping);
            
            memory = [];
        end
    end
end
