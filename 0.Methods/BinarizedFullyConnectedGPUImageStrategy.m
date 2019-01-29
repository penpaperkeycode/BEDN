classdef BinarizedFullyConnectedGPUImageStrategy < nnet.internal.cnn.layer.util.FullyConnectedImageStrategy
    % FullyConnectedHostImageStrategy   Execution strategy for running the
    % fully connected layer on the GPU with image inputs
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function X = sendToDevice(~, X)
            X = gpuArray(X);
        end
        
        function Z = convolveForward(~, X, weights)
            
            weights= sign(weights);  %%%%%%%
            weights(weights==0)=1;
            
            Z = nnet.internal.cnngpu.convolveForward2D(X, weights, 0, 0, 0, 0, 1, 1);
        end
        
        function dX = convolveBackwardData(~, X, weights, dZ)
            
            weights= sign(weights);  %%%%%%%
            weights(weights==0)=1;
            dX = nnet.internal.cnngpu.convolveBackwardData2D(X, weights, dZ, 0, 0, 0, 0, 1, 1);
        end
        
        function dW = convolveBackwardFilter(~, X, weights, dZ)
            
            weights= sign(weights);  %%%%%%%
            weights(weights==0)=1;
            dW = nnet.internal.cnngpu.convolveBackwardFilter2D(X, weights, dZ, 0, 0, 0, 0, 1, 1);
        end
        
        function dB = convolveBackwardBias(~, dZ)
            dB = nnet.internal.cnngpu.convolveBackwardBias2D(dZ);
        end
    end
end