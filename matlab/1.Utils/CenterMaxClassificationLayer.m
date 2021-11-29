classdef CenterMaxClassificationLayer < nnet.layer.ClassificationLayer
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        %   This layer has no learnable parameters.
        
        % Categories (column categorical array) The categories of the classes
        % It can store ordinality of the classes as well.
        Categories
        
        Center 
        
        Lamda=0.5;
    end
    
%     properties (Learnable)
%        Center 
%         
%     end
%     
    
    methods
        function this = CenterMaxClassificationLayer(name, numClasses)
            % (Optional) Create a myClassificationLayer.
            
            % Layer constructor function goes here.
            
            this.Name = name;
            
            this.NumClasses = numClasses;
            this.Categories = categorical();
            %             this.ObservationDim = 4;
            
            this.Center=single(randn(1,1,numClasses));
            
        end
        
        function loss = forwardLoss(this, Y, T)
            % Return the loss between the predictions Y and the
            % training targets T.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     ? Predictions made by network
            %         T     ? Training targets
            %
            % Output:
            %         loss  - Loss between Y and T
            
            % Layer forward loss function goes here.
            
            
            sY = size( Y );
            numElems = prod( sY(4:end) );
            loss = -sum( sum( sum(T.*log(nnet.internal.cnn.util.boundAwayFromZero(Y))) ./numElems ) );
            
            
            totalcenter=[];
            for tmp=1:size(Y,4)
                totalcenter=cat(4,this.Center,totalcenter);
            end
            loss= loss + this.Lamda* sum(sum(sum(abs(nnet.internal.cnn.util.boundAwayFromZero(Y)-totalcenter))));
            
        end
        
        function dX = backwardLoss(this, Y, T)
            % Backward propagate the derivative of the loss function.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     ? Predictions made by network
            %         T     ? Training targets
            %
            % Output:
            %         dLdY  - Derivative of the loss with respect to the predictions Y
            
            % Layer backward loss function goes here.
            
            numObservations = size( Y, 4 );
            dX = (-T./nnet.internal.cnn.util.boundAwayFromZero(Y))./numObservations;
            
            totalcenter=[];
            for tmp=1:numObservations
                totalcenter=cat(4,this.Center,totalcenter);
            end
            
            dcenter=single((nnet.internal.cnn.util.boundAwayFromZero(Y)-reshape(totalcenter,[1,1,this.NumClasses,numObservations])));
            dX = gpuArray(dX + this.Lamda*dcenter);
%             this.Center= 0.1.*this.Center;
        end
    end
end