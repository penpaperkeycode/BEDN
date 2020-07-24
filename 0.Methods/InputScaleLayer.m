classdef InputScaleLayer < nnet.layer.Layer
    % Hardware Tanh layer
    % perform the min, max saturation on input value
    % over the range of [-1 1] will be saturated,
    % in the range of [-1,1] will remain still
    % Simple script
    %  Warning! This function does not support GPU acceleration
    %
    %  Custom layer designed by Jeonghoon Kim
    
    methods
        function layer = InputScaleLayer(name)
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
            
            memory=[];
            scale_min =-1;
            scale_max =1;
            Ztmp=scale_min + (scale_max - scale_min) .* single(X) ./ 255;
%             Ztmp=reshape(Ztmp,1,[]);
%             Xvaluetmp = sfi(single(Ztmp),8,7);
%             Xvaluetmp2=str2num(Xvaluetmp.Value);
%             Xvalue = reshape(Xvaluetmp2,size(X,1),size(X,2),size(X,3),size(X,4));
%             Ztmp(-1<Ztmp<0)= Ztmp(-1<Ztmp<0)- 0.0078125;
            Zmod=mod(Ztmp,0.0078125);
            Xvalue=Ztmp-Zmod;
            Xvalue(Xvalue==1)= Xvalue-0.0078125;
% % %             Xvalue(Xvalue>=0)=Xvalue(Xvalue>=0)+0.0078125;
% % %             Xvalue(Xvalue>1)=Xvalue(Xvalue>1)-0.0078125;
            
            %             Xvalue=str2double(Xvaluetmp.Value); %str2num
            Z=single(Xvalue);
            
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
            
            scale_min =-1;
            scale_max =1;
            Ztmp=scale_min + (scale_max - scale_min) .* single(X) ./ 255; 
            
%             Ztmp=reshape(Ztmp,1,[]);
%             Xvaluetmp = sfi(single(Ztmp),8,7);
%             Xvaluetmp2=str2num(Xvaluetmp.Value);
%             Xvalue = reshape(Xvaluetmp2,size(X,1),size(X,2),size(X,3),size(X,4));
            %permute or reshape -> sfi -> str2double -> reshape
            
            Zmod=mod(Ztmp,0.0078125);
            Xvalue=Ztmp-Zmod;
            Xvalue(Xvalue==1)= Xvalue(Xvalue==1)-0.0078125;
% % %             Xvalue(Xvalue>=0)=Xvalue(Xvalue>=0)+0.0078125;
% % %             Xvalue(Xvalue>1)=Xvalue(Xvalue>1)-0.0078125;
%             Xvalue(Xvalue<0)= Xvalue(Xvalue<0)- 0.0078125;
%             Xvalue(Xvalue<-1)=-1;
            %             Xvaluetmp = sfi(Ztmp,8,7);
            %             Xvaluetmp2=str2num(Xvaluetmp.Value); %str2num
            %             Xvalue = reshape(Xvaluetmp2,size(X,1),size(X,2),size(X,3),size(X,4));
            Z=single(Xvalue);
            %             Z=scale_min + (scale_max - scale_min) * X / 255;
            
        end
        
        function dLdX = backward(~, ~, ~, dLdZ, ~)
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
            
            
            dLdX = dLdZ;
            
            
            
            
        end
    end
end