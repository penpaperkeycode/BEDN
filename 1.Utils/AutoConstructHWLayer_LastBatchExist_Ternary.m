function HardwareLayers=AutoConstructHWLayer_LastBatchExist_Ternary(Net)
% Input = Binarized Neural Network
% Output = Hardware Neural Network for Classification (Prediction) Only
HardwareLayers=[];
conv_memory1=[];
conv_memory2=[];
first_conv_finder=[];
Xilinx=0;

for i=1:size(Net.Layers,1)
    tmp_string=Net.Layers(i,1).Name;
    disp(tmp_string)
    
    if ismethod(Net.Layers(i,1),'BatchNormalizationLayer')
        
        if i~=size(Net.Layers,1)-2
            
            epsilon=1.0e-11;
            Mean1d = Net.Layers(i, 1).TrainedMean;
            Variance1d = Net.Layers(i, 1).TrainedVariance;
            Scale1d = Net.Layers(i, 1).Scale;
            Offset1d =Net.Layers(i, 1).Offset;
            
            BatchBias1d=(  (  sqrt(Variance1d).*Offset1d./(Scale1d+epsilon)  )  -Mean1d   );
            Scale_sign=sign(Scale1d); %check Guinness paper, they were wrong about BatchNorm equation.
            BatchBias1d=Scale_sign.*BatchBias1d; %For Right assumption, cross their own sign.
            tmpW=Net.Layers(conv_memory1, 1).Weights;
            tmpW(tmpW>0.0001)=1;
            tmpW(tmpW<-0.0001)=-1;
            tmpW(abs(tmpW)<=0.0001)=0;
            
            HardwareLayers(conv_memory2,1).Weights= tmpW  ;
            
%             HardwareLayers(conv_memory2,1).Weights(HardwareLayers(conv_memory2,1).Weights==0)=1 ;  %Check whethere this line is right
            
            if Scale_sign<0  % Check the rule when Gamma equals zero
                HardwareLayers(conv_memory2,1).Weights(HardwareLayers(conv_memory2,1).Weights==1)=2 ;
                HardwareLayers(conv_memory2,1).Weights(HardwareLayers(conv_memory2,1).Weights==-1)=1 ;
                HardwareLayers(conv_memory2,1).Weights(HardwareLayers(conv_memory2,1).Weights==2)=-1 ;
            end
            
            
            if first_conv_finder==1 && Xilinx==1
                bias1d_obj = sfi(Net.Layers(conv_memory1,1).Bias + BatchBias1d,24,8);
                Xvalue=double(str2num(bias1d_obj.Value));
                Xvalue=reshape(Xvalue,[1,1,64]);
                %             Ztmp=Net.Layers(conv_memory1,1).Bias + BatchBias1d;
                %             Zmod=mod(Ztmp,0.00390625);%1/256 24bit quantization
                %             Xvalue=Ztmp-Zmod;
                HardwareLayers(conv_memory2,1).Bias=Xvalue;% Net.Layers(conv_memory1,1).Bias + BatchBias1d;
            else
                if ismethod(HardwareLayers(conv_memory2,1),'FullyConnectedLayer')
                    HardwareLayers(conv_memory2,1).Bias= floor(Net.Layers(conv_memory1,1).Bias + reshape(BatchBias1d,10,1) );
                else
                    HardwareLayers(conv_memory2,1).Bias= floor(Net.Layers(conv_memory1,1).Bias + BatchBias1d );
                end
            end
            conv_memory1=[];
            conv_memory2=[];
        else
                    HardwareLayers=[
            HardwareLayers;
            Net.Layers(i,1)
            ];
        end
        
    elseif ismethod(Net.Layers(i,1),'Convolution2DLayer')
        if isempty(first_conv_finder)
            first_conv_finder=1;
        else
            first_conv_finder=2;
        end
        HardwareLayers=[
            HardwareLayers;
            convolution2dLayer(Net.Layers(i,1).FilterSize, ...
            Net.Layers(i,1).NumFilters, ...
            'Padding',Net.Layers(i,1).PaddingSize, ... %             'Padding','same', ... %             'PaddingSize',Net.Layers(i,1).PaddingSize, ...
            'Stride',Net.Layers(i,1).Stride, ...
            'Name',Net.Layers(i,1).Name)
            ];
        conv_memory1=i;
        conv_memory2=size(HardwareLayers,1);
        
    elseif ismethod(Net.Layers(i,1),'TransposedConvolution2DLayer')
        HardwareLayers=[
            HardwareLayers;
            transposedConv2dLayer(Net.Layers(i,1).FilterSize, ...
            Net.Layers(i,1).NumFilters, ...
            'Cropping','same', ..., ...%'same', ... %'CroppingSize',Net.Layers(i,1).CroppingSize, ...
            'Stride',Net.Layers(i,1).Stride, ...
            'BiasLearnRateFactor',0,...
            'Name',Net.Layers(i,1).Name)
            ];
        conv_memory1=i;
        conv_memory2=size(HardwareLayers,1);
        
    elseif ismethod(Net.Layers(i,1),'FullyConnectedLayer')
        if isempty(first_conv_finder)
            first_conv_finder=1;
        else
            first_conv_finder=2;
        end
        HardwareLayers=[
            HardwareLayers;
            fullyConnectedLayer(Net.Layers(i,1).OutputSize, ...
            'Name',Net.Layers(i,1).Name)
            ];
        conv_memory1=i;
        conv_memory2=size(HardwareLayers,1);
    else
        HardwareLayers=[
            HardwareLayers;
            Net.Layers(i,1)
            ];
    end
end

% figure('Units','normalized','Position',[0 0.1 0.3 0.8]);
% plot(layerGraph(HardwareLayers))
% grid on
% title('Trained Network Architecture')
% ax = gca;
% outerpos = ax.OuterPosition;
% ti = ax.TightInset;
% left = outerpos(1) + ti(1);
% bottom = outerpos(2) + ti(2);
% ax_width = outerpos(3) - ti(1) - ti(3);
% ax_height = outerpos(4) - ti(2) - ti(4);
% ax.Position = [left bottom ax_width ax_height];
% % axis off
% set(gca,'XTickLabel',[],'YTickLabel',[]) %Axis number off
% fig = gcf;
% fig.PaperPositionMode = 'auto';
% fig_pos = fig.PaperPosition;
% fig.PaperSize = [fig_pos(3) fig_pos(4)];
end