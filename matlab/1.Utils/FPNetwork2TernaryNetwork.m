function SimNet=FPNetwork2TernaryNetwork(Net)
% Input = Binarized Neural Network
% Output = Hardware Neural Network for Classification (Prediction) Only

lastBN=1; %last BatchNrom Exist == 1

TernaryNetwork=[];
conv_memory1=[];
conv_memory2=[];
first_conv_finder=[];
%Xilinx=0;

for i=1:size(Net.Layers,1)
    tmp_string=Net.Layers(i,1).Name;
    disp(tmp_string)
    
    if ismethod(Net.Layers(i,1),'BatchNormalizationLayer')
        if ~isempty(conv_memory1)
            tmpW=Net.Layers(conv_memory1, 1).Weights;
            tmpW(tmpW>0.01)=1;
            tmpW(tmpW<-0.01)=-1;
            tmpW(abs(tmpW)<=0.01)=0;
            TernaryNetwork(conv_memory2,1).Weights= tmpW  ;
            
            if i~=size(Net.Layers,1)-2 
                epsilon=1.0e-11;
                Mean1d = Net.Layers(i, 1).TrainedMean;
                Variance1d = Net.Layers(i, 1).TrainedVariance;
                Scale1d = Net.Layers(i, 1).Scale;
                Offset1d =Net.Layers(i, 1).Offset;
                
                BatchBias1d=(  (  sqrt(Variance1d).*Offset1d./(Scale1d+epsilon)  )  -Mean1d   );
                Scale_sign=sign(Scale1d); %check Guinness paper, they were wrong about BatchNorm equation.
                BatchBias1d=Scale_sign.*BatchBias1d; %For Right assumption, cross their own sign.
                
                if Scale_sign<0  % Check the rule when Gamma equals zero
                    TernaryNetwork(conv_memory2,1).Weights(TernaryNetwork(conv_memory2,1).Weights==1)=2 ;
                    TernaryNetwork(conv_memory2,1).Weights(TernaryNetwork(conv_memory2,1).Weights==-1)=1 ;
                    TernaryNetwork(conv_memory2,1).Weights(TernaryNetwork(conv_memory2,1).Weights==2)=-1 ;
                end
            else
                TernaryNetwork=[
                    TernaryNetwork;
                    Net.Layers(i,1)
                    ];
            end
            
            TernaryNetwork(conv_memory2,1).Bias=zeros(size(Net.Layers(conv_memory1,1).Bias));
            
            %             if first_conv_finder==1 && Xilinx==1
            %                 bias1d_obj = sfi(Net.Layers(conv_memory1,1).Bias + BatchBias1d,24,8);
            %                 Xvalue=double(str2num(bias1d_obj.Value));
            %                 Xvalue=reshape(Xvalue,[1,1,64]);
            %                 %             Ztmp=Net.Layers(conv_memory1,1).Bias + BatchBias1d;
            %                 %             Zmod=mod(Ztmp,0.00390625);%1/256 24bit quantization
            %                 %             Xvalue=Ztmp-Zmod;
            %                 TernaryNetwork(conv_memory2,1).Bias=Xvalue;% Net.Layers(conv_memory1,1).Bias + BatchBias1d;
            %             else
            %                 if ismethod(TernaryNetwork(conv_memory2,1),'FullyConnectedLayer')
            %                     TernaryNetwork(conv_memory2,1).Bias= floor(Net.Layers(conv_memory1,1).Bias + reshape(BatchBias1d,10,1) );
            %                 else
            %                     TernaryNetwork(conv_memory2,1).Bias= floor(Net.Layers(conv_memory1,1).Bias + BatchBias1d );
            %                 end
            %             end
            conv_memory1=[];
            conv_memory2=[];
        end
        
    elseif ismethod(Net.Layers(i,1),'Convolution2DLayer')
        if isempty(first_conv_finder)
            first_conv_finder=1;
        else
            first_conv_finder=2;
        end
        TernaryNetwork=[
            TernaryNetwork;
            convolution2dLayer(Net.Layers(i,1).FilterSize, ...
            Net.Layers(i,1).NumFilters, ...
            'Padding',Net.Layers(i,1).PaddingSize, ... %             'Padding','same', ... %             'PaddingSize',Net.Layers(i,1).PaddingSize, ...
            'Stride',Net.Layers(i,1).Stride, ...
            'Name',Net.Layers(i,1).Name)
            ];
        conv_memory1=i;
        conv_memory2=size(TernaryNetwork,1);
        
    elseif ismethod(Net.Layers(i,1),'TransposedConvolution2DLayer')
        TernaryNetwork=[
            TernaryNetwork;
            transposedConv2dLayer(Net.Layers(i,1).FilterSize, ...
            Net.Layers(i,1).NumFilters, ...
            'Cropping','same', ..., ...%'same', ... %'CroppingSize',Net.Layers(i,1).CroppingSize, ...
            'Stride',Net.Layers(i,1).Stride, ...
            'BiasLearnRateFactor',0,...
            'Name',Net.Layers(i,1).Name)
            ];
        conv_memory1=i;
        conv_memory2=size(TernaryNetwork,1);
        
    elseif ismethod(Net.Layers(i,1),'FullyConnectedLayer')
        if isempty(first_conv_finder)
            first_conv_finder=1;
        else
            first_conv_finder=2;
        end
        TernaryNetwork=[
            TernaryNetwork;
            fullyConnectedLayer(Net.Layers(i,1).OutputSize, ...
            'Name',Net.Layers(i,1).Name)
            ];
        conv_memory1=i;
        conv_memory2=size(TernaryNetwork,1);
        
    elseif ismethod(Net.Layers(i,1),'TernaryActivation')
        
        thP=Net.Layers(i,1).thP;
        thN=Net.Layers(i,1).thN;
        
        epsilon=1.0e-11;
        Mean1d = Net.Layers(i-1, 1).TrainedMean; %mu
        Variance1d = Net.Layers(i-1, 1).TrainedVariance;
        Scale1d = Net.Layers(i-1, 1).Scale; %gamma
        Offset1d =Net.Layers(i-1, 1).Offset; %beta
        
        BNthP=floor(((thP-Offset1d).*(sqrt(Variance1d)./(epsilon+Scale1d))+Mean1d)-BatchBias1d);
        BNthN=floor(((thN-Offset1d).*(sqrt(Variance1d)./(epsilon+Scale1d))+Mean1d)-BatchBias1d);
        
        TernaryNetwork=[
            TernaryNetwork;
            TernaryActivation(BNthP,BNthN,Net.Layers(i,1).Name)
            ];
    else
        TernaryNetwork=[
            TernaryNetwork;
            Net.Layers(i,1)
            ];
    end
end

nettmp=TernaryNetwork;

if isa(Net,'DAGNetwork')
    Connectionstmp=Net.Connections;
    ConnectionsArray=table2array(Connectionstmp);
    ConnectionsArray1=ConnectionsArray(:,1);
    ConnectionsArray2=ConnectionsArray(:,2);
    ConnectionsArray1index=[];
    ConnectionsArray2index=[];
    for i=1:size(Connectionstmp,1)
        if strfind(ConnectionsArray1{i},'BatchNorm')
            ConnectionsArray1index=[ConnectionsArray1index,i];
        elseif strfind(ConnectionsArray2{i},'BatchNorm')
            ConnectionsArray2index=[ConnectionsArray2index,i];
        end
    end
    if lastBN==1
        ConnectionsArray1index=ConnectionsArray1index(1:end-1);
        ConnectionsArray2index=ConnectionsArray2index(1:end-1);
    end
    ConnectionsArray1(ConnectionsArray1index)=[];
    ConnectionsArray2(ConnectionsArray2index)=[];
    ConnectionsArray=[ConnectionsArray1,ConnectionsArray2];
    ConnectionsTable = cell2table(ConnectionsArray,'VariableNames',{'Source','Destination'});
    lgraph = createLgraphUsingConnections(nettmp,ConnectionsTable);
else
    lgraph=layerGraph(nettmp);
    %SimNet=SeriesNetwork(HardwareLayers);
end
SimNet=DAGNetwork.loadobj(lgraph);

end