function HardwareLayers=AutoConstructHWLayerCustomScaleVersion(Net)
% Input = Binarized Neural Network
% Output = Hardware Neural Network for Classification (Prediction) Only
HardwareLayers=[];
conv_memory1=[];
conv_memory2=[];

for i=1:size(Net.Layers,1)
    tmp_string=Net.Layers(i,1).Name;
    disp(tmp_string)
    
    if ismethod(Net.Layers(i,1),'BatchNormalizationLayer')
        epsilon=1.0e-10;
        Mean1d = Net.Layers(i, 1).TrainedMean;
        Variance1d = Net.Layers(i, 1).TrainedVariance;
        Scale1d = Net.Layers(i, 1).Scale;
        Offset1d =Net.Layers(i, 1).Offset;
        
        BatchBias1d=  (  sqrt(Variance1d).*Offset1d./(Scale1d+epsilon)  )  -Mean1d   ;
        Scale_sign=sign(Scale1d); %check Guinness paper, they were wrong about BatchNorm equation.
        BatchBias1d=Scale_sign.*BatchBias1d; %For Right assumption, cross their own sign.
        
        HardwareLayers(conv_memory2,1).Weights= sign(Net.Layers(conv_memory1, 1).Weights)  ;
        HardwareLayers(conv_memory2,1).Bias= floor(Net.Layers(conv_memory1,1).Bias+BatchBias1d)  ;
        
        conv_memory1=[];
        conv_memory2=[];
        
    elseif ismethod(Net.Layers(i,1),'BinarizedConvolution2DLayerFixer')
        
        HardwareLayers=[
            HardwareLayers;
            convolution2dLayer(Net.Layers(i,1).FilterSize, ...
            Net.Layers(i,1).NumFilters, ...
            'Padding',Net.Layers(i,1).PaddingSize, ...
            'Stride',Net.Layers(i,1).Stride, ...
            'Name',Net.Layers(i,1).Name)
            ];
        conv_memory1=i;
        conv_memory2=size(HardwareLayers,1);
        
    elseif ismethod(Net.Layers(i,1),'ImageInputLayer')
        HardwareLayers=[
            HardwareLayers;
            imageInputLayer([128 128 3],'Name','input','Normalization','none')
            ];
        
    elseif ismethod(Net.Layers(i,1),'AveragePooling2DLayer')
        HardwareLayers=[
            HardwareLayers;
            averagePooling2dLayer(16,'Name','avePool1')
            ];
        
    else
        HardwareLayers=[
            HardwareLayers;
            Net.Layers(i,1)
            ];
    end
end

% HardwareLayers=[
%     HardwareLayers(1,1);
%     %     Floorer('floor');
%     HardwareLayers(2:end,1)
%     ];

% plot(layerGraph(HardwareLayers))

end
