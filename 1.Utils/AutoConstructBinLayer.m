function BinLayers=AutoConstructBinLayer(Net)
% Input = Binarized Neural Network
% Output = Hardware Neural Network for Classification (Prediction) Only
BinLayers=[];

numbsign=0;
for i=1:size(Net.Layers,1)
    tmp_string=Net.Layers(i,1).Name;
    disp(tmp_string)
    
    if ismethod(Net.Layers(i,1),'ReLULayer')
        
%         strtmp=['sign',num2str(numbsign)];
        strtmp=Net.Layers(i,1).Name;
        numbsign=numbsign+1;
        
        BinLayers=[
            BinLayers;
            SignumActivation(strtmp)
            ];
        
    elseif ismethod(Net.Layers(i,1),'Convolution2DLayer')
        
        BinLayers=[
            BinLayers;
            Binarizedconvolution2dLayer(Net.Layers(i,1).FilterSize, ...
            Net.Layers(i,1).NumFilters, ...
            'Padding',Net.Layers(i,1).PaddingSize, ...
            'Stride',Net.Layers(i,1).Stride, ...
            'Name',Net.Layers(i,1).Name)
            ];
        conv_memory1=i;
        conv_memory2=size(BinLayers,1);
        
        BinLayers(conv_memory2,1).Weights=Net.Layers(conv_memory1, 1).Weights  ;
        BinLayers(conv_memory2,1).Bias= Net.Layers(conv_memory1,1).Bias  ;
        
        
    else
        BinLayers=[
            BinLayers;
            Net.Layers(i,1)
            ];
    end
end

plot(layerGraph(BinLayers))

end
