function HardwareLayers=AutoConstructHWLayer(Net)
% Input = Binarized Neural Network
% Output = Hardware Neural Network for Classification (Prediction) Only
HardwareLayers=[];
conv_memory1=[];
conv_memory2=[];

for i=1:size(Net.Layers,1)
    tmp_string=Net.Layers(i,1).Name;
    disp(tmp_string)
    
    if ismethod(Net.Layers(i,1),'BatchNormalizationLayer')
        epsilon=1.0e-11;
        Mean1d = Net.Layers(i, 1).TrainedMean;
        Variance1d = Net.Layers(i, 1).TrainedVariance;
        Scale1d = Net.Layers(i, 1).Scale;
        Offset1d =Net.Layers(i, 1).Offset;
        
        BatchBias1d=(  (  sqrt(Variance1d).*Offset1d./(Scale1d+epsilon)  )  -Mean1d   );
        Scale_sign=sign(Scale1d); %check Guinness paper, they were wrong about BatchNorm equation.
        BatchBias1d=Scale_sign.*BatchBias1d; %For Right assumption, cross their own sign.

        HardwareLayers(conv_memory2,1).Weights= sign(Net.Layers(conv_memory1, 1).Weights)  ;
        HardwareLayers(conv_memory2,1).Bias= floor(Net.Layers(conv_memory1,1).Bias+BatchBias1d ) ;

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
        
    else
        HardwareLayers=[
            HardwareLayers;
            Net.Layers(i,1)
            ];
    end
end

% HardwareLayers=[
%     HardwareLayers(1,1);
%     FloorLayer('floor')
%     HardwareLayers(2:end,1)
%     ];

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