function lgraph=BinarizedSegnetArchitectureMaker_Unconv(net,imageSize)

% imageSize=[360 480 3]; %[240,352,3];%
% netWidth = 16;

maxpoolidx=[];
layerstmp=imageInputLayer(imageSize,'Name','input','Normalization','none') ;

%Extract Usable Layers from input network
for i=2:size(net.Layers,1)
    if ismethod(net.Layers(i,1),'AveragePooling2DLayer')
        break
    elseif ismethod(net.Layers(i,1),'MaxPooling2DLayer')
        maxpoolidx=[maxpoolidx;i];
        layerstmp=[layerstmp;net.Layers(i,1)];
    else
        layerstmp=[layerstmp;net.Layers(i,1)];
    end
end


%Making reversed layers
reverselayerstmp=[];
% for i=2:size(layerstmp,1)

numChannels=256;

for i=size(layerstmp,1):-1:2
    tmplayer=[];
    if ismethod(layerstmp(i,1),'MaxPooling2DLayer')
        tmplayer=maxUnpooling2dLayer('Name',['Un',layerstmp(i,1).Name]);
    elseif ismethod(layerstmp(i,1),'BinarizedConvolution2DLayerFixer')
        for n=0:1:2
            if ismethod(layerstmp(i+n,1),'BinarizedConvolution2DLayerFixer')%n==0
                tmplayer=[tmplayer;
                    BinarizedtransposedConv2dLayer(layerstmp(i+n,1).FilterSize, ...
                    layerstmp(i+n,1).NumFilters, ...
                    'Stride',2,'BiasLearnRateFactor',0, ...
                    'Name',['Transposed',layerstmp(i+n,1).Name])
                    ];
%                                     'NumChannels', numChannels, ...
%                     'Cropping',1, ...
%                 numChannels=layerstmp(i+n,1).NumFilters;
                %                                 tmplayer=[tmplayer;
                %                     Binarizedconvolution2dLayer(layerstmp(i,1).FilterSize, ...
                %                     layerstmp(i,1).NumFilters, ...
                %                     'Padding',layerstmp(i,1).PaddingSize, ...
                %                     'Stride',layerstmp(i,1).Stride,...
                %                     'Name',[layerstmp(i,1).Name,'_rvrs'])
                %                     ];
%                 if ~ismethod(layerstmp(i-1,1),'MaxPooling2DLayer')
%                     tmplayer(end,1).Weights=layerstmp(i+n,1).Weights;
%                 end
            elseif ismethod(layerstmp(i+n,1),'BatchNormalizationLayer') %n==1
                tmplayer=[tmplayer;
                    batchNormalizationLayer('Name',['Transposed',layerstmp(i+n,1).Name])
                    ];
            else %n==2
                tmplayer=[tmplayer;layerstmp(i+n,1)];
                tmplayer(end,1).Name=['Transposed',tmplayer(end,1).Name];
            end
            
        end
    end
    reverselayerstmp=[reverselayerstmp;tmplayer;];
end


% %remove first unpool
% reverselayerstmp=reverselayerstmp(2:end,1);
% reverselayerstmp(end+1,1)=maxUnpooling2dLayer('Name','UnMaxPoolLast');

tailLayer=[    
    Binarizedconvolution2dLayer(3,11,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConvTail')
    batchNormalizationLayer('Name','BatchNormTail')
    softmaxLayer('Name', 'softmax')
    pixelClassificationLayer('Name', 'pixelLabels')
    ];

layers=[
    layerstmp;
    reverselayerstmp;
    tailLayer
    ];


lgraph=layerGraph(layers);
plot(lgraph)


end
% 
% function layerNames = iGetLayerNames(layers, type)
% isOfType = arrayfun(@(x)isa(x,type),layers,'UniformOutput', true);
% layerNames = {layers(isOfType).Name}';
% end
% 
% function idx = iFindLayer(layers, type)
% results = arrayfun(@(x)isa(x,type),layers,'UniformOutput', true);
% idx = find(results);
% end
