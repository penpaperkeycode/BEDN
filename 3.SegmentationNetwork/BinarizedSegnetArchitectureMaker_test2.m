function lgraph=BinarizedSegnetArchitectureMaker_test2(net,imageSize,classes)

% imageSize=[360 480 3]; %[240,352,3];%
% netWidth = 16;
catnumb=size(classes,1);
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

if ~ismethod(layerstmp(end,1),'MaxPooling2DLayer')
    layerstmp(end+1)= maxPooling2dLayer([2 2], 'Stride', 2, 'Name', 'MaxPoolLast', 'HasUnpoolingOutputs', true);
    maxpoolidx=[maxpoolidx;size(layerstmp,1)];
end


%Making reversed layers
reverselayerstmp=[];
reverseSize=maxpoolidx(end)-maxpoolidx(end-1);
% for i=2:size(layerstmp,1)

numChannels=256;

for i=size(layerstmp,1)-reverseSize:-1:2
    tmplayer=[];
    if ismethod(layerstmp(i,1),'MaxPooling2DLayer')
        tmplayer=maxUnpooling2dLayer('Name',['Un',layerstmp(i,1).Name]);
    elseif ismethod(layerstmp(i,1),'BinarizedConvolution2DLayerFixer')
        for n=0:1:2
            if ismethod(layerstmp(i+n,1),'BinarizedConvolution2DLayerFixer')%n==0
                tmplayer=[tmplayer;
                    Binarizedconvolution2dLayer(layerstmp(i+n,1).FilterSize, ...
                    layerstmp(i+n,1).NumFilters, ...
                    'NumChannels', numChannels, ...
                    'Padding','same', ...
                    'Stride',1,'BiasLearnRateFactor',0, ...
                    'Name',[layerstmp(i+n,1).Name,'_rvrs'])
                    ];
                
                numChannels=layerstmp(i+n,1).NumFilters;
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
                    batchNormalizationLayer('Name',[layerstmp(i+n,1).Name,'_rvrs'])
                    ];
            else %n==2
                tmplayer=[tmplayer;layerstmp(i+n,1)];
                tmplayer(end,1).Name=[tmplayer(end,1).Name,'_rvrs'];
            end
            
        end
    end
    reverselayerstmp=[reverselayerstmp;tmplayer;];
end

if ~ismethod(reverselayerstmp(end,1),'MaxUnpooling2DLayer')
    reverselayerstmp(end+1)= maxUnpooling2dLayer('Name','UnMaxPoolLast');
end

% %remove first unpool
% reverselayerstmp=reverselayerstmp(2:end,1);
% reverselayerstmp(end+1,1)=maxUnpooling2dLayer('Name','UnMaxPoolLast');

tailLayer=[
    Binarizedconvolution2dLayer(1,64,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConvTail1')
    batchNormalizationLayer('Name','BatchNormTail1')
    SignumActivation('SignTail1')
    Binarizedconvolution2dLayer(3,64,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConvTail2')
    batchNormalizationLayer('Name','BatchNormTail2')
    SignumActivation('SignTail2')
    Binarizedconvolution2dLayer(1,64,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConvTail3')
    batchNormalizationLayer('Name','BatchNormTail3')
    SignumActivation('SignTail3')
    
    Binarizedconvolution2dLayer(3,catnumb,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConvTail')
    batchNormalizationLayer('Name','BatchNormTail')
    softmaxLayer('Name', 'softmax')
    pixelClassificationLayer('Name', 'labels')
    ];

layers=[
    layerstmp;
    reverselayerstmp;
    tailLayer
    ];


% lgraph=layerGraph(layers);
% plot(lgraph)

maxPoolLayerIndices = iFindMaxPoolingLayer(layers);
maxPoolLayerID = iGetMaxPoolingLayerNames(layers);
unpoolLayerID  = flip(iGetUnpoolingLayerNames(layers));
assert(numel(maxPoolLayerID) == numel(unpoolLayerID));

% Change max pooling layers to have HasUnpoolingOutputs true
for i = 1:numel(maxPoolLayerID)
    currentMaxPooling = layers(maxPoolLayerIndices(i));
    poolSize = currentMaxPooling.PoolSize;
    stride = currentMaxPooling.Stride;
    name = currentMaxPooling.Name;
    padding = currentMaxPooling.PaddingSize;
    layers(maxPoolLayerIndices(i)) = maxPooling2dLayer(poolSize, 'Stride', stride, 'Name', name, 'Padding', padding, 'HasUnpoolingOutputs', true);
end

lgraph = layerGraph(layers);

% Connect all max pool outputs to unpooling layers
for i = 1:numel(maxPoolLayerID)
    lgraph = connectLayers(lgraph, [maxPoolLayerID{i} '/indices'], [unpoolLayerID{i} '/indices']);
    lgraph = connectLayers(lgraph, [maxPoolLayerID{i} '/size'], [unpoolLayerID{i} '/size']);
end

plot(lgraph)

end

function layerNames = iGetLayerNames(layers, type)
isOfType = arrayfun(@(x)isa(x,type),layers,'UniformOutput', true);
layerNames = {layers(isOfType).Name}';
end

function idx = iFindLayer(layers, type)
results = arrayfun(@(x)isa(x,type),layers,'UniformOutput', true);
idx = find(results);
end
function idx = iFindMaxPoolingLayer(layers)
idx = iFindLayer(layers, 'nnet.cnn.layer.MaxPooling2DLayer');
end
function layerNames = iGetUnpoolingLayerNames(layers)
layerNames = iGetLayerNames(layers, 'nnet.cnn.layer.MaxUnpooling2DLayer');
end
%--------------------------------------------------------------------------
function layerNames = iGetMaxPoolingLayerNames(layers)
layerNames = iGetLayerNames(layers, 'nnet.cnn.layer.MaxPooling2DLayer');
end