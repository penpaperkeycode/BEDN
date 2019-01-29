function lgraph=BinarizedSegnetLayers()
imageSize=[360 480 3];
netWidth = 16;
layers = [
    imageInputLayer(imageSize,'Name','input','Normalization','none') %32*32*3
    
    
    Binarizedconvolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv1')
    batchNormalizationLayer('Name','BatchNorm1')
    SignumActivation('Sign1')
    Binarizedconvolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv2')
    batchNormalizationLayer('Name','BatchNorm2')
    SignumActivation('Sign2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool1')
    
    Binarizedconvolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv3')
    batchNormalizationLayer('Name','BatchNorm3')
    SignumActivation('Sign3')
    Binarizedconvolution2dLayer(3,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv4')
    batchNormalizationLayer('Name','BatchNorm4')
    SignumActivation('Sign4')
    
    
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool2')
    
    Binarizedconvolution2dLayer(3,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv5')
    batchNormalizationLayer('Name','BatchNorm5')
    SignumActivation('Sign5')
    Binarizedconvolution2dLayer(3,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv6')
    batchNormalizationLayer('Name','BatchNorm6')
    SignumActivation('Sign6')
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv7')
    batchNormalizationLayer('Name','BatchNorm7')
    SignumActivation('Sign7')
    
    
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool3')
    
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv8')
    batchNormalizationLayer('Name','BatchNorm8')
    SignumActivation('Sign8')
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv9')
    batchNormalizationLayer('Name','BatchNorm9')
    SignumActivation('Sign9')
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv10')
    batchNormalizationLayer('Name','BatchNorm10')
    SignumActivation('Sign10')
    
    
    maxPooling2dLayer(2,'Stride',2,'Name','MaxPool4')
    maxUnpooling2dLayer('Name','UnPool1')
    
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv11')
    batchNormalizationLayer('Name','BatchNorm11')
    SignumActivation('Sign11')
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv12')
    batchNormalizationLayer('Name','BatchNorm12')
    SignumActivation('Sign12')
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv13')
    batchNormalizationLayer('Name','BatchNorm13')
    SignumActivation('Sign13')
    
    maxUnpooling2dLayer('Name','UnPool2')
    
    Binarizedconvolution2dLayer(3,16*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv14')
    batchNormalizationLayer('Name','BatchNorm14')
    SignumActivation('Sign14')
    Binarizedconvolution2dLayer(3,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv15')
    batchNormalizationLayer('Name','BatchNorm15')
    SignumActivation('Sign15')
    Binarizedconvolution2dLayer(3,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv16')
    batchNormalizationLayer('Name','BatchNorm16')
    SignumActivation('Sign16')
    
    maxUnpooling2dLayer('Name','UnPool3')
    
    Binarizedconvolution2dLayer(3,12*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv17')
    batchNormalizationLayer('Name','BatchNorm17')
    SignumActivation('Sign17')
    Binarizedconvolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv18')
    batchNormalizationLayer('Name','BatchNorm18')
    SignumActivation('Sign18')
    
    maxUnpooling2dLayer('Name','UnPool4')
    
    Binarizedconvolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv19')
    batchNormalizationLayer('Name','BatchNorm19')
    SignumActivation('Sign19')
    Binarizedconvolution2dLayer(3,11,'Padding','same','Stride',1,'BiasLearnRateFactor',0,'Name','binConv20')
    batchNormalizationLayer('Name','BatchNorm20')
    softmaxLayer('Name', 'softmax')
    pixelClassificationLayer('Name', 'pixelLabels')
    
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