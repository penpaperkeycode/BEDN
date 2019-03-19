BNNpath='C:\Works\NeuralNetworks\Matlab2018b_BNN';
addpath(genpath(BNNpath))

%===========================:Prepare Data:===========================%
imageSize = [128 128 3] ; %[360 480 3];

% imageFolder='C:\Works\NeuralNetworks\Datasets\CustomVGGFace2\Class300\imagesize128\trainset';
% validesetFolder='C:\Works\NeuralNetworks\Datasets\CustomVGGFace2\Class300\imagesize128\testset';
imageFolder='C:\Works\NeuralNetworks\Datasets\CustomVGGFace2\Class150\imagesize128\trainset';
validesetFolder='C:\Works\NeuralNetworks\Datasets\CustomVGGFace2\Class150\imagesize128\testset';

imds = imageDatastore(imageFolder,'IncludeSubfolders',true,'LabelSource','foldernames');
testset_imds = imageDatastore(validesetFolder,'IncludeSubfolders',true,'LabelSource','foldernames');
catsize=size(unique(imds.Labels),1);

% load('C:\Works\NeuralNetworks\SpecDecision\Class150Layer12Image32Channel256\OriginNet.mat')

HardwareLayers=AutoConstructHWLayer(VGGFace2BNet16);
SimNet=SeriesNetwork(HardwareLayers);

%============================:Feature Extraction:======================%
featuresOut=[];
tic
for i=1: size(testset_imds.Labels,1)
    
    tmpimg=readimage(testset_imds,i);
    
    featuresOut = [
        featuresOut;
        activations(SimNet,tmpimg,SimNet.Layers(30,1).Name,'OutputAs','rows')
        ];
    percentage=(i/size(testset_imds.Labels,1))*100
    
end
toc

rng default % for reproducibility
Y = tsne(featuresOut,'Algorithm','barneshut','Perplexity',50,'NumPCAComponents',128);
figure
gscatter(Y(:,1),Y(:,2),testset_imds.Labels)
grid on


