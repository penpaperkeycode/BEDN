%===========================:Prepare Data:===========================%
%Download the CIFAR-10 data set.
datadir = tempdir;
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
helperCIFAR10Data.download(url,datadir);

%Load the CIFAR-10 images and use the CIFAR-10 test images for network validation.
[XTrain,YTrain,XValidation,YValidation] = helperCIFAR10Data.load(datadir);

load('G:\Work\NeuralNetwork\QuantizedNeuralNetwork\BinarizedNeuralNetwork\TheBNN\HWSyncNet\SyncNet7.mat')

HardwareLayers=AutoConstructHWLayer_ScaleFree(SyncNet7)
SimNet=SeriesNetwork(HardwareLayers);

WeightFCN=SimNet.Layers(19, 1).Weights;
Weight_face=WeightFCN(:,:,:,2);
opencheck=1;
height=32;
width=32;




figure('Units','normalized','Position',[0 0.1 0.2 0.4]);
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset;
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
% axis off
set(gca,'XTickLabel',[],'YTickLabel',[]) %Axis number off
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

for i=1:size(XValidation,4)
%     tic
%     [YValPred,probs] = classify(SimNet,XValidation(:,:,:,i));
%     toc
    
    strCat=findCat_CIFAR10(YValPred);
    numCat=TFChecker(YValPred,YValidation(i,1));
    
    prob = num2str(100*max(probs(1,:)),3);
    figure(1)
    imshow(XValidation(:,:,:,i))
    title([strCat,',  Confidence=',num2str(prob),',  ',numCat])
    pause(0.5)
end