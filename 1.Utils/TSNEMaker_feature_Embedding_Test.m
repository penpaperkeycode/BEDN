imds = imageDatastore('E:\Datasets\Stanford_Dog_Dataset\Images', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
    
inputSize = [32 32];
imds.ReadFcn = @(loc)imresize(imread(loc),inputSize);
% %============================:Raw Image Flatten:========================%
% featuresOut=[];
% tic
% for i=1:size(Xdata,4)
%     featuresOut = [
%         featuresOut;
%         reshape(permute(Xdata(:,:,:,i),[1 2 3]),1,[]);
%         ];
%     percentage=(i/size(Xdata,4))*100
%     
% end
% featuresOut=single(featuresOut);
% toc

featuresOut2=activations(OriginNet,read(imds),OriginNet.Layers(23,1).Name,'OutputAs','rows');
%============================:Feature Extraction:======================%
featuresOut=[];
tic
for i=1:size(Xdata,4)
    featuresOut = [
        featuresOut;
        activations(OriginNet,Xdata(:,:,:,i),OriginNet.Layers(23,1).Name,'OutputAs','rows')
        ];
    percentage=(i/size(Xdata,4))*100
    
end
toc

rng default % for reproducibility
Y2 = tsne(featuresOut2,'Algorithm','barneshut','Perplexity',50,'NumPCAComponents',128);
figure
gscatter(Y2(:,1),Y2(:,2),imds.Labels)
grid on


%=============================:Sample tSNE:=========================%
Num_set=80;

Num_people_start=803;
Num_people_end=853; %853

featuresOut_sample=[];
for i=Num_people_start:Num_people_end
    for j=1:Num_set
        featuresOut_sample=[featuresOut_sample;featuresOut(j+(i-1)*500,:)];
    end
end

Ydata_sample=[];
for i=Num_people_start:Num_people_end
    for j=1:Num_set
        Ydata_sample=[Ydata_sample;Ydata(j+(i-1)*500,:)];
    end
end

rng default % for reproducibility
Y = tsne(featuresOut_sample,'Algorithm','barneshut','Perplexity',20,'NumPCAComponents',128);
% Y = tsne(featuresOut_sample,'Algorithm','barneshut','Perplexity',5);
figure
gscatter(Y(:,1),Y(:,2),Ydata_sample)
grid on

%=====================================================================%


n1=200;
n2=50;

rng default % for reproducibility
Y = tsne(featuresOut(853*(n1-1)+1:853*(n1+n2),:),'Algorithm','barneshut','Perplexity',50,'NumPCAComponents',50);
% figure
% gscatter(Y(:,1),Y(:,2),Ydata(853*(n1-1)+1:853*n1*n2,:))
% grid on

figure
gscatter(Y(400*(n1-1)+1:400*(n1+n2),1),Y(400*(n1-1)+1:400*(n1+n2),2),Ydata(400*(n1-1)+1:400*(n1+n2),:))
grid on


% figure(1)
% grid on
% hold on
% for n=1:100
%     
%     gscatter(Y(835*(n-1)+1:835*n,1),Y(835*(n-1)+1:835*n,2),Ydata(835*(n-1)+1:835*n,:))
% end



% Y3 = tsne(featuresOut,'Algorithm','barneshut','NumPCAComponents',50,'NumDimensions',3);
% figure
% scatter3(Y3(:,1),Y3(:,2),Y3(:,3),15,YValidation,'filled');
% view(-93,14)
% grid on


%%%%%%%%%%%%%%%%%%% : Integer Featrue for NM Test : %%%%%%%%%%%%%%%%%%
i=21;
Net=FaceSyncNet7_2;

epsilon=1.0e-11;
Mean1d = Net.Layers(i, 1).TrainedMean;
Variance1d = Net.Layers(i, 1).TrainedVariance;
Scale1d = Net.Layers(i, 1).Scale;
Offset1d =Net.Layers(i, 1).Offset;
Scale_sign=sign(Scale1d);
gamma=Scale1d./(sqrt(Variance1d+epsilon));
gamma=reshape(gamma,1,256);
Scale_sign=reshape(Scale_sign,1,256);


featuresOut=[];
tic
for i=1:size(Xdata,4)
    featuresOut = [
        featuresOut;
        round(rescale((16.*Scale_sign.*gamma.*activations(SimNet,Xdata(:,:,:,i),SimNet.Layers(16,1).Name,'OutputAs','rows')),0,255))
        ];
    percentage=(i/size(Xdata,4))*100
    
end
toc