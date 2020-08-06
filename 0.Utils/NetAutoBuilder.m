function SimNet=NetAutoBuilder(NetSet,n,w,b)

InputSize=[n,n,3];
% Input = Binarized Neural Network Setting, Input dim (InputSize=[row, column, color]), weight bit, bias bit;
% NetSet= [Layer bit, Numb Channel, Filter size, Numb Filters];
% w= 0/1 and else == rand sign
% b= 0 -Max
%    1 +Max
%    2 zero
%    else rand integer
% Output = H/W Neural Network for Prediction Only


bit_ImageInputLayer = 0;
bit_intConv = 1;
bit_binConv = 2;
bit_MaxPool = 4;
bit_AvePool = 8;
bit_Affine  = 16;
bit_Softmax = 32;
bit_Classification= 64;
bit_Regression=128;
bit_AffineFin =256;

HardwareLayers=[];


for i=1:size(NetSet,1)
    
    if NetSet(i,1)==bit_ImageInputLayer
        CurrentLayer=imageInputLayer(InputSize,'Name',[num2str(i),'_Input'],'Normalization','none') ;
        
    elseif NetSet(i,1)==bit_intConv || NetSet(i,1)==bit_binConv
        
        CurrentLayer=convolution2dLayer(NetSet(i,3),NetSet(i,4),'Padding',[1,1,1,1],'Stride',1,'BiasLearnRateFactor',0,'Name',[num2str(i),'_Conv']);
        
        
        CurrentLayer=wbSupply(CurrentLayer,NetSet,i,b,w);
        
        CurrentLayer=[CurrentLayer;
            SignumActivation([num2str(i),'_Sign'])
            ];
        
    elseif NetSet(i,1)==bit_MaxPool
        
        CurrentLayer=maxPooling2dLayer(2,'Stride',2,'Name',[num2str(i),'_Maxpool']);
        
    elseif NetSet(i,1)==bit_AvePool
        
        howmany_maxpool=0;
        for j=1:size(HardwareLayers,1)
            if ismethod(HardwareLayers(j,1),'MaxPooling2DLayer')
                howmany_maxpool=howmany_maxpool+1;
            end
        end
        AveSize=floor(InputSize(1,1)/(2^howmany_maxpool));
        if AveSize==0
            AveSize=1;
        end
        
        
        CurrentLayer=averagePooling2dLayer(AveSize,'Name',[num2str(i),'_Avepool']);
        
        CurrentLayer=[CurrentLayer;
            SignumActivation([num2str(i),'_Sign'])
            ];
        
    elseif NetSet(i,1)==bit_Affine
        CurrentLayer=convolution2dLayer(NetSet(i,3),NetSet(i,4),'Stride',1,'Name',[num2str(i),'_Affine'],'BiasLearnRateFactor',1,'BiasL2Factor',1,'Padding',[0,0,0,0]) ;
        CurrentLayer=wbSupply(CurrentLayer,NetSet,i,b,w);
        CurrentLayer=[CurrentLayer;
            SignumActivation([num2str(i),'_Sign'])
            ];
        
    elseif NetSet(i,1)==bit_AffineFin
        CurrentLayer=convolution2dLayer(NetSet(i,3),NetSet(i,4),'Stride',1,'Name',[num2str(i),'_Affine'],'BiasLearnRateFactor',1,'BiasL2Factor',1,'Padding',[0,0,0,0]) ;
        CurrentLayer=wbSupply(CurrentLayer,NetSet,i,b,w);
    elseif NetSet(i,1)==bit_Softmax
        CurrentLayer=softmaxLayer('Name',[num2str(i),'_Softmax']);
        
    elseif NetSet(i,1)==bit_Classification
        CurrentLayer=classificationLayer('Name',[num2str(i),'_Classoutput']);
        
    elseif NetSet(i,1)==bit_Regression
        CurrentLayer=regressionLayer('Name',[num2str(i),'_Regressionoutput']);
    end
    
    HardwareLayers=[
        HardwareLayers;
        CurrentLayer
        ];
end



SimNet=SeriesNetwork(HardwareLayers);

end

function CurrentLayer=wbSupply(CurrentLayer,NetSet,i,b,w)

if b==1
    CurrentLayer.Bias=double((intmax('int32'))*int32(ones(1,1,NetSet(i,4))));
elseif b==0
    CurrentLayer.Bias=double((intmin('int32'))*int32(ones(1,1,NetSet(i,4))));
elseif b==2
    CurrentLayer.Bias=(0*ones(1,1,NetSet(i,4)));
else
    
    lowerbound = -20000;
    upperbound = 20000;
    Bias_tmp = (upperbound-lowerbound).*rand(1,1,NetSet(i,4)) + lowerbound;
    
    CurrentLayer.Bias=double(round(Bias_tmp));
end

if w==1
    CurrentLayer.Weights=double(1*ones(NetSet(i,3),NetSet(i,3),NetSet(i,2),NetSet(i,4)));
elseif w==0
    CurrentLayer.Weights=double(-1*ones(NetSet(i,3),NetSet(i,3),NetSet(i,2),NetSet(i,4)));
else
    CurrentLayer.Weights=double(sign(randn(NetSet(i,3),NetSet(i,3),NetSet(i,2),NetSet(i,4))));
    CurrentLayer.Weights(CurrentLayer.Weights==0)=1;
end

end
