function FINNActivation2Text(SimNet,Address,I)
%SimNet = Binarized Network
%Address = file root
%I = input image
%1st conv layer output precision is '%3.7f' other is integer
ConvFlag=0;
TextoutputNameFlag=2; % 1 == Add numbering to text title, else no numbering
mkdir(Address)
mkdir([Address,'\Activations'])

for i=1:size(SimNet.Layers,1)
    if TextoutputNameFlag==1
        txtName=[Address,'\Activations\',num2str(i),'_',SimNet.Layers(i, 1).Name,'.txt'];
    else
        txtName=[Address,'\Activations\',SimNet.Layers(i, 1).Name,'.txt'];
    end
    if ismethod(SimNet.Layers(i,1),'SignumActivation')
        featureMap = activations(SimNet, I, SimNet.Layers(i, 1).Name,'ExecutionEnvironment','gpu'  );
        Value = (reshape(permute(featureMap,[2 1 3 4]),size(featureMap,1)*size(featureMap,2),[]));
        %Value = (reshape(permute(featureMap,[3 2 1 4]),size(featureMap,1)*size(featureMap,2)*size(featureMap,3),[]));
        Value(Value==-1)=0;
        Value=logical(Value);
        Value = fliplr(Value);
        hexval = binaryVectorToHex(Value);
        dlmwrite(txtName,hexval,'newline','pc','delimiter','')
        
    elseif ismethod(SimNet.Layers(i,1),'MaxPooling2DLayer')
        if SimNet.Layers(i,1).HasUnpoolingOutputs ==0
            featureMap = activations(SimNet, I, SimNet.Layers(i, 1).Name,'ExecutionEnvironment','gpu'  );
        else
            featureMap = activations(SimNet, I,[SimNet.Layers(i, 1).Name,'/out'],'ExecutionEnvironment','gpu'  );
        end
        Value = (reshape(permute(featureMap,[2 1 3 4]),size(featureMap,1)*size(featureMap,2),[]));
        Value(Value==-1)=0;
        Value=logical(Value);
        Value = fliplr(Value);
        hexval = binaryVectorToHex(Value);
        dlmwrite(txtName,hexval,'newline','pc','delimiter','')
        
    elseif ismethod(SimNet.Layers(i,1),'MaxUnpooling2DLayer')
        featureMap = activations(SimNet, I, SimNet.Layers(i, 1).Name,'ExecutionEnvironment','gpu'  );
        Value = (reshape(permute(featureMap,[2 1 3 4]),size(featureMap,1)*size(featureMap,2),[]));
        Value(Value==-1)=0;
        Value=logical(Value);
        Value = fliplr(Value);
        hexval = binaryVectorToHex(Value);
        dlmwrite(txtName,hexval,'newline','pc','delimiter','')
        
    elseif ismethod(SimNet.Layers(i,1),'ImageInputLayer')
        Value=I;
        if size(Value,1)~=1
            Value = (reshape(permute(Value, [2,1,3]),1,[]))';
        end
        dlmwrite(txtName,Value,'newline','pc')
        imwrite(I,[Address,'\Activations','\inputimage_justforwatch.jpg'])
        
    elseif ismethod(SimNet.Layers(i,1),'InputScaleLayer')
        featureMap = activations(SimNet, I, SimNet.Layers(i, 1).Name,'ExecutionEnvironment','gpu'  );
        if size(featureMap,1)~=1
            Value = (reshape(permute(featureMap, [2,1,3]),1,[]))';
        end
        %Value = (reshape(permute(featureMap,[2 1 3 4]),size(featureMap,1)*size(featureMap,2),[]));
        dlmwrite([Address,'\Activations\Scale1_realNumb.txt'],Value,'delimiter','','precision',16)
        Value_sfi=sfi(Value,8,7);  % Need Check here
        Value_bin=bin(Value_sfi);
        dlmwrite(txtName,Value_bin,'delimiter','','precision',8)
        
    elseif ismethod(SimNet.Layers(i,1),'Convolution2DLayer')
        if ConvFlag==0
            featureMap = activations(SimNet, I, SimNet.Layers(i, 1).Name,'ExecutionEnvironment','gpu'  );
            %Value = (reshape(permute(featureMap,[2 1 3 4]),size(featureMap,1)*size(featureMap,2),[]));
            Value = (reshape(permute(featureMap,[3 2 1 4]),size(featureMap,1)*size(featureMap,2)*size(featureMap,3),[]));
            dlmwrite(txtName,(Value),'newline','pc','delimiter','\t','precision','%3.8f')
            
            %For only accumulation values (minus bias)
            featureMap=featureMap-SimNet.Layers(i,1).Bias;
            %Value=(reshape(permute(featureMap,[2 1 3 4]),size(featureMap,1)*size(featureMap,2),[]));
            Value = (reshape(permute(featureMap,[3 2 1 4]),size(featureMap,1)*size(featureMap,2)*size(featureMap,3),[]));
            dlmwrite([txtName(1:end-4),'_minusBias.txt'],Value,'newline','pc','delimiter','\t','precision','%3.8f')
            ConvFlag=1;
        else
            featureMap = activations(SimNet, I, SimNet.Layers(i, 1).Name,'ExecutionEnvironment','gpu'  );
            Value = (reshape(permute(featureMap,[3 2 1 4]),size(featureMap,1)*size(featureMap,2)*size(featureMap,3),[]));
            dlmwrite(txtName,Value,'newline','pc','delimiter','\t')
            
            %For only accumulation values (minus bias)
            featureMap=featureMap-SimNet.Layers(i,1).Bias;
            Value = (reshape(permute(featureMap,[3 2 1 4]),size(featureMap,1)*size(featureMap,2)*size(featureMap,3),[]));
            dlmwrite([txtName(1:end-4),'_minusBias.txt'],Value,'newline','pc','delimiter','\t')
        end
        
    elseif ismethod(SimNet.Layers(i,1),'TransposedConvolution2DLayer')
        featureMap = activations(SimNet, I, SimNet.Layers(i, 1).Name,'ExecutionEnvironment','gpu'  );
        Value = (reshape(permute(featureMap,[3 2 1 4]),size(featureMap,1)*size(featureMap,2)*size(featureMap,3),[]));
        dlmwrite(txtName,Value,'newline','pc','delimiter','\t')
        
        %For only accumulation values (minus bias)
        featureMap=featureMap-SimNet.Layers(i,1).Bias;
        Value = (reshape(permute(featureMap,[3 2 1 4]),size(featureMap,1)*size(featureMap,2)*size(featureMap,3),[]));
        dlmwrite([txtName(1:end-4),'_minusBias.txt'],Value,'newline','pc','delimiter','\t')
        
    elseif ismethod(SimNet.Layers(i,1),'OutputScaleLayer')
        featureMap = activations(SimNet, I, SimNet.Layers(i, 1).Name,'ExecutionEnvironment','gpu'  );
        Value = (reshape(permute(featureMap,[2 1 3 4]),size(featureMap,1)*size(featureMap,2),[]));
        dlmwrite(txtName,Value,'newline','pc','delimiter','\t','precision',32)
        
    elseif i==size(SimNet.Layers,1)
        featureMap = activations(SimNet, I, SimNet.Layers(i, 1).Name,'ExecutionEnvironment','gpu'  );
        [~,SegmentationLabels]=max(featureMap, [], 3);
        Value = (reshape(permute(SegmentationLabels,[2 1 3 4]),size(SegmentationLabels,1)*size(SegmentationLabels,2),[]));
        dlmwrite([Address,'\Activations\','2DLabelOutput.txt'],Value,'newline','pc','delimiter','\t','precision',8)
    else
        featureMap = activations(SimNet, I, SimNet.Layers(i, 1).Name,'ExecutionEnvironment','gpu'  );
        Value = (reshape(permute(featureMap,[2 1 3 4]),size(featureMap,1)*size(featureMap,2),[]));
        dlmwrite(txtName,Value,'newline','pc','delimiter','\t','precision',16)
    end
    
    
end





end