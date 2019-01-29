function Activation2Text(SimNet,Address,testImage_tmp)

for i=1:size(SimNet.Layers,1)
    
    %     txtName=[Address,'\Activations\',num2str(i),'_',SimNet.Layers(i, 1).Name,'.txt'];
    txtName=[Address,'\Activations\',SimNet.Layers(i, 1).Name,'.txt'];
    
    if ismethod(SimNet.Layers(i,1),'SignumActivation') %~isempty((strfind(SimNet.Layers(i,1).Name,'Sign')))
        featureMap = activations(SimNet, testImage_tmp, SimNet.Layers(i, 1).Name,'ExecutionEnvironment','gpu'  );
        Value = (reshape(permute(featureMap,[2 1 3 4]),size(featureMap,1)*size(featureMap,2),[]));
        
        Value(Value==-1)=0;
        Value=logical(Value);
        Value = fliplr(Value);
        hexval = binaryVectorToHex(Value);
        dlmwrite(txtName,hexval,'newline','pc','delimiter','')
        
    elseif ismethod(SimNet.Layers(i,1),'MaxPooling2DLayer')
        featureMap = activations(SimNet, testImage_tmp, [SimNet.Layers(i, 1).Name,'/out'],'ExecutionEnvironment','gpu'  );
        Value = (reshape(permute(featureMap,[2 1 3 4]),size(featureMap,1)*size(featureMap,2),[]));
        
        Value(Value==-1)=0;
        Value=logical(Value);
        Value = fliplr(Value);
        hexval = binaryVectorToHex(Value);
        dlmwrite(txtName,hexval,'newline','pc','delimiter','')
        
    elseif ismethod(SimNet.Layers(i,1),'MaxUnpooling2DLayer')
        featureMap = activations(SimNet, testImage_tmp, SimNet.Layers(i, 1).Name,'ExecutionEnvironment','gpu'  );
        Value = (reshape(permute(featureMap,[2 1 3 4]),size(featureMap,1)*size(featureMap,2),[]));
        
        Value(Value==-1)=0;
        Value=logical(Value);
        Value = fliplr(Value);
        hexval = binaryVectorToHex(Value);
        dlmwrite(txtName,hexval,'newline','pc','delimiter','')
        
    else
        featureMap = activations(SimNet, testImage_tmp, SimNet.Layers(i, 1).Name,'ExecutionEnvironment','gpu'  );
        Value = (reshape(permute(featureMap,[2 1 3 4]),size(featureMap,1)*size(featureMap,2),[]));
        dlmwrite(txtName,int32(Value),'newline','pc','delimiter','\t','precision',15)
    end
    
    
end





end