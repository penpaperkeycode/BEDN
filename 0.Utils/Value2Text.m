function Value2Text(Value,Address,Name)

if size(Value,1)~=1
    Value = (reshape(permute(Value,[3 2 1 4]),1,[]))';
end

txtName=[Address,'\',Name,'.txt'];
dlmwrite(txtName,int32(Value),'newline','pc','precision',15)




end