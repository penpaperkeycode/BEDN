

tmpX=testImage_tmp(:,:,3);
XValidation_BGR=cat(3,tmpX,testImage_tmp(:,:,2));
XValidation_BGR=cat(3,XValidation_BGR,testImage_tmp(:,:,1)); % RGB to BGR


tmp_all=[];
for i=1:size(XValidation_BGR,4)
    
    
    testImage=XValidation_BGR(:,:,:,i);
    tmp=(reshape(permute(testImage,[3 2 1 4]),1,[]));
    tmp_all=[tmp_all;tmp];
    
    
end

tabletmp=table(tmp_all,YValidation);
SortedData = sortrows(tabletmp,2);


for i=1:size(XValidation_BGR,4)
    strtmp=['test_img_',num2str(i-1),'.bin']; %'test_img_0~9999';
    fileID = fopen(strtmp,'w');
    
    fwrite(fileID,SortedData(i,1).Variables);
    fclose(fileID);
end
