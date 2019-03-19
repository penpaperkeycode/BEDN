% imageFolder='C:\Works\NeuralNetworks\Datasets\vggface2_train\vggface2_train\train\labelsResized_128';
% PreprocessedDir='C:\Works\NeuralNetworks\Datasets\CustomVGGFace2\imagesize128';
% numb_image=500;
% numb_testset=50;
function ImageDatasetSpliter(rawimageFolder,PreprocessedDir,numb_image,numb_testset)

MyFolderInfo = dir(rawimageFolder);

if ~exist(PreprocessedDir,'dir')
    mkdir(PreprocessedDir)
end

%make a list of folders which have above 500 images
folderlist=[];
for i=1:size(MyFolderInfo,1)-2
    
    localFolderinfo=dir([rawimageFolder,'\',MyFolderInfo(i+2).name]);
    
    if size(localFolderinfo,1)-2 >numb_image
        
        if ~exist([PreprocessedDir,'\testset\',MyFolderInfo(i+2).name],'dir')
            mkdir([PreprocessedDir,'\testset\',MyFolderInfo(i+2).name])
        end
        
        if ~exist([PreprocessedDir,'\trainset\',MyFolderInfo(i+2).name],'dir')
            mkdir([PreprocessedDir,'\trainset\',MyFolderInfo(i+2).name])
        end
        
        folderlist=[folderlist;MyFolderInfo(i+2).name];
        
        for ii=1:numb_testset
            copyfile([localFolderinfo(1).folder,'\',localFolderinfo(ii+2).name],[PreprocessedDir,'\testset\',MyFolderInfo(i+2).name])
        end
        
        for iii=numb_testset+1:size(localFolderinfo,1)-2
            copyfile([localFolderinfo(1).folder,'\',localFolderinfo(iii+2).name],[PreprocessedDir,'\trainset\',MyFolderInfo(i+2).name])
        end
    end
end

end

