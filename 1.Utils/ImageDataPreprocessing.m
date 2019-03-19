% imageFolder='C:\Works\NeuralNetworks\Datasets\vggface2_train\vggface2_train\train\labelsResized_128';

function ImageDatasetSpliter(imageFolder,numb_image,numb_testset)

MyFolderInfo = dir(imageFolder);

PreprocessedDir='C:\Works\NeuralNetworks\Datasets\CustomVGGFace2\imagesize128';
if ~exist(PreprocessedDir,'dir')
    mkdir(PreprocessedDir)
end

%make a list of folders which have above 500 images
folderlist=[];
for i=1:size(MyFolderInfo,1)-2
    localFolderinfo=dir([imageFolder,'\',MyFolderInfo(i+2).name]);
    if size(localFolderinfo,1)-2 >500
        folderlist=[folderlist;MyFolderInfo(i+2).name];
        for ii=1:50
            if ~exist([PreprocessedDir,'\testset\',MyFolderInfo(i+2).name],'dir')
                mkdir([PreprocessedDir,'\testset\',MyFolderInfo(i+2).name])
            end
            copyfile([localFolderinfo(1).folder,'\',localFolderinfo(ii+2).name],[PreprocessedDir,'\testset\',MyFolderInfo(i+2).name])
        end
        
        for iii=51:size(localFolderinfo,1)-2
            if ~exist([PreprocessedDir,'\trainset\',MyFolderInfo(i+2).name],'dir')
                mkdir([PreprocessedDir,'\trainset\',MyFolderInfo(i+2).name])
            end
            copyfile([localFolderinfo(1).folder,'\',localFolderinfo(iii+2).name],[PreprocessedDir,'\trainset\',MyFolderInfo(i+2).name])
        end
    end
end

end

