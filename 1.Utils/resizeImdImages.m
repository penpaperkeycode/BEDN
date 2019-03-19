function imds = resizeImdImages(imds, imageFolder,imageSize)
% Resize images to [360 480].

if ~exist(imageFolder,'dir')
    mkdir(imageFolder)
else
    imds = imageDatastore(imageFolder,'IncludeSubfolders',true,'LabelSource','foldernames');
    return; % Skip if images already resized
end

reset(imds)
while hasdata(imds)
    % Read an image.
    [I,info] = read(imds);
    
    % Resize image.
    I = imresize(I,imageSize(1,1:2));
    
    % Write to disk.
    
    imageName=[imageFolder,'\',char(info.Label)];
    
    if ~exist(imageName,'dir')
        mkdir(imageName)
    end
    [~, filename, ext] = fileparts(info.Filename);
    imwrite(I,[imageName '\' filename ext])
end

imds = imageDatastore(imageFolder,'IncludeSubfolders',true,'LabelSource','foldernames');
end
