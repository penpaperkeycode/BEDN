cam=webcam;
faceDetector = vision.CascadeObjectDetector();
bbox=[];
while(1000000)
    
    img = snapshot(cam);
    bbox            = step(faceDetector, img);
    
    
    if ~isempty(bbox)
                videoFrame = insertShape(img, 'Rectangle', bbox);
                imshow(videoFrame)
        b1=bbox(1,1);
        b2=bbox(1,2);
        
        b3=bbox(1,1)+bbox(1,3);
        %         if b3>size(img,1)
        %             b3=size(img,1);
        %         end
        b4=bbox(1,2)+bbox(1,4);
        %         if b4>size(img,2)
        %             b4=size(img,2);
        %         end
        
        
        img=img(b2:b4,b1:b3,:);
        %         imshow(img)
        img=imresize(img,[128,128]);
        YValPredHWtmp= classify(SimNet,img)
        
    else
        YValPredHWtmp=[];
    end
    
    bbox=[];
end