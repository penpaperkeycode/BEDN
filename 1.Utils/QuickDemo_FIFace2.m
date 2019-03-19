
while(1000000)
    
    img = snapshot(cam);
%  img = flipdim(img,2);
    imshow(img)
     
        img=imresize(img,[128,128]);
        YValPredHWtmp= classify(SimNet,img)
        

end