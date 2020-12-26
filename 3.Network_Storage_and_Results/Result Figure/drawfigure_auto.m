
filename='Binarized_SegNet_VGG16';

cmap = camvidColorMap;

for i=0:5
    I= imread(imdsTest.Files{end-i});
    C = semanticseg(I, OriginNet);
    
    B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0);
    fig=figure(1);
    imshow(B)
    
    saveas(fig,[filename,'_',num2str(i),'.fig'])
    saveas(fig,[filename,'_',num2str(i),'.eps'])
    saveas(fig,[filename,'_',num2str(i),'.png'])
end