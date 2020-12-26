
filename='Origin';

cmap = camvidColorMap;

for i=0:5
    I= imread(imdsTest.Files{end-i});
%     C = imread(pxdsTest.Files{end-i});
    
%     B = labeloverlay(C,C,'Colormap',cmap,'Transparency',0);
    fig=figure(1);
    imshow(I)
    
    saveas(fig,[filename,'_',num2str(i),'.fig'])
    saveas(fig,[filename,'_',num2str(i),'.eps'])
    saveas(fig,[filename,'_',num2str(i),'.png'])
end