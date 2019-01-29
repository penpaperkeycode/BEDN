
% testImage=XTrain(:,:,:,1);
% testImage=XValidation(:,:,:,7);
testImage=imresize(imread('cars3.jpg'),[128,128]);

featureMap = activations(SimNet, testImage, SimNet.Layers(17, 1).Name  );
% montage(featureMap)
WeightFCN=SimNet.Layers(20, 1).Weights;
% size(WeightFCN)
% 
% ans =
% 
%      1     1   256    10

Weight_cat=WeightFCN(:,:,:,2);
 
tmp=Weight_cat.*featureMap;
tmptmp=sum(tmp,3);
[height, width, ~] = size(testImage);

actMapOrign = imresize(tmptmp, [height, width]);

tmptmp(tmptmp<120)=0;

actMap = imresize(tmptmp, [height, width]);

BW = imregionalmax(actMap);

featureMapOnImage = imfuse(testImage, actMap);
tmpin=rgb2gray(featureMapOnImage);
propied=regionprops(actMap,'BoundingBox');
bbx = vertcat(propied.BoundingBox);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xmin = bbx(:,1);
ymin = bbx(:,2);
xmax = xmin + bbx(:,3) - 1;
ymax = ymin + bbx(:,4) - 1;

% Expand the bounding boxes by a small amount.
expansionAmount = 0.02;
xmin = (1-expansionAmount) * xmin;
ymin = (1-expansionAmount) * ymin;
xmax = (1+expansionAmount) * xmax;
ymax = (1+expansionAmount) * ymax;

% Clip the bounding boxes to be within the image bounds
xmin = max(xmin, 1);
ymin = max(ymin, 1);
xmax = min(xmax, size(testImage,2));
ymax = min(ymax, size(testImage,1));

% Show the expanded bounding boxes
expandedBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];
IExpandedBBoxes = insertShape(testImage,'Rectangle',expandedBBoxes,'LineWidth',3);

imshow(IExpandedBBoxes)
expandedBBoxes(expandedBBoxes<1)=1;
overlapRatio = bboxOverlapRatio(expandedBBoxes, expandedBBoxes);

% Set the overlap ratio between a bounding box and itself to zero to
% simplify the graph representation.
n = size(overlapRatio,1); 
overlapRatio(1:n+1:n^2) = 0;

% Create the graph
g = graph(overlapRatio);

% Find the connected text regions within the graph
componentIndices = conncomp(g);

xmin = accumarray(componentIndices', xmin, [], @min);
ymin = accumarray(componentIndices', ymin, [], @min);
xmax = accumarray(componentIndices', xmax, [], @max);
ymax = accumarray(componentIndices', ymax, [], @max);

% Compose the merged bounding boxes using the [x y width height] format.
textBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];

% Remove bounding boxes that only contain one text region
numRegionsInGroup = histcounts(componentIndices);
textBBoxes(numRegionsInGroup == 1, :) = [];

% Show the final text detection result.
ITextRegion = insertShape(testImage, 'Rectangle', textBBoxes,'LineWidth',3);

figure
imshow(ITextRegion)
title('Detected Text')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% RGB = insertShape(testImage,'Rectangle',bbx,'LineWidth',1);

RGB = insertObjectAnnotation(testImage,'rectangle',bbx,...
        '','Color','r');
figure
imshow(RGB)
figure
imshow(featureMapOnImage)
% imshow(featureMapOnImage)


% {'airplane';'automobile';'bird';'cat';'deer';'dog';'frog';'horse';'ship';'truck'}