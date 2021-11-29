%======================:Evaluate Trained Network:=====================%
%Calculate the final accuracy on the training set
%(without data augmentation) and validation set. Plot the confusion matrix.
[YValPred,probs] = classify(SimNet,XValidation);
validationError = mean(YValPred ~= YValidation);
YTrainPred = classify(SimNet,XTrain);
trainError = mean(YTrainPred ~= YTrain);
disp("Training error: " + trainError*100 + "%")
disp("Validation error: " + validationError*100 + "%")


% SimNet, TripleLayerNet
Xdata=cat(4,XTrain,XValidation);
Ydata=cat(1,YTrain,YValidation);

[YtotalPred,totalprobs] = classify(SimNet,Xdata);
totalsetError = mean(YtotalPred ~= Ydata);
disp("Total set error: " + totalsetError*100 + "%")



figure
plotconfusion(YValidation,YValPred,'Validation Data')

%Display some test images together with their predicted classes
%and the probabilities of those classes.
figure
idx = randperm(size(XValidation,4),16);
for i = 1:numel(idx)
    subplot(4,4,i)
    imshow(XValidation(:,:,:,idx(i)));
    prob = num2str(100*max(probs(idx(i),:)),3);
    predClass = char(YValPred(idx(i)));
    title([predClass,', ',prob,'%'])
end


%======================:Visualizations:=======================%
% Extract the first convolutional layer weights
w = SimNet.Layers(2).Weights; %2 5 8

% rescale the weights to the range [0, 1] for better visualization
w = rescale(w);

figure
montage(w)

%for some test image
testImage=XValidation(:,:,:,1);
%Extract the activations from the softmax layer. These are the classification scores produced by the network as it scans the image.
featureMap = activations(SimNet, testImage, SimNet.Layers(17, 1).Name  );
montage(featureMap)

featureMapOnImageBox=[];
for i=1:size(featureMap,3)
    % The softmax activations are stored in a 3-D array.
%     size(featureMap);
    FaceMap = featureMap(:, :, i);
    % Resize stopSignMap for visualization
    [height, width, ~] = size(testImage);
    FaceMap = imresize(FaceMap, [height, width]);
    % Visualize the feature map superimposed on the test image.
    featureMapOnImage = imfuse(testImage, FaceMap);
    %     figure
    %     imshow(featureMapOnImage)
    featureMapOnImageBox=cat(4,featureMapOnImageBox,featureMapOnImage);

end

montage(featureMapOnImageBox)


% featureMap = imresize(featureMap, [height, width]);
% for f=1:size(featureMap,3)
%     featureMapOnImage = imfuse(testImage, featureMap(:,:,f));
%     figure
%     imshow(featureMapOnImage)
% end