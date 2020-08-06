function dlYPred = modelPredictions(dlnet,dlX,miniBatchSize)

numObservations = size(dlX,4);
numIterations = ceil(numObservations / miniBatchSize);

numClasses = dlnet.Layers(11).OutputSize;
dlYPred = zeros(numClasses,numObservations,'like',dlX);

for i = 1:numIterations
    idx = (i-1)*miniBatchSize+1:min(i*miniBatchSize,numObservations);
    
    dlYPred(:,idx) = predict(dlnet,dlX(:,:,:,idx));
end

end