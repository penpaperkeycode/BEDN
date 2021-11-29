function [dlgradients,dlloss,state] = modelGradients_spatialentropy(dlnet,dlX,dlY)
[dlYPred,state] = forward(dlnet,dlX);
% dlYPred = softmax(dlYPred);

dlloss = customSpatialCrossEntropy(dlYPred,dlY);
dlgradients = dlgradient(dlloss,dlnet.Learnables);
end


