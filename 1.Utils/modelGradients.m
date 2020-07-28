function [dlgradients,dlloss,state] = modelGradients(dlnet,dlX,dlY)
[dlYPred,state] = forward(dlnet,dlX);
dlYPred = softmax(dlYPred);

dlloss = crossentropy(dlYPred,dlY);
dlgradients = dlgradient(dlloss,dlnet.Learnables);
end