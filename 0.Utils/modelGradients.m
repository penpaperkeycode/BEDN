function [gradients,state,loss] = modelGradients(dlnet,dlX,Y)

[dlYPred,state] = forward(dlnet,dlX);
Y=reshape(Y,1,1,10,1000);
loss = crossentropy(dlYPred,Y);
gradients = dlgradient(loss,dlnet.Learnables);

end