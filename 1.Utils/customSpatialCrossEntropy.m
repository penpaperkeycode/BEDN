function loss = customSpatialCrossEntropy( Y, T )

numObservations = size(Y, 4) * size(Y, 1) * size(Y, 2);
loss_i = T .* log(customBoundAwayFromZero(double(Y)));
loss = -sum( sum( sum( sum(double(loss_i), 3).*(1./numObservations), 1), 2));  

end