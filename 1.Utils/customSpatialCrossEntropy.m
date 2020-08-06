function loss = customSpatialCrossEntropy( Y, T )

numObservations = size(Y, 4) * size(Y, 1) * size(Y, 2);
loss_i = T .* log(customBoundAwayFromZero(single(Y)));
loss = -sum( sum( sum( sum(loss_i, 3).*(1./numObservations), 1), 2));  
end