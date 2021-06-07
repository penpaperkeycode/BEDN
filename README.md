# Binarized Neural Networks on MATLAB

"Binarized Encoder-Decoder Network and Binarized Deconvolution Engine for Semantic Segmentation"
(Kim, H., Kim, J., Choi, J., Lee, J., & Song, Y. H. (2020). Binarized Encoder-Decoder Network and Binarized Deconvolution Engine for Semantic Segmentation. IEEE Access.)
paper link: https://ieeexplore.ieee.org/document/9311614

Application: CIFAR10 Classification & CamVid Segmentation.

If you want a reproduce BEDN that I uploaded, follow below:
1. load('BEDN_11.mat', 'OriginNet');
2. layers=AutoConstructHWLayer(OriginNet);
3. lgraph=layerGraph(layers);
4. SimNet=DAGNetwork.loadobj(lgraph);
5. featureMap = activations(SimNet, I, SimNet.Layers(5, 1).Name,'ExecutionEnvironment','gpu' ); %5 is 'binConv2', I is any images with 360x480x3 size

SimNet is final BEDN, OriginNet is a pure model state without final binarization after training. (OriginNet for reuse to any other applications)

*Don't forget to "add to path" before running the script.  
*Do not add both "0.Methods" at the same time.  
