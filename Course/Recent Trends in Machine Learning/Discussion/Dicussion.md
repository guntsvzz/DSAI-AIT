## Discussion 1 AlexNet - 2012
1. what is this paper try to convey ?
- GPU parallelization boosts CNN model performance.
- Large deep CNNS are achieve high accuracy on large datasets.
- GPU training enables large CNNs to learn large image datasets
2. Why do we need to use large datasets for ML model ?
- having large datset enablize to constant the large parameter in model
3. 
-  
4. 
- 
5. 
- 

## Discussion 2 GoogLeNet - 2014
1. What is this paper try to convey ?
- inception module improve computing in neural network
2. GoogLeNet vs AlexNet 
- GoogLeNet is faster, more accurate 
3. What is computional budget mean
-  1.5 billon multiply-adds at inference time (training time)
- less parameter, less overfitting
4. What this they do in CNN ?
- using stack more layer
5. how many layer in GoogLeNet ?
- 22-layers deep model 
6. What is Network-in-Network ?
- dimension reduction by applying 1x1 convolution layer
- result, increasing the depth and width (number of filter)
7. 1x1 convolution layer
- dimension reduction remove computational bottlenecks.
8. What is receptive field 
- indicate size of the region
- later layers, bigger receptive field
- we desgin highter-level features deeeper in model
9. why adding conv and max pooling 
- max pooling to capture difference position
10. Why adding up 1x1 conv before 3x3 conv or 5x5 conv ?
- reducing parameter in convolution
11. What is V and S in  GoogLeNet network with all the bells and whistles
- conv 1x1 + 1(s) does not effect 
12. mean subtraction 
- 0-255 pixel subtract from the input imagner mean value [123.68, 116.779, 103.939]
13. What is auxilliary classifier
- we dont expect this layer
14. Polyak averaging
- SGD to optimal but set new parameter.
15. How image classifer to detection 
- Selective Search give a bunch of box


## Discussion 3 ResNet - 2015