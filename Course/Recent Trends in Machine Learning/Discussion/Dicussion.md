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
1. What is this paper try to convey ?
- Residual block prevent vansihing/exploding gradient when network is deeper
2. Why they are try to more deep rather than width. 
-  Because previous competition attempt to use 
3. What is vanishing/exploding gradient.
- vanishing gradient occur when 
- when activation is bigger over and over then exploding gradient
4. Traditional way to prevent vanishing gradient
- Normalization layer
5. What is degradation phenomenal 
- higher training error 
6. What if they want more deeper layer, what they want to do
- adding layers identity mapping
7. Explaining Figure 2 Residual learning
- 
- linear combinations
8. Skipping connection
- No one 
9. Why Residual block have same dimension
- If it doesn't equally, it cannot concatenation
10. 
- Use transformation to collect size
11. Where do use nonlinearity when
- add nonlinearity between layer except the last residual block
12. What the difference FC and Convolution
- FC layer output is a vector adding likely sqaure 
- Convolution multi-channel 
13. VGG-19
- mostly have 3x3 filters and stride 2 convolution, feature map 
14. 34-layer plain
- stride 2 con
15. dashline
- linear projection 
16. Figure 4 what happen
- Plain-18 layer validation lower than train, overfitting
- Resnet-18 is almost the same
- Resnet-34 validation overfitting add the last with straight line
17. What option they use 
- B
18. Figure 5 right why they adding 1x1 conv
- Dimension reduction
19. How can we justify,why inception thing that Bottleneck is not a good thing
- Inception try to keep concept of previous and current layer to continous by not removing anything inside
- Bottleneck will losing information 256-d to 64-d
20. Figure 6. CIFAR-10
-
21. Figure 7. Std of layer response
- Resnet much lower validation across different layer
22. How did they try in Object Detection 
- using using Faster R-CNN and replacing VGG-net to ResNet-101
- improve loss, improve accuaracy.