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

## Discussion 4 Batch Normalization - 2015
1. What is this paper try to convey ?
- BN helps the network train faster amd achieve higher accuracy.
- Eliminating internal coveriat shift helps model train gaster with better performacne
- Transforming activation distributions to standatd normals accelerates amd strengthens learning
2. Why do use many mini-batch
- more accurate 
3. What is covariate shift 
- how coreelation to random varaible are 
- covariance : you got some random variable that depends on 
4. What is internal covariate shift
- change of distribution during learning
5. What is equation l = F2(F1(u,theta1),theta2)
- l depends on distribution of x that why it called covariance shift
6. how would it help in accelerate normalize to saturate non-linearity
- saturate non-linearity
7. What is whitening 
- transforming to have mean zero and unit variances. PCA
8. What about bias during normalization
- changing b is no effect but calculaing gradient b, u get some varaible then change b by some amount. it doesn't effect loss remain
9. what is equation mean x = Norm(x,X)
- In some situations, normalize entire the data set.
10. What is wrong with whitening 
-  high computation : a sqaure of nubmer of the layers.
- z-scaling
11. What the main gamma and beta
- each input 
- scale and shift with limited region
12. explain patial derivative  
- 
13. What 
- 
14. what is batch norm go 
- go before the non-linearity
15. BN claim that enable u higher learning rate  
- increase the scale of layer parameters
16. BN does not need drop out
- BN introducr random 

## Discussion 5 Adam - 2014
- Adam is invaraint to diagonal rescaling of the gradients
- sparse gradient is a gradient vector in which most of the elements have a value of zero which benefit to reduce the amount of computation required and improve the training speed.

## DenseNet - 2017