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
1. What is this paper try to convey ?
- Adam is invaraint to diagonal rescaling of the gradients 
- Optimization with momentum and adaptive step size outperform other methods
2. First order and higher order
- higher order
3. why first order is suitable to large scale optimization
- loss function respect to each functions
4. SGD noisy
- dropout, mini-btach
5. Adam is
- Adaptive Moment Estimation
6. What is expectation of E[f(/theta)]
- mean sampliing, population mean f() = cost function
- experiment perform the different function
7. What is f1, f2, and so on
- loss at the timestep
8. what is gt
- gradient with specific timestep
9. what is mt
- moving average of gradient
10. what is vt
- moving average of the squared gradient
11. what is bias
- average something with 0 add bias
12. update rule
- 
13. If B is larger
- decay rate are small
14. If epsilon is 0
- gradient will be same
15. minist experiment
- adam covnerges as fast as Adagard
- sparse feature
- learing rate

## Discussion 5 DenseNet - 2017
1. What is this paper try to convey ?
- the preceding layers improve performance and reduce parameters
2. DenseNet characteristic
- skipping connection 
3. What is DenseNet to 
- concentanate from previous output
4. what is k
- number of feature map 
5. transalation layer
- add projection
6. DenseNet comparing with ResNet
- ResNet have to learn previous block otherwise DenseNet becuase they concatenate which improve flow training
7. x = H(x0,x1,...xl-1)
- H is composite function : BN, ReLU, pooling 
8. why adding pooling, conv
- down-sampling , dimension reduction
9. Table 2
-
10. 



## Discussion 6 YOLOv1 - 2016
1. What is this paper try to convey ?
- end-to-end optimization of CNN detection performance yield fast.
2.  
- train seperate past and well not optimize
3. single regression problem 
- bounding box coordinates and class probability
4. Titan X
- 45 fps 
5. YOLO less error background
- 
6. Art Work
-
7. What is each grid respond for
- For predicting particular center point
8. 
-
9. class probability
-
10. boundiing box consist of
- 
11. why only + C
- this is a problem when grid is hesitate that image is dog or bicycle. it cannot predict what it is.
12. pretrain
- the first 20 convolutional layers followed by a average-pooling layer and a fully connected layer
13. 
- one dimension to 7x7x30
14. how 7x7x30 bound  
- predicts both class probabilities and bounding box coordinates
15. activation
- leaky ReLU
16. limitation
- large 
17. Fast YOLO
- fewer layer (9 instead of 24)
18. Fast R-CNN vs YOLO
- Fast R-CNN perform localize better than YOLO but fail in background 3 times of YOLO instead

## Discussion 6 YOLOv4 - 2020
1. What is this paper try to convey ?
carefully slection detector training and inference modules greatly improves performance.
2. stand-alone process management and human input reduction 
- 
3. single GPU training
- fast enough
4. data augmentation
-
5. bag of freebies
-
6. class imbalance
- hard negative example mining and online hard hard example mining
- focal loss 
7. negative class 
- dont have object detection
8. association
- soft label
- one-hot label
8. mish activation
- smoothly transition

## Discussion 7 YOLOR - 2021
1. What is this paper try to convey ? 
- a unified network which serve various tasks
2. explicit knowledge
- shallow layers
3. implicit learning
- bike bicycle
4. general representation
- some informations are throwed away
- CNN transformation -> multi-task 
5. knowleadge modelling
- embedding, sparse, memory
6. implicit knowledge work
- tensor
7. Manifold space reduction
- pose check in x axis 
- classififcation in y axis
8. Kernel space alignment
- due to multi-task each output wil not align together
9. More function
10. conventional network
11. panoptic segmentation
12. Feature alignment
- Kernel space 
13. Prediction refinement
- add tensor to yolo output vector
14. 

## Discussion 8 Mask RCNN - 2017
1. What is this paper try to convey ? 
- Faster rcnn with mask branch perform instance segmentation 
2. segmentation
3. Mask branch
- predict an mxm mask from each ROI using an FCN
4. RPN
5. ROIPool 
6. ROIAlign is better than ROIPool
7. Lmask
- sigmoid and binary

## Discussion 9 YOLACT - 2019
1. What is this paper try to convey ? 
- parallel subtask is given faster real-time instance segemtation
2. Why other take a long time
- RCN and ROIpool take a long possessing
3. How YOLACT fix
- generating prototype mask and coefficient mask
4. How fast as it inference time
- 5 ms to evaluate
5. YOLOACT architecture
6. Rationale
- coherent work in convolutional layer
ึ7. Protonet Architecture
- after P3 
8. k
- number of prototype mask
9. ReLU in protonet
- To be unbounded, interpretable prototypes stricly positive size
10. Why tanh to k mask coeffient
- extra bonus from positive 
11. Why sigmoid in final mask
- it give keep range 0-1. it easier of mask
12. Mask predict one by one how YOLOACT work
- mask produce coefficient predict anchor box
13. learns how to localize instances on its own via different activations in its prototypes
- 
14. YOLOACT
- 5 different scale 24,48,96,192,384
15. FAST NMS
- sequential in IoU then sort descending by score 
16.
