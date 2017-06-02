# **Traffic Sign Recognition** 

## Writeup Template

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization1.jpg "Visualization1"
[image2]: ./examples/visualization2.jpg "Visualization2"
[image3]: ./examples/visualization3.jpg "Visualization3"
[image4]: ./examples/grayscale.jpg "Grayscaling"
[image5]: ./examples/normalization.jpg "normalized"
[image6]: ./examples/shuffled.jpg "shuffled"
[image7]: ./examples/random_translation.jpg "translation"
[image8]: ./examples/rotation.jpg "rotation"
[image9]: ./examples/random_perspective.jpg "perspective"
[image10]: ./examples/intensity_adjustment.jpg "intensity"
[image11]: ./examples/random_warping.jpg "warping"
[image12]: ./examples/dataaugmentation.jpg "augmentation"
[image13]: ./examples/new_images.jpg "Traffic Signs"
[image14]: ./examples/top3softmax.jpg "softmax"
[image15]: ./examples/top5softmax.jpg "softmax5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my project.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). And all the outputs are included.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 samples.
* The size of the validation set is 4410 samples.
* The size of test set is 12630 samples.
* The shape of a traffic sign image is (32,32,3).
* The number of unique classes/labels in the data set is 43.

---

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

First of all, I plotted traffic signs by demostrating 24 random traffic sign images, as follows:

![alt text][image1]

Second, I plotted the histogram of single training set to demostrate the distribution of training data as follows:

![alt text][image2]

Then I draw a multi-histogram of all three datasets to show the distributions of training/validation/test data at the same time. You can see the training data and validation data are unbalanced among classes.

![alt text][image3]

---

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

In this section, my pre-processing to training data and validation data includes two parts: the traditional pre-processing and data augmentation pre-processing.

First of all, the traditional pre-processing contains grayscale-convert, data normalization and data shuffle. 

- I did grayscale-convert because according to LeNet and AlexNet, this color transition works well for the classification performance, and it can reduce the training time, which is very important in practice. Here the first and third rows are original images, and the second and forth rows are corresponding grayscale images. And the shape of original training data is (34799,32,32,3) and converted gray training data is (34799,32,32,1).

![alt text][image4]


- I did data normalization because data scaling and feature normaliztion are basic and essential steps in machine learning, which helps the classifier converge fast. Here the data normalization changes the (mean, standard deviation) of training/validation/test sets from (82.677589037,66.0097957522),(83.5564273756,67.9870214471)and(82.1484603612, 66.7642435759) to (-0.354081335648,0.515701529314), (-0.347215411128,0.531148605055) and (-0.358215153428,0.521595652937), respectively.


![alt text][image5]

- I did data shuffle because in original training set and validation set, similar samples or samples from the same class are distributed closely as the first and second rows below, which is not good for classifier training. And after data shuffle, the samples are distributed randomly, as the third and forth rows below, which can improve the learning proformance of classifier, according to the experiences fo LeNet, SermanNet and AlexNet.

![alt text][image6]


**However, the original training/validation sets are limited and have less variations, and datasets are essential for classifier learning. So I decided to generate additional data with artifical data augmentation. Because data augmentation can increase the size and variations of training samples, which is very important for enhancing classifier performance. And many experiences about training CNNs have prove this. **

Here I boosted the training sample_numbers of each class to 800, and validation sample_numbers of each class to 200. And each augmented sample is pre-processed througn 5 data augmentation functions. The differences of each single pre-processing function between the original data set and the augmented data set are as follows:

- random_translation: also called shift. applies a random affine translation to images in [-2,2] horizontally or vertically or both.

![alt text][image7]

- rotation: applies a 2D rotation affine transformation(5 degrees) to images.

![alt text][image8]

- random perspective: applies a random perspective transformation to images.

![alt text][image9]

- intensity adjustment: adjusts the intensities of images in new scales.

![alt text][image10]

- random warping:applies a random 2D affine transformation to images.

![alt text][image11]

- **horizontal reflection is not adopted**. Because traffic sign recognition is based on semantic recognition, horizontal reflection may lead to totally different misunderstanding, like the speed sign '60'and '09'.

The reference docs:

- [opencv Geometric Image Transformation](http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html?highlight=warpaffine)
- [cnblogs](www.cnblogs.com/pakfahome/p/3914318.html)

Therefore, before data augmentation, X_train,y_train shapes are (34799, 32, 32, 1),(34799,) and maximum and minimum sample_numberes are 2010, 180. X_validation, y_validation shapes are(4410, 32, 32, 1),(4410,) and maximum and minimum of sample_numbers are 240,30, repectively.

After data augmentation, X_train,y_train shapes are (46480, 32, 32, 1), (46480,)and maximum and minimum sample_numberes are 2010, 800. And X_valid,  y_valid shapes are (8770, 32, 32, 1), (8770,)and maximum and minimum of sample_numbers are 240,200,respectively. The multi-histogram of training/validation/test set is as follows: data distributions are more balanced.

![alt text][image12]

---

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc. Consider including a diagram and/or table describing the final model.

**My final model is named as modified Multi-scale CNN(modified MSCNN for short). And I tried three CNN architectures from Modified LeNet(Source: Yan LeCun - "Gradient-based Learning Applied to Document Recognition" Proceedings of IEEE 1998), Multi-scale CNN(MSCNN for short,Source: Pierre Serment and Yann LeCun - "Traffic Sign Recognition with Multi-scale Convolutional Networks" IJCNN2011) to MSCNN(Source: Alex Krizhevsky, Iiya Sutskever and Geoffrey E.Hinton - "ImageNet Classification with Deep Convolutionnal Neural Networks" NIPS2012). And modified MSCNN has the best performance.**

The parcific description about modified MSCNN can be listed here. For more details you can refer to the source code and my notes in traffic_sign_classifier.ipynb.

### Final Model Architecture : Multi-scale CNN  

**Input**

This CNN model accepts a 32x32x1 image as input.

**Architecture**

**Layer 1: Convolutional.** The output shape should be 28x28x6.

**Activation.** Choose Rectified Linear Unit(ReLU) as activation function.

**Pooling.** The output shape should be 14x14x6.

**Layer 2: Convolutional.** The output shape should be 10x10x16.

**Activation.** Choose Rectified Linear Unit(ReLU) as activation function.

**Pooling.** The output shape should be 5x5x16.

**Layer 3: Convolutional.** The output shape should be 1x1x400.

**Activation.** Choose Rectified Linear Unit(ReLU) as activation function. 

**Layer 4: Flatten.** Flatten the outputs of layer 2 (scale1: 5x5x16 = 400)and layer 3 (scale2: 1x1x400 = 400)as 1D instead of 3D. The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you. Two outputs of 2 scales stand for different levels of image features: local and global abstractions, respectively.

**Layer 5: Concatenated Layer.** The output shape including scale 1 and scale 2 should be 800(400+400 = 800). And this output will be used to feed classifier layer.

**A dropout layer is added here with keep_prob 0.5, according to the experience of AlexNet.** 

**Layer 6: Fully Connected.** This layer has an output of 120.

**Activation.** Choose Rectified Linear Unit(ReLU) as activation function.

**A dropout layer is added with keep_prob 0.5, based on the experience of AlexNet.**

**Layer 7: Fully Connected(Logits).** This should have 43 outputs.

**Output** 

Return the results of 43 classes.

And the visualization of this architecture are as follows:

| Layer         		  |     Description	        					        | 
|:---------------------  :|:---------------------------------------------------:| 
| Input         		  | 32x32x1 grayscale image   				            | 
| Convolution1 (5,5,1,6)  | 1x1 stride, valid padding, outputs 28x28x6 	        |
| RELU					  |												        |
| Max pooling	      	  | 2x2 stride,  outputs 14x14x6 		    	        |
| Convolution2 (5,5,6,16) | 1x1 stride, valid padding, outputs 10x10x16         |
| RELU                    |                                                     |
| Max pooling	      	  | 2x2 stride,  outputs 5x5x16 		    	        |
| Convolution3(5,5,16,400)| 1x1 stride, valid padding, outputs 1x1x400 	        |
| RELU                    |                                                     |
| Flatten4                | input1:5x5x16,output1:400. input2:1x1x400,output2:400|
| Concatenated5           | Input: 400 + 400, Output:800                        |
| Dropout                 | keep_prob: 0.5                                      |
| Fully connected6		  | Input:800. Output:120.                              |
| Dropout                 | keep_prob: 0.5                                      |
| Fully connected7		  | Input:120. Output:43.                               |
| Output                  | 43 classes


---

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer in modified MSCNN. And the final setting of parameters are:

- epochs:100
- batch_size: 256
- learning_rate:0.0009
- dropout keep_prob: 0.5.
- mu:0
- sigma: 0.1.

---

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.936.
* test set accuracy of 0.960.

**- It's very interesting that my test accuracy(0.960) is pretty higher than training accuracy(0.936). I think this is because in data augmentation the original image data is artificially pre-processed through 5 augmentation algorithms(random_translation, rotation, random_perspective, intensity_adjustment and random_warping), and the size and variety of training/validation data are much enhanced(34799 training images -> 46480 training images and 4410 validation images->8770validation images). **

**- This pre-processing has improved the generalization of my classifier efficiently. Then when it is tested on testing data(original test images where the difficulty decreases), and the test accuracy is higher.**


My solution is kinda combination of a well known architecure and an iterative approach. 

* What was the first architecture that was tried and why was it chosen?

As I wrote earlier, my first architecture is Modified LeNet with two dropout layers(Source: Yan LeCun - "Gradient-based Learning Applied to Document Recognition" Proceedings of IEEE 1998). The reason why I chose this well-known model is it's the first useful CNN architecture I learnt from Udacity course. 

Here is the description of this modified LeNet.

### Model Architecture 1 : Modified LeNet 
**Input**

This CNN model accepts a 32x32x1 image as input.

**Architecture**

**Layer 1: Convolutional.** The output shape should be 28x28x6.

**Activation.** Choose Rectified Linear Unit(ReLU) as activation function, according to AlexNet.

**Pooling.** The output shape should be 14x14x6.

**Layer 2: Convolutional.** The output shape should be 10x10x16.

**Activation.** Choose Rectified Linear Unit(ReLU) as activation function.

**Pooling.** The output shape should be 5x5x16.

**Flatten.** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you. The output shape should be 400.

**Layer 3: Fully Connected.** This should have 120 outputs.

**Activation.** Choose Rectified Linear Unit(ReLU) as activation function.

**A dropout layer is added here with keep_prob 0.5, according to the experience of AlexNet.** 

**Layer 4: Fully Connected.** This should have 84 outputs.

**Activation.** Choose Rectified Linear Unit(ReLU) as activation function.

**A dropout layer is added here with keep_prob 0.5, according to the experience of AlexNet.** 

**Layer 5: Fully Connected (Logits). **This should have 43 outputs.

**Output** 

Return the results of 43 classes.


* What were some problems with the initial architecture?

After lots of debugging and trials, the validation accuracy of architecture 1 stays on 0.966 and cannot be improved. Then I tried architecture 2: Multi-scale CNN(MSCNN for short,Sources: 1. Pierre Serment and Yann LeCun - "Traffic Sign Recognition with Multi-scale Convolutional Networks" IJCNN2011 2. Alex Krizhevsky, Iiya Sutskever and Geoffrey E.Hinton - "ImageNet Classification with Deep Convolutionnal Neural Networks" NIPS2012 ). The details about MSCNN are listed as follows:

### Model Architecture2 : Multi-scale CNN  

**Input**

This CNN model accepts a 32x32x1 image as input.

**Architecture**

**Layer 1: Convolutional.** The output shape should be 28x28x6.

**Activation.** Choose Rectified Linear Unit(ReLU) as activation function.

**Pooling.** The output shape should be 14x14x6.

**Layer 2: Convolutional.** The output shape should be 10x10x16.

**Activation.** Choose Rectified Linear Unit(ReLU) as activation function.

**Pooling.** The output shape should be 5x5x16.

**Layer 3: Convolutional.** The output shape should be 1x1x400.

**Activation.** Choose Rectified Linear Unit(ReLU) as activation function. 

**Layer 4: Flatten.** Flatten the outputs of layer 2 (scale1: 5x5x16 = 400)and layer 3 (scale2: 1x1x400 = 400)as 1D instead of 3D. The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you. Two outputs of 2 scales stand for different levels of image features: local and global abstractions, respectively.

**Layer 5: Concatenated Layer.** The output shape including scale 1 and scale 2 should be 800(400+400 = 800). And this output will be used to feed classifier layer.

**A dropout layer is added here with keep_prob 0.5, according to the experience of AlexNet.** 

**Layer 6: Fully Connected(Logits).** This layer has an output of 43.

**Output** 

Return the results of 43 classes.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. 

The differences between modified LeNet and MSCNN are: 

1.MSCNN combines the outputs of layer2 and layer3 with concatenated layer, which means different-level feature maps of input images are used. This may help to increase the representative power of classifier, leading to a better performance. And the experiments have proven my idea. The best validation accuracy is 0.968.

2.a new concatencated layer and less fully connected layers are used. So MSCNN has less parameters.

Then my final model is modified Multi-scale CNN(modified MSCNN for short) with an adding fully-connected layer6 and a following dropout layer. The details are in question2 above. And modified MSCNN has the best performance: validation accuracy of 0.973 on traditional preprocessed training data and validation accuracy of 0.936 on the hard artifical data augmented training data.


* Which parameters were tuned? How were they adjusted and why?

Here is my log about the whole debugging process, including all details about architectures and parameters.

# Debugging log

- 2017/05/10 - validation accuracy: 0.910
   - model: Modified LeNet
   - data preprocessing: grayscale convert, data normalization, data shuffle.
   - parameters: epochs: 10, batch_size: 128, keep_prob: 0, learning_rate: 0.001, mu: 0, sigma: 0.1
- 2017/05/10 - validation accuracy: 0.936
   - model: Modified LeNet
   - data preprocessing and all other parameters remain the same except epochs :100
- 2017/05/10 - validation accuracy: 0.920
   - model: Modified LeNet
   - data preprocessing stays the same
   - parameters: epochs: 100, batch_size: 512, the rest stay the same: epochs: 100, learning_rate: 0.001, mu: 0, sigma: 0.1.
- 2017/05/11 - validation accuracy: 0.951
   - model: Modified LeNet
   - data preprocessing remains the same. Two dropout layers are designed to follow the two fully connected layers with keep_prob: 0.5 , batch_size: 512, epochs: 100, learning_rate: 0.001, mu: 0, sigma: 0.1, according to the experience of AlexNet.
- 2017/05/11 - validation accuracy: 0.966
   - model: Modified LeNet
   - data preprocessing and all other parameters remain the same, except batch_size: 128.
- 2017/05/12 - validation accuracy: 0.945
   - model: multi-scale CNN(MSCNN)
   - data preprocessing remains the same. Please refer to the cells above about MSCNN for more details.
   - parameters: epochs: 100, batch_size: 128, keep_prob: 0 , learning_rate: 0.001, mu: 0, sigma: 0.1.
- 2017/05/12 - validation accuracy: 0.968
   - model: MSCNN
   - data preprocessing and all parameters remain the same. Except that one dropout layer is added after the layer 5(concatenated layer) with keep_prob: 0.5.
- 2017/05/12 - validation accuracy: 0.953
   - model: MSCNN
   - data preprocessing stays the same.
   - parameters: epochs: 100, batch_size: 128, keep_prob: 0 , learning_rate: 0.0005 because there are some fluctuations of accuracies among last epochs, mu: 0, sigma: 0.1.
- 2017/05/12 - validation accuracy: 0.963
   - model: MSCNN
   - learning_rate: 0.0008. and other data preprocessing and parameters remain the same.
- 2017/05/13 - validation accuracy: 0.970
   - model: modified MSCNN
   - a fully connected layer FC7 and a dropout layer are added with keep_prob: 0.5. 
   - parameters: epochs: 100, batch_size: 128, learning_rate: 0.0008, mu: 0, sigma: 0.1.
- 2017/05/13 - validation accuracy: 0.973
   - model: modified MSCNN
   - a fully connected layer FC7 and a dropout layer are added with keep_prob: 0.5. 
   - parameters: epochs: 100, batch_size: 256, learning_rate: 0.0008, mu:0, sigma:0.1.
- 2017/05/14 - 05/17  - validation accuracy: 0.923->0.925->0.930->0.931->0.895->0.935->0.932
   - model: modified MSCNN
   - implementing data augmentation with my five self-built functions. Training samples: 34799->46480; validation: 4410->8770
   - a lot of a lot of exhausive debugging to improve the accuracy. Even on GPUs, it was a long story.
   - learning_rate: 0.0008-> 0.0008->0.0005->0.0001->0.001->0.0012->0.001
- 2017/05/18 - validation accuracy: 0.936->0.934.
   - model: modified MSCNN
   - parameters: epochs: 100, batch_size: 128, learning_rate: 0.001->0.0008,  keep_prob: 0.5, mu:0, sigma:0.1.
   - It seems data augmentation did not improve the accuracy after debugging, which made me very upset.
- 2017/05/18 - validation accuracy: 0.934.
   - model: modified MSCNN
   - parameters: epochs: 100, batch_size:256, learning_rate: 0.0008, keep_prob:0.5, mu:0 , sigma:0.1.
- 2017/05/18  - validation accuracy: 0.903.
   - model: modified MSCNN
   - parameters: epochs: 100, batch_size: 512, learning_rate: 0.0008, keep_prob:0.5, mu:0 , sigma:0.1.
- 2017/05/18 - validation accuracy: 0.920.
   - model: modified MSCNN
   - parameters: epochs: 150, batch_size: 512, learning_rate:0.0008, keep_prob:0.5, mu:0, sigma:0.1.
- 2017/05/18 - validation accuracy: 0.917.
   - model: modified MSCNN
   - parameters: epochs: 200, batch_size:512, learning_rate: 0.0008, keep_prob: 0.5, mu:0, sigma:0.1. 
- 2017/05/19 - validation accuracy: 0.915.
   - model: modified MSCNN
   - parameters: epochs:100, batch_size:256, learning_rate: 0.0008, keep_prob: 0.5, mu: 0, sigma:0.2.
- 2017/05/19  -validation accuracy: 0.905.
  - model: modified MSCNN
  - parameters:epochs:100, batch_size:256, learning_rate: 0.0008, keep_prob: 0.5, mu: 0, sigma:0.3.
- 2017/05/19  -validation accuracy: 0.917.
  - model: modified MSCNN
  - parameters:epochs:100, batch_size:256, learning_rate: 0.0008, keep_prob: 0.5, mu: 0, sigma:0.05.
- 2017/05/19  -validation accuracy: 0.904.
  - model: modified MSCNN
  - parameters:epochs:100, batch_size:256, learning_rate: 0.0008, keep_prob: 0.75, mu: 0, sigma:0.1. 
- 2017/05/29 - validation accuracy: 0.934
  - model: modified MSCNN
  - parameters: epochs:100, batch_size: 128, learning_rate: 0.0009, keep_prob: 0.5, mu:0, sigma:0.1.
- 2017/05/29 - validation accuracy: 0.936
  - model: modified MSCNN
  - parameters: epochs:100, batch_size: 256, learning_rate: 0.0009, keep_prob: 0.5, mu:0, sigma:0.1.


* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Multi-scale convolutional architectures and dropout layers are important design choices. A convolution layer is important because it can exact more high-level or global abstraction about the input images. A dropout layer is important becaust it can prevent the classifier overfitting.
 
---

### Test a Model on New Images

#### 1. Choose ten German traffic signs found on the web and provide them in the report. Discuss what quality or qualities might be difficult to classify.

Here are 10 German traffic signs that I found on the website named [Gettyimages](https://www.vcg.com/). All the 10 new images are pre-processed with resize, traditional grayscale-conversion and data normalization. So the images seem a little low-resolution.

![alt text][image13] 

All the 10 new images have brighter color intensities and backgrounds, and include few image borders, which are quite different from the real-world GTSRB dataset. Thus I think they may not be that easy to classify.

---

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.


Here are the results of the predictions:

| Images		                        |     Predictions	        					| ClassID |
|:-------------------------------------:|:---------------------------------------------:|:-------:|
| General Caution  		                | General Caution								|    18   |
| Turn right ahead		                | Turn right ahead  							|    33   |
| Children Crossing 	                | Children Crossing								|    28   |
| Speed limit(60 km/h)              	| Speed limit(60 km/h)  		 				|     3   |
| Slippery Road			                | Slippery Road      							|    23   |
| Road work                             | Road work                                     |    25   |
| No entry                              | No entry                                      |    17   |
| Roundabout mandatory                  | Roundabout mandatory                          |    40   |
| End of all speed and passing limits   | End of all speed and passing limits           |    32   |
| Go straight or right                  | Go straight or right                          |    36   |


My modified MSCNN classifier can correctly guess 10 of the 10 new test traffic signs, with a perfect accuracy of 100%. This compares favorably to the accuracy on the test set of 96%. This is a good sign that my model works well on different new data. It's reasonable to inference that the accuracy would not be so high given more data points, judging by the low fidelity of a number of images in the training dataset.It's also reasonable to assume that if the real-world data were all as easily distinguishable as the 10 new images chosen that the accuracy would remain very high.

---

#### 3. Describe how certain the model is when predicting on each of the 10 new images by looking at the softmax probabilities for each prediction. Provide the top 3 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model named modified MSCNN is located in the 44th cell of the Ipython notebook. Because all the top guess of these 10 new images, i.e. the probabilites of top_1  are 100%, and the rest top guesses are 0%. Here I list the top 3 softmax probabilities for each iamge along with the sign type of each probality.

![alt text][image14]

For the first image, the model is relatively sure that this is a General Caution sign (probability of 100%). The top three soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| General Caution   	      			     	| 
| 0.0     				| Speed limits(20km/h) 							|
| 0.0					| Speed limits(30km/h)	            			|


For the second image,the model is relatively sure that this is a Turn Right Ahead sign (probability of 100%). The top three soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Turn Right Ahead  	      			     	| 
| 0.0     				| Right-of-way at the next intersection	    	|
| 0.0					| Priority Road                     			|


For the third image, the model is relatively sure that this is a Children Crossing sign (probability of 100%). The top three soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Children Crossing  	      			     	| 
| 0.0     				| Slippery Road                             	|
| 0.0					| Beware of ice/snow                  			|


For the forth image, the model is relatively sure that this is a Speed Limits(60km/h) sign (probability of 100%). The top three soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed Limits(60km/h)       			     	| 
| 0.0     				| Keep Right 	         						|
| 0.0					| Speed limits(50km/h)	            			|


For the fifth image, the model is relatively sure that this is a Slippery Road sign (probability of 100%). The top three soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Slippery Road     	      			     	| 
| 0.0     				| Beware of ice/snow 							|
| 0.0					| Children Crossing 	            			|


For the sixth image, the model is relatively sure that this is a Road Work sign (probability of 100%). The top three soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Road Work   	      	        		     	| 
| 0.0     				| Beware of ice/snow 							|
| 0.0					| Bicycles crossing                 			|


For the seventh image, the model is relatively sure that this is a No Entry sign (probability of 100%). The top three soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No Entry        	      		  	        	| 
| 0.0     				| Yield     						         	|
| 0.0					| No passing       	                			|


For the eighth image, the model is relatively sure that this is a Roundabout Mandatory sign (probability of 100%). The top three soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Roundabout Mandatory   	      			   	| 
| 0.0     				| Priority Road     							|
| 0.0					| Speed limits(30km/h)	            			|

For the ninth image, the model is relatively sure that this is a End of all speed and passing limits sign (probability of 100%). The top three soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| End of all speed and passing limits 	     	| 
| 0.0     				| End of no passing 							|
| 0.0					| End of Speed limits(80km/h)	       			|


For the tenth image, the model is relatively sure that this is a Go Straight or Right sign (probability of 100%). The top three soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Go Straight or Right 	      			     	| 
| 0.0     				| Yield             							|
| 0.0					| Priority Road     	            			|

Besides, here is the top 5 softmax probabilites for the predictions on the 10 new german traffic signs.

![alt text][image15]