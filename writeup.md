# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./writeup/distribution.png "Distribution"

[image4]: ./Germam_traffic_signs/11.jpg "Traffic Sign 1"
[image5]: ./Germam_traffic_signs/13.jpg "Traffic Sign 2"
[image6]: ./Germam_traffic_signs/20.jpg "Traffic Sign 3"
[image7]: ./Germam_traffic_signs/25.jpg "Traffic Sign 4"
[image8]: ./Germam_traffic_signs/29.jpg "Traffic Sign 5"


### Code

Here is a link to my [project code](https://github.com/xpharry/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

1. Summary

    I use the python built-in functions to calculate summary statistics of the traffic signs data set:

    * The size of training set is 34799.
    * The size of the validation set is 4410.
    * The size of test set is 12630
    * The shape of a traffic sign image is (32, 32, 3)
    * The number of unique classes/labels in the data set is 43.

2. Exploratory visualization of the dataset.

    Here is an exploratory visualization of the data set. It is a bar chart showing how the training data distributed.

    ![alt text][image1]

### Data Preprocessing

1. Image Preprocessing

    1) Techniques

        * Grayscale
        * Normalize
        * Histogram-equalize

    2) Pipeline

        * As a first step, I decided to convert the images to grayscale because color here is not a key effective feature for the traffic sign classification.

        * As a second step, I normalized the image data because it evenly considers the effects of all the features.

        * As the last step, I do the histogram-equalize because it adjusts image intensities to enhance contrast which highlights the features.

2. Data Augumentation

    1) Techniques

        * rotation
        * transplant
        * brightness
    
    2) Pipeline

        I decided to generate additional data because this makes it unlikely to happen over-fitting.

        To add more data to the the data set, I used the above techniques because the generated images can be distinguished and keep the key features.

    3) Conclusion

        * The difference between the original data set and the augmented data set is that

            the augemented images are all transformed with the preprocessing techniques indtroduced in 1) and

            the new dataset is way bigger than the original by transplanting, rotating and brightening the original images.

### Design and Test a Model Architecture

1. Model Description

    I start with the LeNet-5 implementation shown in the class which is a solid starting point. I only need to change the number of classes and possibly the preprocessing.

    My final model consisted of the following layers:

    | Layer         		|     Description	        					|
    |:---------------------:|:---------------------------------------------:|
    | Input         		| 32x32x1 gray image   							|
    | Convolution 3x3     	| 1x1 stride, 'valid' padding, outputs 28x28x64 |
    | RELU					|												|
    | Max pooling	      	| 2x2 stride,  outputs 14x14x64 				|
    | Convolution 3x3     	| 1x1 stride, 'valid' padding, outputs 10x10x16 |
    | RELU					|												|
    | Max pooling	      	| 2x2 stride,  outputs 5x5x64 		    		|
    | Flatten       	    | 5x5x64 --> 400								|
    | Fully connected		| 400 --> 120   								|
    | RELU             	    |             									|
    | Fully connected		| 120 --> 84        							|
    | RELU             	    |             									|
    | Fully connected		| 84 --> 43        								|

2. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

    To train the model, I used an Adam optimizer, a batch size of 128 and 50 epochs.

    The learning rate is using a general value 0.001 which proves good enough.

3. Result Analysis
    My final model results were:
    * training set accuracy of 0.999
    * validation set accuracy of 0.963
    * test set accuracy of 0.940

### Test a Model on New Images

1. Choose five German traffic signs found on the web and provide them in the report.

    Here are five German traffic signs that I found on the web:

    ![alt text][image4] ![alt text][image5] ![alt text][image6]
    ![alt text][image7] ![alt text][image8]

2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

    Here are the results of the prediction:

    | Image			                        |     Prediction	        					|
    |:-------------------------------------:|:---------------------------------------------:|
    | Slippery Road 		                | Slippery Road   								|
    | Dangerous curve to the right          | Dangerous curve to the right 					|
    | Yield					                | Yield											|
    | Bicycles crossing		                | Keep right					 				|
    | Right-of-way at the next intersection	| Right-of-way at the next intersection         |


    The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.

    For the first image, the model is relatively sure that this is a "Dangerous curve to the right" sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

    | Probability         	|     Prediction	        					|
    |:---------------------:|:---------------------------------------------:|
    | .90         			| Dangerous curve to the right					|
    | .07     				| Children crossing 							|
    | .05					| End of no passing								|
    | .01	      			| End of all speed and passing limits   		|
    | .00				    | Slippery Road      							|

    For the second, fourth and fifth images, the model is confident and accurate that they are "Road work", "Yeild" and "Right-of-way at the next intersection" with almost 100% probability respectively.

    The only exception is when it comes to the third image, it is relatively confident with a wrong label which should be "Bicycles crossing".

    | Probability         	|     Prediction	        					|
    |:---------------------:|:---------------------------------------------:|
    | .70         			| No entry					                    |
    | .15     				| End of speed limit (80 km/h)  				|
    | .10					| Turn left ahead								|
    | .03	      			| End of no passing                       		|
    | .02				    | Slippery Road      							|

    I guess the biggest reason it goes wrong is that the image is quite askew.

### (Optional) Visualizing the Neural Network

1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

### Conclusion


