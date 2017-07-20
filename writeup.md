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

[image_dataset]: ./writeup/distribution.png "Distribution"
[image_lenet]: ./writeup/lenet.png "LeNet"

[resized_image1]: ./writeup/11-resized.jpg "Traffic Sign 1"
[resized_image2]: ./writeup/13-resized.jpg "Traffic Sign 2"
[resized_image3]: ./writeup/20-resized.jpg "Traffic Sign 3"
[resized_image4]: ./writeup/25-resized.jpg "Traffic Sign 4"
[resized_image5]: ./writeup/29-resized.jpg "Traffic Sign 5"

[predict_image]: ./writeup/predict.png

### Code

Here is a link to my [project code](https://github.com/xpharry/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

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

    ![alt text][image_dataset]

### Data Preprocessing

1. Image Preprocessing

    1) Techniques

        * Grayscale
        * CLAHE
        * Normalize
        * Histogram-equalize

    2) Pipeline

        * As the 1st step, I decided to convert the images to grayscale because color here is not a key effective feature for the traffic sign classification.
        * As the 2nd step, I enhanced the contrast of the grayscale image by transforming the values using contrast-limited adaptive histogram equalization (CLAHE).
        * As the 3rd step, I normalized the image data because it evenly considers the effects of all the features.
        * As the last step, I do the histogram-equalize because it adjusts image intensities to enhance contrast which highlights the features.

2. Data Augumentation

    1) Techniques

        * Translate
        * Rotate
        * Scale
        * Brighten
    
    2) Pipeline

        I decided to generate additional data because this makes it unlikely to happen over-fitting.

        To add more data to the the data set, I used the above techniques because the generated images can be distinguished with the original and also keep the key features.

    3) Conclusion

        * The difference between the original data set and the augmented data set is that
            * the augemented images are all transformed with the preprocessing techniques indtroduced in 1) and
            * the new dataset is way bigger than the original by transplanting, rotating and brightening the original images.

### Design and Test a Model Architecture

1. Model Description

    ![alt text][image_lenet]

    I start with the LeNet-5 implementation shown in the class which is a solid starting point. I only need to change the number of classes and possibly the preprocessing.

    In the beginning, I use the original LeNet-5 model without dropout layers and train it on the dataset without pre-processing. The validation accuracy could reach 92% already.

    To reach a better performance, besides pre-processing the images and augment the dataset, the model can be modified slightly. Among those potential techniques, adding dropout layers highlights itself for its simplicity and effectiveness.

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
    | Dropout          	    | keep_prob = 0.5								|
    | Fully connected		| 120 --> 84        							|
    | RELU             	    |             									|
    | Dropout          	    | keep_prob = 0.5								|
    | Fully connected		| 84 --> 43        								|

2. Idea

    * What was the first architecture that was tried and why was it chosen?

        **Answer:**

        LeNet-5 as shown in the above picture. It is a starting point since it has been proved its effectiveness for image classification and the architecture is simple and understandable.

    * What were some problems with the initial architecture?

        **Answer:**

        There is no regularization in the initial architecture. Dropout is the first choice for regularization due to its simplicity and effectiveness.

    * How was the architecture adjusted and why was it adjusted?

        **Answer:**

        The input and output shapes were to be modified to match the dataset. After that, input was 32x32x1 and output was a one dimensional vector with the size as 43.

        Then as explained in the previous point, two dropout layers were added in right before two fully connected layers, respectively.

    * Which parameters were tuned? How were they adjusted and why?

        **Answer:**

        The number of epochs, the learning rate, the batch size, and the drop out probability were all parameters tuned.

         As well, the random generated image data were tuned.

         For the number of epochs, I tuned this simply based on how it variate along the training process. I use the number to make sure that I could reach my accuracy goals.

         The batch size I increased only slightly since starting once I increased the dataset size.

         The learning rate I think could of been left at .001 which is as I am told a normal starting point, but I just wanted to try something different so .00097 was used. I think it mattered little.

         The dropout probability mattered a lot early on, but after a while I set it to 50% and just left it. The biggest thing that effected my accuracy was the data images generated with random modifications.

         This would turn my accuracy from 1-10 epochs from 40% to 60% max to 70% to 90% within the first few evaluations. Increasing the dataset in the correct places really improved the max accuracy as well.

    * What are some of the important design choices and why were they chosen?

        **Answer:**

        The learning rate and the batch size need to be tuned considering both the accuracy and training speed. Make sure the best capcity of the CPU / GPU could be used.

3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

    To train the model, I used an Adam optimizer, a batch size of 128 and 50 epochs.

    The learning rate was using a general value 0.001 which proves good enough.

3. Result Analysis

    My final model results were:

    * training set accuracy of 0.998
    * validation set accuracy of 0.969
    * test set accuracy of 0.936

### Test a Model on New Images

1. Visualization and Initial Analysis

    Here are the 5 German Traffic Sign images that I captured on the web,

    ![][resized_image1]  ![][resized_image2] ![][resized_image3] ![][resized_image4] ![][resized_image5]

    The five images are downloaded along with their watermarks which I am not sure how much degree will effect the feature matching.

    The first image is a little dark which yet I believe can be ignored with the pre-processing.

    The third image is also a little challenging for its light reflection which decrease the pixel difference or the image contrast.

    The fifth image is skewed which increases the difficulty to be matched with the right label.

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

    ![][predict_image]

    For the first image, the model is quite sure that this is a "Dangerous curve to the right" sign (probability of 0.93), and the image does contain a stop sign. The top five soft max probabilities were

    | Probability         	|     Prediction	        					|
    |:---------------------:|:---------------------------------------------:|
    | .93         			| Dangerous curve to the right					|
    | .03     				| Children crossing 							|
    | .03					| Right-of-way at the next intersection			|
    | .01	      			| Slippery road                         		|
    | .00				    | Beware of ice/snow   							|

    For the first image, the model is relatively sure that this is a "Dangerous curve to the right" sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

    | Probability         	|     Prediction	        					|
    |:---------------------:|:---------------------------------------------:|
    | .84         			| Road work			                    		|
    | .16     				| Bumpy road         							|
    | .00					| Beware of ice/snow	                		|
    | .00	      			| Bicycle crossing                      		|
    | .00				    | General caution    							|

    The only exception is when it comes to the third image, it is relatively confident with a wrong label which should be "Bicycles crossing".

    | Probability         	|     Prediction	        					|
    |:---------------------:|:---------------------------------------------:|
    | .82         			| Keep right    			                    |
    | .15     				| Slippery road                   				|
    | .004					| Bicycle crossing      						|
    | .002	      			| Dangerous curve to the right            		|
    | .002				    | Speed limit (60 km/h)							|

    I guess the biggest reason it goes wrong is that the image is quite askew.

    For the fourth and fifth images, the model is confident and accurate that they are "Yeild" and "Right-of-way at the next intersection" with almost 100% probability respectively.

### (Optional) Visualizing the Neural Network

1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

### Conclusion

Two more tasks need to be done for better understanding of this project:

1. Model Visualization in real time with TensorBoard

2. Feature Visualization

which will be finished then.

