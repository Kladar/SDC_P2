

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This is my write up. My code can be found on my [github](github.com/kladar). It is also submitted via github on the project submission page.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell. I use numpy to look at an random image and then output a histogram of the new amount of images in each class in the fourth code cell as well. the Angling really helps give a good dataset.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is augmented to 78117
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43, which is what we came in with.

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fourth code cell of the jupyter notebook.

It shows a histogram of the number in each class and a random example image.

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

I didn't do much preprocessing because it turned out to not be necessary, and we could be losing some information such as color, which is important in street sign classification (though the NN doesn't look at it here, we could be using it in the future). 

I originally normalized the data but then removed it when debugging, forgot to add it back in, and it didn't seem to matter much, but it wouldn't be difficult to add back in. We then have to remember to do it to the new images as well but I'm as far behind as I am in this course so I'm going to move on. I do shuffle the data to avoid overfitting.

I thought of normalization because we want to avoid high variance in the data, which makes it easy to overshoot our optima and hard to tune the learning rate to a value that works generally. 

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

 In the same cells as the summary statistics is where we augment the data and set up the training, validation, and testing features and labels. The data came split from the .zip file. The summary statistics from above show how much was in each set after the augmentation. 

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

In cells 7, 8, 9, and 10 is where the model architecture lives. It has a similar architecture to the LeNet lab, but I added some important features. First, I resized the depths to fit our 3 color images and 43 classes. The model consists of two convolutional layers of similar size to the LeNet lab, and three fully connected layers. There is also a flattening feature before the first fully connected layer. I also aded a drop out with keep probability of 0.5 after the first fully connected layer. Adding more seemed to decrease the ability of the model to perform, but adding one really helped it excel. Validation takes place in the 11th code cell.

Architecture 
* Input 32x32x3
* Conv Layer 1 5x5x3 filter with stride 1 (valid padding) to 28x28x6
* Relu Activation
* Max Pooling 2x2 kernel, 1x1 stride to 14x14x6
* Conv Layer 2 5x5x6 filter with stride 1 (valid padding) to 14x14x16
* Relu Activation
* Max Pooling 2x2 kernel, 2x2 stride and Flatten to 400 (vector)
* Fully Connected Layer 400 to 120
* Relu Activation
* Dropout with keep probability of 0.5
* Fully Connected Layer 120 to 84
* Relu Activation
* Fully Connected Layer 84 to 43 (our class number)



####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Training is located in the 12th code cell. It uses an Adam Optimizer as this proved better than stochastic gradient descent. The batch size remained at 128 because I'm not sure the memory size of the AWS gpu, but since I was using the AWS GPU I put it to the test and gave it 50 epochs and a low learning rate of 0.001. This took some time but the model accuracy came out pretty good.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the validation of the model is located in the 11th cell and the testing accuracy is measured in the 13th. 

My final model results were:
* validation set accuracy of 99.3% 
* test set accuracy of 94.6%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
I started with the LeNet lab architecture and added and took away features as they improved the model, then when I got something that was pretty good on my home CPU I moved over to an AWS GPU and cranked the hyper parameters. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I found 5 German signs on the web, and they output in cell 14. I provide them as .png Extras 1-5. 

These images seem easy to classify because they are well lit and straight on, cropped well, and already sized correctly. However, the last two are not in our classes, so they should be impossible to classify. If they are classified by the neural net we are all in trouble. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Image 0 probabilities: [ 0.1000223   0.0850348   0.07647138  0.0607005   0.05989883] 
 and predicted classes: [ 0  8  6 20  5]
Image 1 probabilities: [ 0.19398579  0.12281812  0.09570348  0.04908011  0.04147678] 
 and predicted classes: [ 5  6  3  8 32]
Image 2 probabilities: [ 0.08721258  0.08555469  0.07046558  0.06643748  0.05488154] 
 and predicted classes: [ 3  8  6  5 20]
Image 3 probabilities: [ 0.07704704  0.06753852  0.05709795  0.05536766  0.05213998] 
 and predicted classes: [ 3 20  6  5  8]
Image 4 probabilities: [ 0.12475     0.10739183  0.07268106  0.05879739  0.04790564] 
 and predicted classes: [ 6  3 32 20  5]


The model was only able to get 2 out of the 5 (if I'm reading this correctly) and it is not very confident on that. This fairs poorly with our test set accuracy, but I think it could be improved by expanding out training to include all the signs.  

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th cell of the notebook. It is not very sure on any of them, which makes sense for the last 2 but not for the first three since I believe those are in our dataset. IT knows most of them are speed limits, but it can't quite figure out which. It looks like it might also be taking in something else because they are all guessed to be speed limits. Hmm.

The image softmax probabilities are shown below:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .10         			|   20km/h 									| 
| .085     				| 120 km/h 										|
| .076					| end of 80km/h											|
| .06	      			| Dangerous curve right					 				|
| .06				    |  80 km/h  							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .19         			| 80 km/h   									| 
| .12     				| end of 80 km/h 										|
| .10					| 60 km/h											|
| .05	      			| 120 km/h					 				|
| .04				    |   end all speed limits    							|
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .087         			|  60km/h  									| 
| .085     				| 120 km/h 										|
| .07					| end of 80km/h											|
| .066	      			| 80 km/h					 				|
| .055				    | dangerous curve right      							|
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .077         			|  60 km/h  									| 
| .067     				| dangerous curve right										|
| .057					| end of 80km/h											|
| .055	      			| 80 km/h					 				|
| .052				    | 120 km/h      							|
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .12         			| end of 80 km/h   									| 
| .107     				| 60 km/h 										|
| .072					| end of all speed limits											|
| .059	      			| dangerous curve right					 				|
| .05				    |  80 km/h 							|

