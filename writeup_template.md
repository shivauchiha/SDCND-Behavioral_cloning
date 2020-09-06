# **Self Driving car Nanodegree** 

## Behavioral cloning

### Report
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Nvidia.jpeg "Model Visualization"


## Rubric Points
-> All files included in submission network.py,drive.py,model.h5,writeup,video.mp4

->The model3.h5 files can be used to successfully drive the car around

->The network.py file has a generator implementation to feed in hughe volumes of data in batch format.
It also successfuly implements a nvidia network to perform a regression based inferencing on steering angles.

->Three 5*5 conv and two 3*3 conv layer model is used. This is based on the nvidia CNN model for behavioral cloning

->Train/test split is done and the model was trained with different sequence captured images to avoid overfitting.

->The model uses 2 sets of data. One is the default given data . This actually drives perfectly on laned roads,However it fails at certain turns.Then data relating to the car turning behaviour at that turn were recorded and used to further train the network.

->Further details regarding are given below. 

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* network.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model3.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model3.h5
```

#### 3. Submission code is usable and readable

The network.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is a traditional regression based CNN.I have used the nvidia model for behavioral cloning .The network consists of intial 3 layers of 5*5 convolutional filters .This is followed by 2 layers of 3*3 convolutional filters .The end is flattened and spread to 1164 neuron fully connected layers.Then the network funnels the activation to reducing size of FC layers (100,50,10).The network finally ends at single node where steering predicted value is delivered. 

#### 2. Attempts to reduce overfitting in the model

The model was trained with different distribution of data to prevent overfitting.I did experiment with drop out but it didnt offer much albeit at a huge cost to performance . The validation of loss was pretty good without dropout. The model did fail at turns .Thus to overcome this a second distribution of data was captured from simulator especially at turns and was used to train the network.After this the car drove around the track perfectly.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My overall approach was simple. I checked default dataset and realised it covers atleast 2 rounds with perfect driving behavior .So i decided will begin training with these dataset.This pretty much worked for most of the track excpet for 2 turns . So i took recording of proper driving over the 2 tracks and also added possible recovery behaviour from left lane and right lane . A total of close to 60000 images were used to train and the model performs satisfactorily.

#### 2. Final Model Architecture

Here is a visualization of the architecture 

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

Apart from tactics described in Solution Design Approach , I also added few more processes to help the network converge faster and predict angles more accurately.
The first process was done within generator itself where i took side camera as well into account and a +/- 0.2 is added to steering angle.
Then an exact mirror copy of above dataset was taken and it was flipped to create additional images albeit if the car has to run in opposite direction and eleminate lane bias.This tactics increased our training set resulting in training optimally fit network.Then before these images are passed in the network they are mean centred normalised and image is cropped such that only road part of the image is sent to the network for training .

finally the network was trained using MSE loss function and relu activation functions were used to introduce non-linearity within the network.
