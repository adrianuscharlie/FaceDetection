# Face Detection
 
Create deep learning model for face recognition using CNN and custom dataset with real time detection. The purpose of building this repository are want to do face detection, and the model will be deployed to the ESP Cam 32 for automatic door using face recognition.
<br>
<img src="https://yt3.googleusercontent.com/ytc/AL5GRJXDeStsPJL7Uz92074WfPjSGB7j810G8LqwhTKKSA=s900-c-k-c0x00ffffff-no-rj" width="200" height="200">
<img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-ar21.png" width="400" height="200"> 
<img src="https://editor.analyticsvidhya.com/uploads/232202.png" width="200" height="200"> 
<img src="https://www.vectorlogo.zone/logos/python/python-ar21.png" width="400" height="200">  

## Features
- Tools
- Dataset
- Modelling
- Real Time Predictions

## Tools
- Python
- Tensorflow
- Opencv
- Keras Image Data Generator

## Dataset
Dataset that used is taken by myself. I gather multiple photo of my friends face. This dataset consist of 3 labels ( Dyah, Debby, and Unknown). The dataset
will be splitted into train and test dataset with ratio 0.8 : 0.2
| People | Label | Quantity |
| --- | --- | --- |
| Dyah | dyah | 280 |
| Debby | debby | 202 |
| UnKnown | UnKnown | 411 |

The Dataset will be loaded using Custom Class that using ImageDataGenerator to make easier load and preprocess the image data. For the custom dataset class, you can
access it through [This Link](https://github.com/adrianuscharlie/FaceDetection/blob/main/dataset.py)

## Modelling
For this face detection case, I using deep learning model that implemented CNN Algorithm. The target image for the model are 80 x 80 pixel with 
3 color channels. This model using adam for the optimizer, relu for the hidden layer activation function and softmax activation function for the
output layer, because the case are multiclass image classification. After that, the model will be trained using 10 epochs, batch size equals to 8,
and will be evaluated using testing data. For detail modelling information, you can access it through [This Link](https://github.com/adrianuscharlie/FaceDetection/blob/main/model.py)

## Realtime Predictions
This is some example of real time predictions using the current model that already trained from previous step.
<p float="left">
  <img src="https://github.com/adrianuscharlie/FaceDetection/blob/main/Image/dyah.jpg" width="400" height="200">
  <img src="https://github.com/adrianuscharlie/FaceDetection/blob/main/Image/debby.jpg" width="400" height="200"> 
  <img src="https://github.com/adrianuscharlie/FaceDetection/blob/main/Image/unknown.jpg" width="400" height="200"> 
</p>
