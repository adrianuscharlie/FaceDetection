# Import all needed libraries
import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
import cv2 as cv
from dataset import Dataset
import numpy as np


# Create Custom Model class for Keras Model
class CustomModel:
    """Create Model class from tensorflow/keras that contains the base model, the parameters of the model.
    This class can do training, evaluating and realtime prediction for face recognition
    using custom datasets
    """
    
    def __init__(self,parameters:dict) -> None:
        self.customModel=tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64,(3,3),input_shape=(parameters['image_size'][0],parameters['image_size'][1],3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16,activation='relu'),
        tf.keras.layers.Dense(3,activation='softmax')
    ])
        # define model parameters
        self.parameters=parameters
        self.customModel.compile(optimizer=parameters['optimizer'],loss=parameters['loss_fn'],metrics=parameters['metrics'])
        self.customModel.summary()

        # Define evaluate result
        self.eval=[]

        # Function untuk train model
    def trainModel(self,train_data:Dataset):
        results=self.customModel.fit(train_data,epochs=self.parameters['epoch'],batch_size=self.parameters['batch_size'])
        return results
        # Function untuk evaluasi model
    def evaluateModel(self,test_data:ImageDataGenerator)->list:
        self.evaluate=self.customModel.evaluate(test_data)
        return self.evaluate
        # Function untuk prediksi real time
    def realTime(self,labels):
        # define the face cascade
        face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # define video capture -> load video dari webcam
        video_capture = cv.VideoCapture(0)

        while True:
            isTrue,frame=video_capture.read()
            
            # preprocess the frame
            frame_rgb=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            frame_resized=cv.resize(frame_rgb,dsize=(self.parameters['image_size']))
            frame_expanded=tf.expand_dims(frame_resized,axis=0)

            # make predictions
            predictions = self.customModel.predict(frame_expanded)
            label=np.squeeze(predictions.astype(int))
            label=np.squeeze(label)
            label=int(np.argmax(label))
            print(int(label),labels)
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame,str(labels[label]),org=(x, y-10),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=0.9,color=(255,0,0),thickness=2,lineType=cv.LINE_AA)
            cv.imshow('Video', frame)

            # Exit loop if 'q' is pressed
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        # Destroy window when the real time predictions end
        cv.destroyAllWindows()

