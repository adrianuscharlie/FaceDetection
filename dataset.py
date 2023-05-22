import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import matplotlib.pyplot as plt
import numpy as np


class Dataset():
    """ Create dataset custom class that contains image data generator from keras preprocessing.
    this class contains labels, batch_size of the dataset, and the core path of the dataset.
    this class can get the sample of image using the get sample function.
    
    """
    
    def __init__(self,path,image_size,batch_size):
        self.path=path
        self.labels={0:'Debby',1:'Dyah',2:'UnKnown'}
        self.batch_size=batch_size
        self.datagen=ImageDataGenerator(
            rescale=1.0/255.0,
            # rotation_range=10,
            # width_shift_range=0.1,
            # height_shift_range=0.1,
        )
        self.data=self.datagen.flow_from_directory(
            directory=path,target_size=image_size,
            class_mode='categorical',
            batch_size=self.batch_size
        )
        self.label=self.data.class_indices

    def getSample(self):
        image,labels=self.data.next()
        random_idx=random.randint(0,self.batch_size-1)
        img,label=image[random_idx],labels[random_idx]
        label=np.argmax(label)
        plt.title(f' Sample Image Of : {self.labels[int(label)]}')
        plt.axis('off')
        plt.imshow(img)
        plt.show(block=True)

    def __str__(self):
        return f'Train dataset \n Path : {self.path}\n Batch Size : {self.batch_size}'
        