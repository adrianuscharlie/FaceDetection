import tensorflow as tf # libraries untuk deep learning
import os # libraries 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
# Import model script to import model
from model import CustomModel
from dataset import Dataset


parameters={
    'image_size':(224,224), # untuk ukuran gambar yang mau dilatih
    'batch_size':32, # Banyaknya data yang mau dilatih dalam 1 epoch
    'epoch':10, # banyaknya iterasi training model
    'optimizer':'adam', # optimizer
    'loss_fn':'binary_crossentropy', # loss function
    'metrics':['accuracy'] # metrics evalaution -> untuk menilai seberapa tepat model
}


#model=CustomModel(parameters=parameters)

## Path dyah
train_dir='./Data/Train'
test_dir='./Data/Test'
train_dataset=Dataset(path=train_dir,image_size=(224,224),batch_size=1)
test_dataset=Dataset(path=test_dir,image_size=(224,224),batch_size=1)
model=CustomModel(parameters=parameters)
model.trainModel(train_dataset.data)
model.evaluateModel(test_dataset.data)
model.realTime()