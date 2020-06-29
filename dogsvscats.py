
# importing libraries 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib.image import imread

import os

data_dir = 'PATH TO DATA'

os.listdir(data_dir)


test_path = data_dir+'/test_set/'
train_path = data_dir+'/training_set/'

os.listdir(test_path)
os.listdir(train_path+'cats')[0]

dim1 = []
dim2 = []

for image_filename in os.listdir(test_path+'cats'):
    
    img= imread(test_path+'cats/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)

#print(dim1)

print(np.mean(d1))
print(np.mean(d2))

iamge_shape = (244,346,3)

image_gen = ImageDataGenerator(rescale = 1. / 255)
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)

model = Sequential() 
model.add(Conv2D(32, (2, 2), input_shape = iamge_shape)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Conv2D(32, (2, 2), input_shape = iamge_shape)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Conv2D(32, (2, 2), input_shape = iamge_shape)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Flatten()) 
model.add(Dense(64)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(1)) 
model.add(Activation('sigmoid')) 
model.compile(loss ='binary_crossentropy', 
                     optimizer ='adam', 
                   metrics =['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor = 'val_loss', patience=2)
batch_size_1 = 16

iamge_shape[:2]

train_image_gen = image_gen.flow_from_directory(train_path,target_size=iamge_shape[:2],color_mode='rgb',batch_size=batch_size_1, class_mode='binary')
test_image_gen = image_gen.flow_from_directory(test_path,target_size=iamge_shape[:2],color_mode='rgb',batch_size=batch_size_1, class_mode='binary', shuffle=False)

train_image_gen.class_indices

from sklearn.preprocessing import LabelEncoder
results = model.fit(train_image_gen, epochs=20, validation_data=test_image_gen,callbacks=[early_stop])
