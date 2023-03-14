import PIL
import os
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageFile
from efficientnet.keras import center_crop_and_resize, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


ImageFile.LOAD_TRUNCATED_IMAGES = True


#Setting
#BATCH_SIZE = 16
#TARGET_SIZE = (600, 600)  
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      brightness_range=[0.5,1.5],
      shear_range=0.4,
      zoom_range=0.2,
      horizontal_flip=False,
      fill_mode='constant')

valid_datagen = ImageDataGenerator(rescale=1./255)

def Data_generator(IMAGE_SIZE, BATCH_SIZE, DF_TRAIN, DF_VAL):
    train_generator = train_datagen.flow_from_dataframe(
            dataframe = DF_TRAIN,
            directory = None,
            x_col = 'pathimg',
            y_col = 'MSDT',
            target_size = (IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            color_mode= 'rgb',
            class_mode='raw')

    val_generator = valid_datagen.flow_from_dataframe(
            dataframe = DF_VAL,
            directory = None,
            x_col = 'pathimg',
            y_col = 'MSDT',
            target_size = (IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            color_mode= 'rgb',
            class_mode='raw')
    return train_generator, val_generator 
