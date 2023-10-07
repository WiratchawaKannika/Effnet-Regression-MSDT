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


'''
Function: Split train validation Set
'''

def split_train_valid(select_MSDT):
    ##  Step 1
    MSDTGropGLY = select_MSDT.groupby(['GLY','folderName']).count()
    MSDTGropGLY_ = MSDTGropGLY.reset_index(level=['GLY', 'folderName'])
    MSDTGropGLY_ = MSDTGropGLY_.iloc[:,:2]
    set_gly = list(set(MSDTGropGLY_['GLY']))
    
    ## Step 2 
    set_gly.sort()
    VALID_SPLIT = 0.16
    trainsetlst, validsetlst = [], []
    for G in set_gly:
        df_G  = MSDTGropGLY_[MSDTGropGLY_['GLY']==G]
        trainset, validset = train_test_split(df_G, test_size=VALID_SPLIT, random_state=42, shuffle=True)
        trainsetFolder = list(set(trainset["folderName"]))
        validsetFolder = list(set(validset["folderName"]))
        ## Step 3: Get Train set.
        for m in range(len(trainsetFolder)):
            select_MSDT_train = select_MSDT[select_MSDT["folderName"]==trainsetFolder[m]]
            trainsetlst.append(select_MSDT_train)
        ## Get Validation set.
        for n in range(len(validsetFolder)):
            select_MSDT_valid = select_MSDT[select_MSDT["folderName"]==validsetFolder[n]]
            validsetlst.append(select_MSDT_valid) 

    concat_train = pd.concat(trainsetlst, ignore_index=True)
    print(f"Data Train Set: Found {concat_train.shape[0]} images")
    concat_valid = pd.concat(validsetlst, ignore_index=True)
    print(f"Data Validation Set: Found {concat_valid.shape[0]} images")

    return concat_train, concat_valid



#Setting
#BATCH_SIZE = 16
#TARGET_SIZE = (600, 600)  
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      brightness_range=[0.5,1.5],
      shear_range=0.4,
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
