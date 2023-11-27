import os
import tensorflow as tf
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
from tensorflow.keras import callbacks
from keras.callbacks import Callback
import pandas as pd
from keras.utils import generic_utils
from keras import layers
from keras import models
from tensorflow.keras import optimizers
from keras.optimizers import Adam
from model import build_modelB7, loadresumemodel, modelR2Unfreze, model_block5Unfreze
from DataLoader import Data_generator
#load Check point
from tensorflow.keras.models import load_model
import argparse

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()

    
def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir,run_id)


def get_preds_and_labels(model, generator):
    """
    Get predictions and labels from the generator
    """
    generator.reset()
    preds = []
    labels = []
    for _ in range(int(np.ceil(generator.samples / BATCH_SIZE))):
        x, y = next(generator)
        preds.append(model.predict(x))
        labels.append(y)
    # Flatten list of numpy arrays
    return np.concatenate(preds).ravel(), np.concatenate(labels).ravel()


def main():
     # construct the argument parser
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train our network for')
    my_parser.add_argument('--gpu', type=int, default=0, help='Number GPU 0,1')
    my_parser.add_argument('--data_path', type=str, default='/home/kannika/codes_AI/Rheology2023/MSDTGLY7Level10fold_datatrain_tSecond.csv')
    my_parser.add_argument('--save_dir', type=str, help='Main Output Path', default="/media/SSD/rheology2023/EffNetB7Model/Regression/tensorflow_GLY7Level")
    my_parser.add_argument('--name', type=str, help='Name to save output in save_dir')
    my_parser.add_argument('--R', type=int, help='[1:R1, 2:R2]')
    my_parser.add_argument('--fold', type=int, help='1-10')
    my_parser.add_argument('--lr', type=float, default=1e-4)
    my_parser.add_argument('--batchsize', type=int, default=16)
    my_parser.add_argument('--resume', action='store_true')
    my_parser.add_argument('--checkpoint_dir', type=str ,default=".")
    my_parser.add_argument('--tensorName', type=str ,default="Mylogs_tensor")
    #my_parser.add_argument('--checkpointerName', type=str ,default="checkpointer")
    my_parser.add_argument('--epochendName', type=str ,default="on_epoch_end")
    my_parser.add_argument('--FmodelsName', type=str ,default="models")
    
    args = my_parser.parse_args()
    
    ## set gpu
    gpu = args.gpu
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}" 
    physical_devices = tf.config.list_physical_devices('GPU') 
    print("Num GPUs:", len(physical_devices))
    
    ## get my_parser
    Fold = f"fold{args.fold}"
    save_dir = args.save_dir
    name = args.name
    R = args.R
    _R = f'R{R}'
    root_base = f'{save_dir}/{name}/{Fold}/{_R}'
    os.makedirs(root_base, exist_ok=True)
    data_path = args.data_path
    BATCH_SIZE = args.batchsize
    ## train seting
    lr = args.lr
    epochs = args.epochs
    
    ### Create Model 
    if args.resume :
         input_shape, model = loadresumemodel(args.checkpoint_dir)
    elif args.R == 2:
        input_shape, model = model_block5Unfreze(args.checkpoint_dir)
    elif args.resume and args.R == 2:
        input_shape, model = loadresumemodel(args.checkpoint_dir)
    else:    
        input_shape, model = build_modelB7(fine_tune=True)
    ##get images size 
    IMAGE_SIZE = input_shape[0]
    model.summary()
    print('='*100)
    
    ## import dataset
    DF = pd.read_csv(data_path)
    DF_TRAIN = DF[DF["fold"]!= args.fold].reset_index(drop=True)
    DF_VAL = DF[DF["fold"]== args.fold].reset_index(drop=True)
    ### Get data Loder
    train_generator, val_generator = Data_generator(IMAGE_SIZE, BATCH_SIZE, DF_TRAIN, DF_VAL)
    
    ## Set mkdir TensorBoard 
    ##root_logdir = f'/media/SSD/rheology2023/VitModel/Regression/tensorflow/ExpTest/R1/Mylogs_tensor/'
    root_logdir = f"{root_base}/{args.tensorName}"
    os.makedirs(root_logdir, exist_ok=True)
    ### Run TensorBoard 
    run_logdir = get_run_logdir(root_logdir)
    tensorboard_cb = callbacks.TensorBoard(log_dir=run_logdir)
    
    modelNamemkdir = f"{root_base}/{args.FmodelsName}"
    os.makedirs(modelNamemkdir, exist_ok=True)
    modelName = f'modelRegress_EffNetB7_RheologyGLY7Level_{Fold}_{_R}.h5'
    Model2save = f'{modelNamemkdir}/{modelName}'
    
    root_Metrics = f'{root_base}/{args.epochendName}/'
    os.makedirs(root_Metrics, exist_ok=True)
    class Metrics(Callback):
            def on_epoch_end(self, epochs, logs={}):
                self.model.save(f'{root_Metrics}{modelName}')
                return
    
    # For tracking Quadratic Weighted Kappa score and saving best weights
    metrics = Metrics()
    
    #Training
    model.compile(loss='mse',
                  optimizer=Adam(lr, decay=lr),
                  metrics=['mse'])
    
#     checkpoint_filepath = f"{root_base}/{args.checkpointerName}"
#     os.makedirs(checkpoint_filepath, exist_ok=True)
    
#     model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_freq='epoch', 
#                                                                        ave_weights_only=False, monitor="val_mean_absolute_percentage_error")
    
    
    ## Fit model 
    model.fit(train_generator, epochs=epochs, validation_data=val_generator,
                callbacks = [metrics, tensorboard_cb])
    
    # Save model as .h5        
    model.save(Model2save)
    ### print
    print(f"Save Linear regression (EfficientNet) as: {Model2save}")
    print(f"*"*100)
    

## Run Function 
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    


