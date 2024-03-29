import tensorflow as tf
from skimage.io import imread
from keras.utils import generic_utils
from keras import layers
from keras import models
from tensorflow.keras import optimizers
from keras.optimizers import Adam
from efficientnet.keras import EfficientNetB7 as Net
#load Check point
from tensorflow.keras.models import load_model



def build_modelB7(fine_tune):
    """
    :param fine_tune (bool): Whether to train the hidden layers or not.
    """
    
    conv_base = Net(weights='imagenet')
    height = width = conv_base.input_shape[1]
    input_shape = (height, width, 3)
    #print(f"Input Shape: {input_shape}")
    
    # loading pretrained conv base model
    conv_base = Net(weights='imagenet', include_top=False, input_shape=input_shape)
    # create new model with a new classification layer
    x = conv_base.output  
    global_average_layer = layers.GlobalAveragePooling2D(name = 'head_pooling')(x)
    # add 1 layer
    dropout_layer_1 = layers.BatchNormalization(name = 'batchNormalization')(global_average_layer)
    dropout_layer_2 = layers.Dropout(0.40,name = 'head_dropout')(dropout_layer_1)
    dense_1 = layers.Dense(64, activation='softmax',name = 'pred_dense_1')(dropout_layer_2)
    dense_2 = layers.Dense(32, activation='softmax',name = 'pred_dense_2')(dense_1)
    prediction_layer = layers.Dense(1, activation='linear',name = 'prediction_layer')(dense_2)
    ### lastlayer 
    model = models.Model(inputs= conv_base.input, outputs=prediction_layer, name = 'EffNet_Regression') 

    print('This is the number of trainable layers '
          'before freezing the conv base:', len(model.trainable_weights))

    if fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for layer in conv_base.layers:
            layer.trainable = False

    print('This is the number of trainable layers '
           'after freezing the conv base:', len(model.trainable_weights))
    print('-'*125)

    return input_shape, model


def loadresumemodel(model_dir):
    model = load_model(model_dir)
    height = width = model.input_shape[1]
    input_shape = (height, width, 3)
    
    return input_shape, model



def modelR2Unfreze(model_dir):
    model = load_model(model_dir)
    height = width = model.input_shape[1]
    input_shape = (height, width, 3)
    print('This is the number of trainable layers '
          'before freezing the conv base:', len(model.trainable_weights))
    model.trainable = True
    set_trainable = False
    for layer in model.layers:
        if layer.name.startswith('block7'):
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    print('This is the number of trainable layers '
          'after freezing the block7 Layer:', len(model.trainable_weights))

    return input_shape, model



def model_block5Unfreze(model_dir):
    model = load_model(model_dir)
    height = width = model.input_shape[1]
    input_shape = (height, width, 3)
    print('This is the number of trainable layers '
          'before freezing the conv base:', len(model.trainable_weights))
    model.trainable = True
    set_trainable = False
    for layer in model.layers:
        if layer.name == 'block5a_se_excite':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    print('This is the number of trainable layers '
          'after freezing the block5a_se_excite Layer:', len(model.trainable_weights))

    return input_shape, model











