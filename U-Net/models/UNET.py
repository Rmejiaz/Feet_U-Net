"""
U-Net implementation to segment feet and background
"""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from os import path
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.layers import ( Conv2D,  Conv2DTranspose, 
                                    Input,  MaxPool2D, 
                                    UpSampling2D,  Concatenate
                                    )


def get_model(output_channels = 1, size = 128, dropout = 0, trainable = False,name='UNET'):

    inputs = Input(shape= [size, size, 3])

    ## Encoder
    conv1 = Conv2D(64, 3, activation = "relu", padding = "same")(inputs)
    conv1 = Conv2D(64, 3, activation = "relu", padding = "same")(conv1)
    pool1 = MaxPool2D(pool_size = (2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = "relu", padding = "same")(pool1)
    conv2 = Conv2D(128, 3, activation = "relu", padding = "same")(conv2)
    pool2 = MaxPool2D(pool_size = (2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = "relu", padding = "same")(pool2)
    conv3 = Conv2D(256, 3, activation = "relu", padding = "same")(conv3)
    pool3 = MaxPool2D(pool_size = (2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = "relu", padding = "same")(pool3)
    conv4 = Conv2D(512, 3, activation = "relu", padding = "same")(conv4)
    pool4 = MaxPool2D(pool_size = (2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation = "relu", padding = "same")(pool4)
    conv5 = Conv2D(1024, 3, activation = "relu", padding = "same")(conv5)

    # Decoder

    up6 = UpSampling2D(size = (2, 2))(conv5)
    up6 = Conv2DTranspose(512, 2, padding = "same")(up6)
    conc6 = Concatenate()([conv4, up6])
    conv6 = Conv2D(512, 3, activation = "relu", padding = "same")(up6)
    conv6 = Conv2D(512, 3, activation = "relu", padding = "same")(conc6)

    up7 = UpSampling2D(size = (2, 2))(conv6)
    up7 = Conv2DTranspose(256, 2, padding = "same")(up7)
    conc7 = Concatenate()([conv3, up7])
    conv7 = Conv2D(256, 3, activation = "relu", padding = "same")(up7)
    conv7 = Conv2D(256, 3, activation = "relu", padding = "same")(conc7)

    up8 = UpSampling2D(size = (2, 2))(conv7)
    up8 = Conv2DTranspose(128, 2, padding = "same")(up8)
    conc8 = Concatenate()([conv2, up8])
    conv8 = Conv2D(128, 3, activation = "relu", padding = "same")(up8)
    conv8 = Conv2D(128, 3, activation = "relu", padding = "same")(conc8)

    up9 = UpSampling2D(size = (2, 2))(conv8)
    up9 = Conv2DTranspose(64, 2, padding = "same")(up9)
    conc9 = Concatenate()([conv1, up9])
    conv9 = Conv2D(64, 3, activation = "relu", padding = "same")(up9)
    conv9 = Conv2D(64, 3, activation = "relu", padding = "same")(conc9)

    out = Conv2D(1, 1, activation = "sigmoid")(conv9)
    model = keras.Model(inputs=inputs, outputs=out,name=name)

                  
    return model

if __name__ == '__main__':
    model = get_model()
    model.summary()
    tf.keras.utils.plot_model(model,to_file='data/model.png',show_shapes=False,show_layer_names=False)