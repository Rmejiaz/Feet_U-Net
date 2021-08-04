from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, concatenate, Dropout, Lambda, Conv2DTranspose, Add
import numpy as np
import tensorflow as tf
import os


def get_model(output_channels = 1, size = 224, name = "FCN", dropout = 0):

    b = 4
    i = Input(shape= (size, size, 3))
    ## Block 1
    x = Conv2D(2**b, (3, 3), activation='elu', padding='same', name='block1_conv1')(i)
    x = Conv2D(2**b, (3, 3), activation='elu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x

    # Block 2
    x = Conv2D(2**(b+1), (3, 3), activation='elu', padding='same', name='block2_conv1')(x)
    x = Conv2D(2**(b+1), (3, 3), activation='elu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # Block 3
    x = Conv2D(2**(b+2), (3, 3), activation='elu', padding='same', name='block3_conv1')(x)
    x = Conv2D(2**(b+2), (3, 3), activation='elu', padding='same', name='block3_conv2')(x)
    x = Conv2D(2**(b+2), (3, 3), activation='elu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    pool3 = x

    # Block 4
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block4_conv1')(x)
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block4_conv2')(x)
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block4_conv3')(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block5_conv1')(pool4)
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block5_conv2')(x)
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block5_conv3')(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    conv6 = Conv2D(2048 , (7, 7) , activation='elu' , padding='same', name="conv6")(pool5)
    conv6 = Dropout(0.5)(conv6)
    conv7 = Conv2D(2048 , (1, 1) , activation='elu' , padding='same', name="conv7")(conv6)
    conv7 = Dropout(0.5)(conv7)

    pool4_n = Conv2D(output_channels, (1, 1), activation='elu', padding='same')(pool4)
    u2 = Conv2DTranspose(output_channels, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7)
    u2_skip = Add()([pool4_n, u2])

    pool3_n = Conv2D(output_channels, (1, 1), activation='elu', padding='same')(pool3)
    u4 = Conv2DTranspose(output_channels, kernel_size=(2, 2), strides=(2, 2), padding='same')(u2_skip)
    u4_skip = Add()([pool3_n, u4])

    o = Conv2DTranspose(output_channels, kernel_size=(8, 8), strides=(8, 8), padding='same', activation='sigmoid')(u4_skip)

    model = Model(inputs=i, outputs=o, name=name)
    
    return model

if __name__ == '__main__':
    model = get_model(output_channels=1)
    model.summary()
    tf.keras.utils.plot_model(model,to_file='data/model.png',show_shapes=False,show_layer_names=False)
