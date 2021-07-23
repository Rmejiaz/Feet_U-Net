# Add here SegNet model

import tensorflow as tf

def upsample(filters,size,strides=2,padding="same",batchnorm=False,dropout=0):

    layer = tf.keras.Sequential()
    layer.add(
        tf.keras.layers.Conv2DTranspose(filters,size,strides,padding,use_bias = False))

    if batchnorm:
        layer.add(tf.keras.layers.BatchNormalization())

    
    layer.add(tf.keras.layers.Dropout(dropout))

    layer.add(tf.keras.layers.ReLU())

    return layer

def get_encoder(input_shape=[None,None,3],name="encoder", trainable = True): 
    Input = tf.keras.layers.Input(shape=input_shape)
    base_model = tf.keras.applications.vgg16.VGG16(input_tensor=Input, include_top=False)
    layer_names = [
    'block1_conv2',   # 64x64
    'block2_conv2',   # 32x32
    'block3_conv3',   # 16x16
    'block4_conv3',  # 8x8
    'block5_conv3',      # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    encoder  = tf.keras.Model(inputs=Input, outputs=layers,name=name)
    encoder.trainable = trainable

    return encoder

def get_decoder(skips,dropout=0):
    up_stack = [
       # upsample(512, 3,dropout=dropout),  # 4x4 -> 8x8
        upsample(256, 3,dropout=dropout),  # 8x8 -> 16x16
        upsample(128, 3,dropout=dropout),  # 16x16 -> 32x32
        upsample(64, 3,dropout=dropout),   # 32x32 -> 64x64
    ]
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up,skip in zip(up_stack,skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x,skip])
    return x

def get_model(output_channels=1,size=224,name="VGG16",dropout=0, trainable = True):
    x = inputs = tf.keras.layers.Input(shape=[size,size,3])

    skips = get_encoder(input_shape=list(x.shape[1:]), trainable = True)(x)

    x = get_decoder(skips, dropout=dropout)

    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same',activation=tf.keras.activations.sigmoid)  #64x64 -> 128x128

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x,name=name)

if __name__ == '__main__':
    model = get_model(output_channels=2)
    model.summary()
    tf.keras.utils.plot_model(model,to_file='data/model.png',show_shapes=False,show_layer_names=False)
