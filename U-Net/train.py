"""
Training using the model defined in model2.py
"""

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
from model import get_model
import matplotlib.pyplot as plt
import utils
import os 
from tensorflow.keras import backend as K
import numpy as np
from model2 import UNET2D
from tensorflow.keras.utils import plot_model


flags.DEFINE_float('val_split',0.2,'size of the validation split')
flags.DEFINE_string('weights','./weights/','path to save the model weights')
flags.DEFINE_integer('buffer_size', 100, 'buffer')
flags.DEFINE_integer('batch_size', 5, 'batch size')
flags.DEFINE_integer('epochs', 10, 'Epochs')
flags.DEFINE_integer('save_freq', 5, 'frequency of epochs to save')


def dice_coef(y_true, y_pred):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection)/(union), axis=0)
  return dice


def main(_argv):

    # Initialize variables
    checkpoint_path = FLAGS.weights+'cp-{epoch:04d}.ckpt'

    image_size = 128
    n_channels = 1
    ## Load the data

    X_path = "./Dataset_Unificado/Train/Processed_Images"
    Y_path = "./Dataset_Unificado/Train/BinaryMasks"
    X = utils.load_data(X_path,size=image_size)
    Y = utils.load_data(Y_path,size=image_size)

    X = X[:,:,:,0]
    Y = Y[:,:,:,0]
    X = np.expand_dims(X,axis=-1)
    Y = np.expand_dims(Y,axis=-1)

    ## Load the model

    model = UNET2D(input_size = (image_size,image_size,n_channels))

    ## Compile model
    
    n_batches_per_epoch = X.shape[0]
    save_freq = int(n_batches_per_epoch * FLAGS.save_freq)

    model.compile(
                optimizer = tf.keras.optimizers.RMSprop(),
                metrics = [tf.keras.metrics.MeanIoU(num_classes = 2),dice_coef,tf.keras.metrics.BinaryAccuracy(name='BinaryAccuracy')],
                loss = tf.keras.losses.BinaryCrossentropy()
                )


    ## Create callbacks
    # save weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
                                                    filepath = checkpoint_path,
                                                    verbose=1,
                                                    save_weights_only = True,
                                                    save_freq = save_freq
                                                    )
    
    model.save_weights(checkpoint_path.format(epoch=0))
    
    # early stop callback

    # Train the model
    model_history = model.fit(
                            x = X,
                            y = Y,
                            epochs =FLAGS.epochs,
                            validation_split = FLAGS.val_split,
                            batch_size = FLAGS.batch_size,
                            callbacks = [cp_callback]
                             )

    # Create the results directory
    if  not 'results' in os.listdir():
        os.mkdir('results')
    
    # Create the history figure
    plt.figure(figsize=(16,9))
    for i in model_history.history:
        plt.plot(model_history.history[i],label=i)
    plt.title('Model history')
    plt.legend()
    plt.grid()

    # Save the figure
    i = 0
    flag = True
    while(flag==True):
        if (f'history{i}' in os.listdir('results')):
            i+=1
        else:
            plt.savefig(f'results/history{i}')
            flag=False

    plt.show()

if __name__ == '__main__':
    app.run(main)

    

