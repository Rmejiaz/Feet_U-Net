"""
Training using the model defined in model2.py
"""

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import matplotlib.pyplot as plt
import utils
import os 
from tensorflow.keras import backend as K
import numpy as np
from model import UNET2D
from tensorflow.keras.utils import plot_model

flags.DEFINE_string('imgs_path','./Dataset_Unificado/Train/Processed_Images','path to the training images')
flags.DEFINE_string('masks_path','./Dataset_Unificado/Train/BinaryMasks','path to the training masks')
flags.DEFINE_string('val_masks','./Dataset_Unificado/Test/BinaryMasks','path to the validation masks')
flags.DEFINE_string('val_imgs','./Dataset_Unificado/Test/Processed_Images','path to the validation images')
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
    
    X_path = FLAGS.imgs_path
    Y_path = FLAGS.masks_path
    X_val_path = FLAGS.val_imgs
    Y_val_path = FLAGS.val_masks
    val_split = FLAGS.val_split

    # Load train dataset
    
    X = utils.load_data(X_path,size=image_size)
    Y = utils.load_data(Y_path,size=image_size)
    
    X = X[:,:,:,0]
    X = np.expand_dims(X,axis=-1)
    
    Y = Y[:,:,:,0]
    Y = np.expand_dims(Y,axis=-1)

    # Load test dataset
    try:
        X_val_path = "./Dataset_Unificado/Test/Processed_Images"
        Y_val_path = "./Dataset_Unificado/Test/BinaryMasks"
        X_val = utils.load_data(X_val_path,size=image_size)
        Y_val = utils.load_data(Y_val_path, size=image_size)
        Y_val = Y_val[:,:,:,0]
        Y_val = np.expand_dims(Y_val,axis=-1)
    except:
        pass

    ## Load the model

    model = UNET2D(input_size = (image_size,image_size,n_channels))

    ## Compile model
    
    n_batches_per_epoch = X.shape[0]
    save_freq = int(n_batches_per_epoch * FLAGS.save_freq)

    ## Create callbacks
    # save weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
                                                    filepath = checkpoint_path,
                                                    verbose=1,
                                                    save_weights_only = True,
                                                    save_freq = save_freq
                                                    )
    
    model.save_weights(checkpoint_path.format(epoch=0))

    model.compile(
                optimizer = tf.keras.optimizers.RMSprop(),
                metrics = [utils.dice_coef, utils.iou_coef],
                loss = tf.keras.losses.BinaryCrossentropy()
                )
    
    # early stop callback

    # Train the model
    if val_split == 0:
        model_history = model.fit(
                                x = X,
                                y = Y,
                                validation_data = (X_val,Y_val),
                                epochs =FLAGS.epochs,
                                batch_size = FLAGS.batch_size,
                                callbacks = [cp_callback]
                                )
    else:
        model_history = model.fit(
                                x = X,
                                y = Y,
                                validation_split = val_split,
                                epochs =FLAGS.epochs,
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
    plt.ylim(0,1)
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

    

