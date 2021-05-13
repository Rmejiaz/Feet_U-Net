from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
from model import get_model
import matplotlib.pyplot as plt
import utils
import os 
from tensorflow.keras import backend as K
import numpy as np

flags.DEFINE_string('train_Dataset','./tfrecords/train-data.tfrecord','path to train Dataset')
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
    image_size = 224
    classes = 1
    train_Dataset_path = FLAGS.train_Dataset


    # Load train dataset
    
    # train_Dataset =  utils.load_tfrecord_dataset(train_Dataset_path,image_size)
    # train_Dataset = train_Dataset.shuffle(buffer_size=FLAGS.buffer_size)
    X_path = "./Dataset_Unificado/Train/Processed_Images"
    Y_path = "./Dataset_Unificado/Train/BinaryMasks"
    X = utils.load_data(X_path,size=image_size)/255.
    Y = utils.load_data(Y_path,size=image_size)
    Y = Y[:,:,:,0]
    Y = np.expand_dims(Y,axis=-1)

    # Split train and validation according to val_split

    # train_Dataset, val_Dataset = utils.split_dataset(train_Dataset,FLAGS.val_split)

    # Batch and prefetch

    # train_Dataset = train_Dataset.batch(FLAGS.batch_size,drop_remainder=True)
    # train_Dataset = train_Dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    # val_Dataset = val_Dataset.batch(FLAGS.batch_size,drop_remainder=True)
    # val_Dataset = val_Dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

    # Load model

    # n_batches_per_epoch = len(list(train_Dataset.as_numpy_iterator()))
    n_batches_per_epoch = X.shape[0]
    save_freq = int(n_batches_per_epoch * FLAGS.save_freq)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
                                                    filepath = checkpoint_path,
                                                    verbose=1,
                                                    save_weights_only = True,
                                                    save_freq = save_freq
                                                    )

    model = get_model(output_channels = classes, size = image_size)
    model.save_weights(checkpoint_path.format(epoch=0))
    model.compile(
                optimizer = 'adam',
                metrics = [tf.keras.metrics.MeanIoU(num_classes = 2),dice_coef],
                loss = tf.keras.losses.BinaryCrossentropy()
                )

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