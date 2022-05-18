from absl import app, flags, logging
from absl.flags import FLAGS
from matplotlib.style import available
import tensorflow as tf
from model import get_model, MODELS
import matplotlib.pyplot as plt
import utils
import os 
from tensorflow.keras import backend as K
import numpy as np

from utils import DiceSimilarity, jaccard

flags.DEFINE_string('imgs_path','./Dataset_3/Train/JPEGImages','path to the training images')
flags.DEFINE_string('masks_path','./Dataset_3/Train/SegmentationClass','path to the training masks')
flags.DEFINE_string('val_masks','./Dataset_3/Test/SegmentationClass','path to the validation masks')
flags.DEFINE_string('val_imgs','./Dataset_3/Test/JPEGImages','path to the validation images')
flags.DEFINE_float('val_split',0,'size of the validation split')
flags.DEFINE_string('weights','./weights/','path to save the model weights')
flags.DEFINE_integer('buffer_size', 100, 'buffer')
flags.DEFINE_integer('batch_size', 5, 'batch size')
flags.DEFINE_integer('epochs', 10, 'Epochs')
flags.DEFINE_integer('save_freq', 5, 'frequency of epochs to save')
flags.DEFINE_string('save_model', './results/Model.h5', 'path to save the model in (.h5 format is recommended')
flags.DEFINE_float('dropout',0, 'decoder dropout')
flags.DEFINE_boolean('no_tl', True, 'whether or not to train all the parameters' )

flags.DEFINE_boolean('weighted_classes', False, 'whether or not to train all the parameters' )
flags.DEFINE_integer('img_size', 224, 'image size used to resize the images')

available_models = list(MODELS.keys())
flags.DEFINE_enum('model',available_models[0],available_models,'Models Availables')

def main(_argv):

    # Initialize variables
    checkpoint_path = FLAGS.weights+'cp-{epoch:04d}.ckpt'
    image_size = int(FLAGS.img_size)
    classes = 1
    X_path = FLAGS.imgs_path
    Y_path = FLAGS.masks_path
    X_val_path = FLAGS.val_imgs
    Y_val_path = FLAGS.val_masks
    val_split = FLAGS.val_split
    save_model = FLAGS.save_model
    dropout = FLAGS.dropout
    trainable = FLAGS.no_tl
    # Load train dataset
    
    X = utils.load_data(X_path,size=image_size)
    Y = utils.load_data(Y_path,size=image_size)
    Y = Y[:,:,:,0]
    Y = np.expand_dims(Y,axis=-1)

    # Load test dataset
    try:
        X_val = utils.load_data(X_val_path,size=image_size)
        Y_val = utils.load_data(Y_val_path, size=image_size)
        Y_val = Y_val[:,:,:,0]
        Y_val = np.expand_dims(Y_val,axis=-1)
    except:
        pass 

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

    model = get_model(model=FLAGS.model,output_channels = classes, size = image_size, dropout=dropout, trainable = trainable)
    model.save_weights(checkpoint_path.format(epoch=0))
    model.compile(
                optimizer = tf.keras.optimizers.Adam(),
                metrics = [utils.dice_coef, utils.iou_coef],
                loss = tf.keras.losses.BinaryCrossentropy()
                )

    # Train the model
    if FLAGS.weighted_classes: 
        class_weights = tf.constant([1,7])
        sample_weights = tf.gather(class_weights, indices=tf.cast(Y, tf.int32))
    else:
        sample_weights = None 

    if val_split == 0:
        model_history = model.fit(
                                x = X,
                                y = Y,
                                validation_data = (X_val,Y_val),
                                epochs =FLAGS.epochs,
                                batch_size = FLAGS.batch_size,
                                callbacks = [cp_callback],
                                sample_weight = sample_weights
                                )
    
    else:
        model_history = model.fit(
                                x = X,
                                y = Y,
                                validation_split = val_split,
                                epochs =FLAGS.epochs,
                                batch_size = FLAGS.batch_size,
                                callbacks = [cp_callback],
                                sample_weight = sample_weights
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

    # Save the model in .h5

    model.save(save_model)

if __name__ == '__main__':
    app.run(main)