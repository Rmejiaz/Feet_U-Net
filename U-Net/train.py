from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
from model import get_model
import matplotlib.pyplot as plt
import utils
import os 

flags.DEFINE_string('train_Dataset','./tfrecords/train-data.tfrecord','path to train Dataset')
flags.DEFINE_float('val_split',0.2,'size of the validation split')
flags.DEFINE_string('weights','./weights/','path to save the model weights')
flags.DEFINE_integer('buffer_size', 100, 'buffer')
flags.DEFINE_integer('batch_size', 5, 'batch size')
flags.DEFINE_integer('epochs', 10, 'Epochs')
flags.DEFINE_integer('save_freq', 5, 'frequency of epochs to save')

def main(_argv):

    # Initialize variables
    checkpoint_path = FLAGS.weights+'cp-{epoch:04d}.ckpt'
    image_size = 224
    classes = 2
    train_Dataset_path = FLAGS.train_Dataset


    # Load train dataset
    train_Dataset =  utils.load_tfrecord_dataset(train_Dataset_path,image_size)
    train_Dataset = train_Dataset.shuffle(buffer_size=FLAGS.buffer_size)
    train_Dataset = train_Dataset.batch(FLAGS.batch_size,drop_remainder=True)
    train_Dataset = train_Dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

    # Split train and validation according to val_split

    train_Dataset, val_Dataset = utils.split_dataset(train_Dataset,FLAGS.val_split)

    # Load model
    n_batches_per_epoch = len(list(train_Dataset.as_numpy_iterator()))
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
                metrics = ['accuracy'], 
                loss = tf.keras.losses.SparseCategoricalCrossentropy()
                )

    # Train the model
    model_history = model.fit(
                            train_Dataset,
                            epochs =FLAGS.epochs,
                            validation_data = val_Dataset,
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