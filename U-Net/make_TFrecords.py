import tensorflow as tf
import numpy as np
from utils import *
from absl import flags, app, logging
from absl.flags import FLAGS
import os


flags.DEFINE_string('img_path', './Dataset_Unificado/JPEGImages/', 'path for input images')
flags.DEFINE_string('mask_path', './Dataset_Unificado/SegmentationClass/', 'path for label images')
flags.DEFINE_string('labels', 'labelmap.txt', 'path to the labels description')


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_example(img, mask, labels):
    """Creates a tensorflow Example from an image with its mask.
    Parameters
    -----------
    img : str
        path to the image
    mask : str
        path to the mask
    labels : dict
        dict with the corresponding rgb mask values of the labels
    Returns
    --------
    tf.train.Example   
    """
    encoded_img = tf.io.read_file(img)
    encoded_mask = tf.io.read_file(mask)
    ## mask preprocessing ##
    decoded_mask = tf.io.decode_image(encoded_mask)
    mask = mask2categorical(decoded_mask, labels)
    mask = tf.expand_dims(mask, axis=-1)

    encoded_mask = tf.io.encode_png(mask) # Re-encoding the mask

    example = tf.train.Example(
                features=tf.train.Features(feature={
                    'image': bytes_list_feature(encoded_img.numpy()),
                    'mask': bytes_list_feature(encoded_mask.numpy())
                    }))
    return example


def main(_argv):

    # Initialize all variables

    PATH_IMG = FLAGS.img_path
    PATH_MASK = FLAGS.mask_path
    
    LABELS = parse_labelfile(FLAGS.labels)
    img_path = [os.path.join(PATH_IMG, imgs) for imgs in np.sort(os.listdir(PATH_IMG))]
    mask_path = [os.path.join(PATH_MASK, imgs) for imgs in np.sort(os.listdir(PATH_MASK))]


    # Create tfrecords file

    if not 'tfrecords' in os.listdir():
        os.mkdir('tfrecords')

    TRAIN_TFRECORD = os.path.join('./tfrecords', "train-data.tfrecord")
    # Create and fill the train-data.tfrecord with the examples

    tf_record_train = tf.io.TFRecordWriter(TRAIN_TFRECORD)
    for img, mask in zip(img_path, mask_path):
        example = create_example(img,mask,LABELS)
        tf_record_train.write(example.SerializeToString())
    tf_record_train.close()

if __name__ == "__main__":
    app.run(main)