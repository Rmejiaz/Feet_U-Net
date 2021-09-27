# convert a model from .h5 to .tflite to run in raspberry

import os
from absl import flags, app, logging
from absl.flags import FLAGS
import tensorflow as tf
import utils

flags.DEFINE_string('input_model',None,'relative path of the .h5 model to convert')
flags.DEFINE_string('output',None,'relative path to save the tflite model')


def convert_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()

def save_model(model, path):
    """
    Save a tflite model to a specified path
    """
    with open(path, 'wb') as f:
        f.write(model)


def main(_argv):
    h5_path = FLAGS.input_model
    tflite_path = FLAGS.output

    try:
        h5_model = tf.keras.models.load_model(path = h5_path, custom_objects={'dice_coef':utils.dice_coef, 'iou_coef':utils.iou_coef})
    except:
        print(f"Cannot open {path}")
        break
    
    if (tflite_path == None):
        tflite_path = h5_path[:-2] + "tflite"

    tflite_model = convert_model(h5_model)

    save_model(tflite_model, tflite_path)

if __name__ == '__main__':
    app.run(main)



