import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR) ## Disable tf Warnings
from absl import app, flags, logging
from absl.flags import FLAGS
from model import get_model
import tensorflow as tf
import matplotlib.pyplot as plt
import utils
import numpy as np
import cv2

flags.DEFINE_string("image_path", "./Dataset_CVAT2/JPEGImages/Test/177.jpg", "input image path")
flags.DEFINE_string("mask_path", None, "path to save the predicted mask (recomended file extension: png)")
flags.DEFINE_string("model_path", "./results/Model.h5", "weights to use or .h5 model")
flags.DEFINE_bool("show_results", True, "show prediction result")
flags.DEFINE_bool("clean_prediction", True, "post-process the prediction (remove all the small objects)")

def main(_argv):

    # Initialize variables
    img_path = FLAGS.image_path
    out_path = FLAGS.mask_path
    show_results = FLAGS.show_results
    clean_prediction = FLAGS.clean_prediction
    model_path = FLAGS.model_path
    img_size = 224

    # Read the image
    img = plt.imread(img_path)/255.
    X = tf.convert_to_tensor(img)
    X = tf.image.resize(X,(img_size,img_size))
    # X = X[:,:,0]
    X = tf.expand_dims(X,0)
    # X = tf.expand_dims(X,-1)

    # Load the model and the weights
    if (model_path[-4:] == 'ckpt'):
        model = get_model(output_channels=1, size=img_size)
        model.load_weights(model_path)

    elif (model_path[-2:] == 'h5'):
        model = tf.keras.models.load_model(model_path, custom_objects = {'dice_coef':utils.dice_coef, 'iou_coef':utils.iou_coef})

    # Make the prediction
    threshold = 0.5
    Y = model.predict(X)   
    Y = Y/Y.max()
    Y = np.where(Y>=threshold,1,0)

    Y = cv2.resize(Y[0,:,:,0], (img.shape[1],img.shape[0]), interpolation = cv2.INTER_NEAREST) # Resize the prediction to have the same dimensions as the input

    if clean_prediction:
        Y = utils.posprocessing(Y)

    if show_results:
        utils.display([img,Y[0,...,0]])

    if out_path != None:
        plt.imsave(out_path, Y, cmap='gray')

if __name__ == "__main__":
    app.run(main)