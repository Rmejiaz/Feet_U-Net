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
flags.DEFINE_string("weights", "./weights/cp-0010.ckpt", "weights parameters path")
flags.DEFINE_string("labels", "./Dataset_CVAT/labelmap.txt", "path to the annotation file")
flags.DEFINE_bool("show_results", True, "show prediction result")



def main(_argv):

    # Initialize variables
    img_path = FLAGS.image_path
    out_path = FLAGS.mask_path
    weights_path = FLAGS.weights
    show_results = FLAGS.show_results
    img_size = 224

    # Read the image
    img = plt.imread(img_path)/255.
    X = tf.convert_to_tensor(img)
    X = tf.image.resize(X,(img_size,img_size))
    # X = X[:,:,0]
    X = tf.expand_dims(X,0)
    # X = tf.expand_dims(X,-1)

    # Load the model and the weights
    model = get_model(output_channels=1, size=img_size)
    model.load_weights(weights_path)

    # Make the prediction
    threshold = 0.5
    Y = model.predict(X)   
    Y = Y/Y.max()
    Y = np.where(Y>=threshold,1,0)
    Y = cv2.resize(Y[0], (img.shape[1],img.shape[0]), interpolation = cv2.INTER_NEAREST) # Resize the prediction to have the same dimensions as the input

    if show_results:
        utils.display([img,Y])

    if out_path != None:
        plt.imsave(out_path, Y, cmap='gray')

if __name__ == "__main__":
    app.run(main)
