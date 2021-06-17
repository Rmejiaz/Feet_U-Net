"""
Do a prediciton of a model and evaluate it
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
from absl import app, flags, logging
from absl.flags import FLAGS
from model import get_model
import cv2
import os

flags.DEFINE_string('test_images','./Dataset_CVAT2/JPEGImages/Test/','path to the test images')
flags.DEFINE_string('test_masks','./Dataset_CVAT2/SegmentationClass/Test','path to the test masks')
flags.DEFINE_string('results','./results/','path to save the results')
flags.DEFINE_string('model_path','./weights/cp-0100.ckpt','path of the weights to use')

def main(_argv):

    img_size = 224

    images_path = FLAGS.test_images
    masks_path = FLAGS.test_masks
    model_path = FLAGS.model_path
    results_path = FLAGS.results
    imgs = utils.load_data(path = images_path, size = None)

    # resize the images
    X = []
    for i in range(imgs.shape[0]):
        photo = tf.image.resize(imgs[i], (img_size, img_size))
        X.append(photo)

    X = np.array(X)

    Y = utils.load_data(path = masks_path, size = None)
    Y = Y[:,:,:,0]
    
    # Load the model and the weights
    if (model_path[-4:] == 'ckpt'):
        model = get_model(output_channels=1, size=img_size)
        model.load_weights(model_path)

    elif (model_path[-2:] == 'h5'):
        model = tf.keras.models.load_model(model_path)

    # Make the prediction
    threshold = 0.5
    Y_pred = model.predict(X)   
    
    Y_pred = Y_pred/Y_pred.max()
    Y_pred = np.where(Y_pred>=threshold,1,0)

    # Resize predictions
    Y_pred_r = []
    for i in range(Y_pred.shape[0]):
        resized = cv2.resize(Y_pred[i], (imgs.shape[2], imgs.shape[1]), interpolation = cv2.INTER_NEAREST)
        Y_pred_r.append(resized)

    Y_pred = np.array(Y_pred_r) 
    Y_pred = np.expand_dims(Y_pred,axis=-1)

    # Save predictions

    if not 'Predictions' in os.listdir(results_path):
        os.mkdir(os.path.join(results_path,'Predictions')) # Create predictions directory

    names = os.listdir(masks_path)
    names.sort()

    for i, name in enumerate(names):
        plt.imsave(os.path.join(results_path, 'Predictions', name), Y_pred[i,:,:,0]+imgs[i,:,:,0], cmap='gray')

    
    # Compute and plot dice and jaccard for each prediction

    Dice = np.array([utils.DiceSimilarity(Y[i,:,:], Y_pred[i,:,:,0]) for i in range(Y.shape[0])])
    Jaccard = np.array([utils.jaccard(Y[i,:,:], Y_pred[i,:,:,0]) for i in range(Y.shape[0])])

    plt.figure(figsize=(16,9))
    plt.boxplot([Dice, Jaccard])
    title = f"""Dice and Jaccard scores on test set
    \n Mean Dice = ${round(Dice.mean(),3)} \pm {round(Dice.std(),3)}$
    \n Mean Jaccard = ${round(Jaccard.mean(),3)} \pm {round(Jaccard.std(),3)}$"""
    plt.title(title,fontsize=10)
    plt.xticks(ticks=[1,2],labels = ['Dice', 'Jaccard'])
    plt.savefig(os.path.join(results_path, 'TestScores.png'))
    plt.show()
    
if __name__ == '__main__':
    app.run(main)