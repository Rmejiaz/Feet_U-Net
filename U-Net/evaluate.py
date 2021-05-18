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

flags.DEFINE_string('test_images','./Dataset_Unificado/Test/Processed_Images/','path to the test images')
flags.DEFINE_string('test_masks','./Dataset_Unificado/Test/BinaryMasks/','path to the test masks')
flags.DEFINE_string('results','./results/','path to save the results')
flags.DEFINE_string('weights','./weights/cp-0094.ckpt','path of the weights to use')

def main(_argv):

    img_size = 128

    images_path = FLAGS.test_images
    masks_path = FLAGS.test_masks
    weights_path = FLAGS.weights
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
    model = get_model(output_channels=1, size=img_size)
    model.load_weights(weights_path)

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

    
    # Compute dice for each prediction

    Dice = [utils.DiceSimilarity(Y[i,:,:], Y_pred[i,:,:,0]) for i in range(Y.shape[0])]

    plt.figure(figsize=(16,9))
    plt.boxplot(np.array(Dice))
    title = f"""Dice score on test set
    \n Mean dice = ${round(np.array(Dice).mean(),3)} \pm {round(np.array(Dice).std(),3)}$"""
    plt.title(title)
    plt.savefig(os.path.join(results_path, 'TestDice.png'))
    plt.show()
    print(f"Mean dice score: {np.array(Dice).mean()} +/- {np.array(Dice).std()}")

if __name__ == '__main__':
    app.run(main)