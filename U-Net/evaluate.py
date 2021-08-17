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
from tabulate import tabulate
from datetime import date

flags.DEFINE_string('test_images','./Dataset_CVAT2/Test/JPEGImages/','path to the test images')
flags.DEFINE_string('test_masks','./Dataset_CVAT2/Test/SegmentationClass','path to the test masks')
flags.DEFINE_string('results_path', './results', 'path to save the results')
flags.DEFINE_string('model_path','./results/Model.h5','path of the weights to use')
flags.DEFINE_string('model_name', 'U-Net Mobilenetv2', 'name of the model used to train')

def main(_argv):


    images_path = FLAGS.test_images
    masks_path = FLAGS.test_masks
    model_path = FLAGS.model_path
    imgs = utils.load_data(path = images_path, size = None)
    model_name = FLAGS.model_name
    results_path = FLAGS.results_path

    img_size = 224
    # Load the model and the weights
    if (model_path[-4:] == 'ckpt'):
        model = get_model(output_channels=1, size=img_size)
        model.load_weights(model_path)

    elif (model_path[-2:] == 'h5'):
        model = tf.keras.models.load_model(model_path, custom_objects = {'dice_coef':utils.dice_coef, 'iou_coef':utils.iou_coef})


    img_size = model.input_shape[1]

    # resize the images
    X = []
    for i in range(imgs.shape[0]):
        photo = tf.image.resize(imgs[i], (img_size, img_size))
        X.append(photo)

    X = np.array(X)

    Y = utils.load_data(path = masks_path, size = None)
    Y = Y[:,:,:,0]
    
    

    # Make the prediction
    threshold = 0.5
    Y_pred = model.predict(X)   
    
    Y_pred = Y_pred/Y_pred.max()
    Y_pred = np.where(Y_pred>=threshold,1,0)  #threshold predictions
    Y = np.where(Y > 0.5, 1, 0)
    
    # Compute scores

    # Resize Predictions

    Y_pred = np.array([cv2.resize(Y_pred[i,:,:,0], (Y.shape[2],Y.shape[1]), interpolation = cv2.INTER_NEAREST) for i in range(Y_pred.shape[0])]) # Resize the prediction to have the same dimensions as the input

    # without refinement

    sens, specs, precs, dices, jaccards = [], [], [], [], []

    for i in range(Y.shape[0]):
        sens.append(utils.mask_sensitivy(Y[i],Y_pred[i]))
        specs.append(utils.mask_specificity(Y[i],Y_pred[i]))
        precs.append(utils.mask_precision(Y[i],Y_pred[i]))
        dices.append(utils.DiceSimilarity(Y[i].reshape(-1), Y_pred[i].reshape(-1)))
        jaccards.append(utils.jaccard(Y[i], Y_pred[i]))



    # with refinement

    sens2, specs2, precs2, dices2, jaccards2 = [], [], [], [], []

    Y_pred_transformed = np.array([utils.remove_small_objects(Y_pred[i]) for i in range(Y_pred.shape[0])])   # Refine the predictions (remove small objects)

    for i in range(Y.shape[0]):
        sens2.append(utils.mask_sensitivy(Y[i],Y_pred_transformed[i]))
        specs2.append(utils.mask_specificity(Y[i], Y_pred_transformed[i]))
        precs2.append(utils.mask_precision(Y[i], Y_pred_transformed[i]))
        dices2.append(utils.DiceSimilarity(Y[i].reshape(-1), Y_pred_transformed[i].reshape(-1)))
        jaccards2.append(utils.jaccard(Y[i],Y_pred_transformed[i]))
   
    table = {"Segmentation":[f'{model_name}',f'{model_name} + Refinement'],
            "Dice": [f"{np.round(100*np.mean(dices),2)} ± {np.round(100*np.std(dices),2)}", f"{np.round(100*np.mean(dices2),2)} ± {np.round(100*np.std(dices2),2)}"],
            "Jaccard (IoU)": [f"{np.round(100*np.mean(jaccards),2)} ± {np.round(100*np.std(jaccards),2)}", f"{np.round(100*np.mean(jaccards2),2)} ± {np.round(100*np.std(jaccards2),2)}"],
            "Specificity": [f"{np.round(100*np.mean(specs),2)} ± {np.round(100*np.std(specs),2)}", f"{np.round(100*np.mean(specs2),2)} ± {np.round(100*np.std(specs2),2)}"],
            "Sensitivity": [f"{np.round(100*np.mean(sens),2)} ± {np.round(100*np.std(sens),2)}", f"{np.round(100*np.mean(sens2),2)} ± {np.round(100*np.std(sens2),2)}"],
            "Precision": [f"{np.round(100*np.mean(precs),2)} ± {np.round(100*np.std(precs),2)}", f"{np.round(100*np.mean(precs2),2)} ± {np.round(100*np.std(precs2),2)}"]}
    

    results_name = os.path.join(results_path, f"{model_name}_{date.today()}.txt")

    with open(results_name, 'w') as f:
        print(f"Results:\n", file = f)
        print(tabulate(table, headers="keys", tablefmt='fancy_grid'),file = f)
        print("\nLatex format: \n", file = f)
        print(tabulate(table, headers="keys", tablefmt='latex'),file = f)
        print("\nMarkdown: \n", file = f)
        print(tabulate(table, headers="keys", tablefmt='github'),file = f)
        
    print(tabulate(table, headers="keys", tablefmt='fancy_grid'))

if __name__ == '__main__':
    app.run(main)