import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tensorflow.keras import backend as K
from sklearn.metrics import jaccard_score
from skimage.morphology import erosion, dilation
from sklearn.metrics import confusion_matrix
import cv2

def download_dataset():
    """
    Function for downloading the feet dataset inside a google colab or kaggle notebook. It can also work in jupyter in linux.
    It just downloads the complete dataset in the current directory
    """
    
    ID = "1-3_oB5iSF-c_V65-uSdUlo024NzlgSYZ"
    script1 = f"""
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='{ID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="{ID} -O Data.zip && rm -rf /tmp/cookies.txt
    """
    script2 = """unzip Data.zip"""

    os.system(script1)
    os.system(script2)


def get_polygons(annotation):
    """Get polygons coordinates of all the labels from a xml file

    Parameters
    ----------
    annotation : str
        relative path to the xml file

    Returns
    -------
    dict
        dict containing all the polygons coordinantes for the each label
    """
    print(f"Loadding: {annotation}")
    tree = ET.parse(annotation)
    root = tree.getroot()
    polygons = {}
    for obj in root.findall('object'):
        name = obj.find('name').text
        id_ = obj.find('id').text
        polygon = []
        for pt in obj.find('polygon').findall('pt'):
            polygon.append([pt.find('x').text, pt.find('y').text])
        if name in polygons:
            x_ref= int(polygons[name]['left'][0][0])
            x = int(polygon[0][0])
            if x > x_ref:
                polygons[name]['right'] = polygons[name]['left']
                id_ = 'left'
            else:
                id_ = 'right'
        else:
            polygons[name] = {}
            id_ = 'left'
        polygons[name][id_] = polygon
    for i in list(polygons.keys()):
        if not('right' in polygons[i]):
            print(i,' only has one polygon: ',polygons[i]['left'])
            y = input('Do you wish to label it as \'right\'? (leave empy if No): ')
            if (y):
                polygons[i]['right'] = polygons[i]['left']
                polygons[i].pop('left')
    return polygons


def categorical2mask(X, labels):
    """Convert a mask to a rgb image according to the labels dict
    Parameters
    -----------
    X: tf.tensor or np.ndarray
        categorical representation of a mask
    labels: dict
        dict containing the labelmap that describes the rgb values of each label
    """
    X_shape = X.shape[0:2]
    if type(X_shape) == tuple:
        X_shape = list(X_shape)
    Y = np.zeros(X_shape + [3], dtype="uint8")
    for i, key in enumerate(labels):
        print(X.shape,Y.shape)
        Y[...,0] = np.where(X==i, labels[key][0], Y[...,0])
        Y[...,1] = np.where(X==i, labels[key][1], Y[...,1])
        Y[...,2] = np.where(X==i, labels[key][2], Y[...,2])
    return Y


def mask2categorical(Mask: tf.Tensor, labels: dict) -> tf.Tensor:
    """Pass a certain rgb mask (3-channels) to an image of ordinal classes"""
    assert type(labels) == dict, "labels variable should be a dictionary"

    X = Mask

    if X.dtype == "float32":
        X = tf.cast(X*255, dtype="uint8")

    Y = tf.zeros(X.shape[0:2] , dtype="float32")
    for i, key in enumerate(labels):
        Y = tf.where(np.all(X == labels[key], axis=-1), i, Y)
    Y = tf.cast(Y, dtype="uint8")
    return Y

def parse_labelfile(path):
    """Return a dict with the corresponding rgb mask values of the labels
        Example:
        >>> labels = parse_labelfile("file/path")
        >>> print(labels) 
        >>> {"label1": (r1, g1, b1), "label2": (r2, g2, b2)} 
    """
    with open(path, "r") as FILE:
        lines = FILE.readlines()


    labels = {x.split(":")[0]: x.split(":")[1] for x in lines[1:]}

    for key in labels:
        labels[key] = np.array(labels[key].split(",")).astype("uint8")

    return labels


def display(display_list):
    plt.figure(figsize=(20, 15))

    if (display_list[0].shape[-1] == 3):
        mask = np.zeros((display_list[0].shape[0], display_list[0].shape[1], display_list[0].shape[2]))
        mask[:,:,0] = mask[:,:,1] = mask[:,:,2] = display_list[1][:,:]
        display_list.append(display_list[0]*mask)
    else:
        display_list.append(display_list[0][:,:,:]*display_list[1])
    title = ['Input Image', 'Predicted Mask','Segmented Image']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        # plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        if len(display_list[i].shape) == 2:
            plt.imshow((display_list[i]),cmap='gray')
        else:
            plt.imshow((display_list[i]))
        plt.axis('off')
    plt.show()


def load_data(path,size, scale = True):
    """
    Load a set of images to memory
    Parameters
    ------------
    path : str
        path to the directory of the images
    size : int
        size to resize the images
    Returns
    --------
    np.ndarray
        array of dimensions (n_images, size, size, n_channels)
    """
    images = os.listdir(path)
    images.sort()

    X = []
    for i, img in enumerate(images):
        photo = plt.imread(os.path.join(path,img))
        if size:
            photo = tf.image.resize(photo, (size, size))
        X.append(photo)
        
    X = np.array(X)
    if scale:
        X = X/X.max() 
    return X  


def display_mask(img,mask):
    plt.figure(figsize=(15,15))
    img = img[:,:,0]
    image = img+mask
    plt.imshow(image,cmap='gray')
    plt.title("Predicted Mask")
    plt.show()    
    
def DiceSimilarity(Pred, Set, label=1): #Dice similarity is defined as 2*|X ∩ Y|/(|X|+|Y|)
    D = np.sum(Pred[Set==label]==label)*2.0 / (np.sum(Pred[Pred==label]==label) + np.sum(Set[Set==label]==label))
    if np.isnan(D):
      D = 1
    return D

def DiceImages(PathPred, PathSet):
    #Predicted Mask
    Pred = plt.imread(PathPred)
    
    #Float to uint
    if Pred.dtype == np.float32: 
        Pred = (Pred*255).astype(np.uint8)
        
    #Image to Tensor
    Pred = tf.convert_to_tensor(Pred, dtype=tf.uint8) 
    
    #Real Mask
    Set = plt.imread(PathSet) 
    
    if Set.dtype == np.float32:
        Set = (Set*255).astype(np.uint8)
    
    Set = tf.convert_to_tensor(Set, dtype=tf.uint8)
    
    #Mask to categorical
    # Pred = mask2categorical(Pred).numpy() 
    # Set = mask2categorical(Set).numpy()
    
    Dice = {}
    for i in np.unique(np.append(np.unique(Pred), np.unique(Set))): #i is the labels in Pred and Set. 
        Dice[i] = DiceSimilarity(Pred, Set, i)                      #Since it is possible for one to have labels the other doesn't we need to append the sets
    
    return Dice
        
    
def jaccard(y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return jaccard_score(y_true,y_pred)
    
def dice_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection)/(union), axis=0)
    return dice

def iou_coef(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection) / (union), axis=0)
    return iou

def n_opening(Img, n):
    for i in range(n):
        Img = erosion(Img)
    for i in range(n):
        Img = dilation(Img)
    return Img

def n_closing(Img, n):
    for i in range(n):
        Img = dilation(Img)
    for i in range(n):
        Img = erosion(Img)
    return Img


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = 100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def remove_small_objects(img, min_size=1200):
    """Remove all the objects that are smaller than a defined threshold

    Parameters
    ----------
    img : np.ndarray
        Input image to clean
    min_size : int, optional
        Threshold to be used to remove all smaller objects, by default 1200

    Returns
    -------
    np.ndarray
        Cleaned image
    """
    img2 = np.copy(img)
    img2 = np.uint8(img2)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img2, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # your answer image
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] < min_size:
            img2[output == i + 1] = 0

    return img2 
