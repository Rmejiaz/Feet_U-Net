import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tensorflow.keras import backend as K
from sklearn.metrics import jaccard_score

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
    """
    Get polygons coordinates of all the labels from a xml file. It returns a dictionary with the polygons for each label for each foot.
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

def load_tfrecord_dataset(dataset_path, size):
    """Load and parse a dataset in tfrecord format. 
    Parameters
    -----------
    dataset_path : str 
        path of the tfrecord dataset
    size : int
        size of the images in the dataset
    
    Returns
    ----------
    tf.data.Dataset
        Dataset with resized and scaled (min-max) images.
    """
    raw_dataset = tf.data.TFRecordDataset([dataset_path])
    return raw_dataset.map(lambda x: parse_dataset(x, size))

IMAGE_FEATURE_MAP = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string)
        }

def parse_dataset(tfrecord, size):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP) 
    X_train = tf.image.decode_jpeg(x['image'], channels=3)
    Y_train = tf.image.decode_png(x['mask'])

    X_train = tf.image.resize(X_train, (size, size))
    Y_train = tf.image.resize(Y_train, (size, size))
    return X_train/255, Y_train

def split_dataset(dataset: tf.data.Dataset, validation_data_fraction: float):
    """
    Splits a dataset of type tf.data.Dataset into a training and validation dataset using given ratio. Fractions are
    rounded up to two decimal places.
    @param dataset: the input dataset to split.
    @param validation_data_fraction: the fraction of the validation data as a float between 0 and 1.
    @return: a tuple of two tf.data.Datasets as (training, validation)
    """

    validation_data_percent = round(validation_data_fraction * 100)
    if not (0 <= validation_data_percent <= 100):
        raise ValueError("validation data fraction must be ∈ [0,1]")

    dataset = dataset.enumerate()
    train_dataset = dataset.filter(lambda f, data: f % 100 > validation_data_percent)
    validation_dataset = dataset.filter(lambda f, data: f % 100 <= validation_data_percent)

    # remove enumeration
    train_dataset = train_dataset.map(lambda f, data: data)
    validation_dataset = validation_dataset.map(lambda f, data: data)

    return train_dataset, validation_dataset

def display(display_list):
    plt.figure(figsize=(20, 15))

    display_list.append(display_list[0][:,:,0]+display_list[1])
    title = ['Input Image', 'Predicted Mask','Overlay']

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
    return np.sum(Pred[Set==label]==label)*2.0 / (np.sum(Pred[Pred==label]==label) + np.sum(Set[Set==label]==label))

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

