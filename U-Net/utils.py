import os
import numpy as np
import tensorflow as tf

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
    Y = np.zeros(X.shape[0:2] + [3], dtype="uint8")
    for i, key in enumerate(labels):
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

