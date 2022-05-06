from tensorflow.keras.models import load_model as ld 

import utils
from models.SegNet import MaxPoolingWithArgmax2D, MaxUnpooling2D

def load_model(model_path):
    model = ld(model_path, custom_objects = {'dice_coef':utils.dice_coef, 
                                                                    'iou_coef':utils.iou_coef,
                                                                     'MaxPoolingWithArgmax2D':MaxPoolingWithArgmax2D,
                                                                     'MaxUnpooling2D':MaxUnpooling2D})
    return model 

if __name__ == "__main__":
    model = load_model('model.h5')
    model.summary()