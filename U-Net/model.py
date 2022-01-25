import tensorflow as tf 

from models.MobileNetV2 import get_model as get_MobileNetV2
from models.FCN import get_model as get_FCN
from models.SegNet import get_model as get_SegNet 
from models.UNET import get_model as get_UNET
from models.VGG16 import get_model as get_VGG16

MODELS = {'mobilenetv2': get_MobileNetV2,
          'fcn':get_FCN,
          'segnet':get_SegNet,
          'unet':get_UNET,
          'vgg16':get_VGG16}

def print_avalible_models():
    for model in MODELS.keys():
        print(f'Model: {model}')


def get_model(model='mobilenetv2', **kwargs):
    model = model.lower()
    try: 
        model_keras = MODELS[model](**kwargs)
        return model_keras
    except KeyError: 
        print(f'Model {model} is not avalaible')
        print(f"posible models {', '.join(MODELS.keys())}")
        exit()
    
if __name__ == '__main__':
    print_avalible_models()
    model = get_model(output_channels=2)
    model.summary()
    tf.keras.utils.plot_model(model,to_file='model.png',show_shapes=False,show_layer_names=False)