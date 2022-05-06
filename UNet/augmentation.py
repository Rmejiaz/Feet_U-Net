import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copy2, move
from absl import app, flags, logging
from absl.flags import FLAGS


flags.DEFINE_string('img_path', './Dataset_3/Train/JPEGImages', 'path for input images')
flags.DEFINE_string('masks_path', './Dataset_3/Train/SegmentationClass', 'path for label images')
flags.DEFINE_string('augmented_path', None, 'path for augmented dataset')
flags.DEFINE_integer('n_images',192,'number of images to generate')
flags.DEFINE_boolean('featurewise_center', False, 'featurewise_center' )
flags.DEFINE_boolean('featurewise_std_normalization', False, 'featurewise_std_normalization' )
flags.DEFINE_integer('rotation_range',30,'Rotation Range')
flags.DEFINE_float('width_shift_range',0.2, 'width_shift_range')
flags.DEFINE_float('height_shift_range',0.2, 'height_shift_range')
flags.DEFINE_float('zoom_range',0.2, 'zoom_range')
flags.DEFINE_float('shear_range',0.2, 'shear_range')
flags.DEFINE_boolean('horizontal_flip', True, 'horizontal_flip' )




def main(argv_):

    # Initialize variables

    ImgDir = FLAGS.img_path
    MasksDir = FLAGS.masks_path
    results_path = FLAGS.augmented_path
    n_images = FLAGS.n_images

    if not FLAGS.augmented_path:
        try:
            os.mkdir('./AugmentedDataset')
        except:
            pass
        results_path = './AugmentedDataset'
        

    try:
        os.mkdir(results_path+'/JPEGImages')
        os.mkdir(results_path+'/SegmentationClass',)
    except:
       pass
  
    
    f=1
    if not(len(os.listdir(ImgDir)) == 1 and os.listdir(ImgDir)[0].find('.') == -1):
        while f:
            try:
                os.mkdir(ImgDir+'/../JPEGImages'+str(f))
                j=f
                f=0
            except:
                f+=1
        ImgDir_2 = ImgDir[:ImgDir.rfind('/')]+'/JPEGImages'+str(j)
        move(ImgDir,ImgDir+'/../JPEGImages'+str(j))
    else:
        ImgDir_2 = ImgDir  

    f=1
    if not(len(os.listdir(MasksDir)) == 1 and os.listdir(MasksDir)[0].find('.') == -1):
        while f:
            try:
                os.mkdir(MasksDir+'/../SegmentationClass'+str(f))
                j=f
                f=0
            except:
                f+=1
        MasksDir_2 = MasksDir[:MasksDir.rfind('/')]+'/SegmentationClass'+str(j)
        move(MasksDir,MasksDir+'/../SegmentationClass'+str(j))
    else:
        MasksDir_2 = MasksDir

    
    image_size = plt.imread(MasksDir_2+'/'+os.listdir(MasksDir_2)[0]+'/'+next(os.walk(MasksDir_2+'/'+os.listdir(MasksDir_2)[0]))[2][0]).shape[:2]

    
    # Data generator:
    datagen = ImageDataGenerator(
        featurewise_center= FLAGS.featurewise_center,
        featurewise_std_normalization= FLAGS.featurewise_std_normalization,
        rotation_range= FLAGS.rotation_range,
        width_shift_range= FLAGS.width_shift_range,
        height_shift_range= FLAGS.height_shift_range,
        zoom_range = FLAGS.zoom_range,
        shear_range = FLAGS.shear_range,
        horizontal_flip = FLAGS.horizontal_flip
        )

    seed = 42
  
    image_generator = datagen.flow_from_directory(directory=ImgDir_2,target_size=image_size,save_to_dir=results_path+'/JPEGImages',
                                                  class_mode=None,save_format='jpg',seed = seed)

    masks_generator = datagen.flow_from_directory(directory=MasksDir_2,target_size=image_size,save_to_dir=results_path+'/SegmentationClass',
                                                  class_mode=None,save_format='png',seed = seed)


    n_iter = int(n_images/32)
    if n_images % 32:
        n_iter += 1
    
    print(f"Generating {n_iter*32} images")
    for i in range(n_iter):
        image_generator.next()
        masks_generator.next()
        print(i+1,'/',n_iter, 'done')


    # Reorder the directories
    if not f:
        move(MasksDir_2+'/'+os.listdir(MasksDir_2)[0], MasksDir_2+'/..')
        move(ImgDir_2+'/'+os.listdir(ImgDir_2)[0],ImgDir_2+'/..')
        os.rmdir(ImgDir_2)
        os.rmdir(MasksDir_2)

    # Copy original images to the AugmentedData directory

    origin_imgs = ImgDir
    origin_masks = MasksDir
    dst_imgs = os.path.join(results_path,'JPEGImages')
    dst_masks = os.path.join(results_path, 'SegmentationClass')

    for file in os.listdir(origin_imgs):
        copy2(os.path.join(origin_imgs,file), os.path.join(dst_imgs, file))
    
    for file in os.listdir(origin_masks):
        copy2(os.path.join(origin_masks, file), os.path.join(dst_masks,file))

    print(f"Total ammount of images: {len(os.listdir(dst_imgs))}")  

if __name__=="__main__":
    app.run(main)

    



    
        

