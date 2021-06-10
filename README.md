# Feet U-Net

Semantic segmentation using a U-Net network architecture and temperature reading from termographic images of feet.

![Example](./U-Net/results/best_prediction.png)

## Google Colab demo: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ODqQGpF4-0Cf6RPxyjy4_AqOZ1WBRb-7?usp=sharing)

## Usage

### Install the requirements

```bash
$ pip install -r requirements.txt
```
### Download the dataset

It is possible to download the [feet dataset](https://drive.google.com/drive/folders/16nbmFG2MucF6UlY05rPajx4G81vttg4O?usp=sharing) in google colab, kaggle, or a linux machine executing the script `download_dataset.py` as follows:

```bash
$ python U-Net/download_dataset.py --path=DOWNLOAD PATH
```
The download path is relative to the current directory, if left unspecified it will download the dataset to the current working direcory.

### (Optional) Data Augmentation

It is possible to do data augmentation to the images and the masks using the script `augmentation.py`. It applies some random transformations to the input images such as rotations, crops and shiftings.

```bash
$ python U-Net/augmentation.py --img_path=PATH TO IMAGES DIRECTORY --masks_path=PATH TO MASKS DIRECTORY --augmented_path=PATH TO SAVE THE NEW DATASET --labels=PATH OF THE LABELMAP n_images=NUMBER OF IMAGES TO GENERATE 
```

It will automatically create a new dataset in the specified path. If left unspecified, it creates the new dataset in the current working directory. By default, `img_path = ./Dataset_Unificado/Train/Processed_Images` , `masks_path = ./Dataset_Unificado/Train/BinaryMasks` , `labels = ./Dataset_Unificado/binary_labelmap.txt` and `n_images = 192`  

(It is advided to be used with caution, as results do not necessarily improve when a large number of images are generated)

### Train 

In order to train the model, run the following command:

```bash
$ python U-Net/train.py --imgs_path=PATH TO IMAGES DIRECTORY --masks_path=PATH TO MASKS DIRECTORY --val_imgs=PATH TO THE IMAGES FOR VALIDATION --val_masks=PATH TO THE MASKS FOR VALIDATION --val_split=VALIDATION SPLIT --weights=PATH TO SAVE THE TRAINED WEIGHTS --buffer_size=BUFFER_SIZE --batch_size=BATCH SIZE --epochs=NUMBER OF EPOCHOS --save_freq=SAVE FREQUENCY FOR THE CHECKPOINTS
```

If `val_split` is set to `0`, `val_imgs` and `val_masks` will be used for validation. Otherwise, it will split the training dataset according to `val_split` and use it for validation.

The defaults are: `imgs_path=./Dataset_CVAT2/JPEGImages/Train` , `masks_path=./Dataset_CVAT2/SegmentationClass/Train` `val_imgs = ./Dataset_CVAT2/JPEGImages/Test` , `val_masks = ./Dataset_CVAT2/SegmentationClass/Test` , `weights = ./weights/` , `buffer_size = 100` , `batch_size = 5` , `epochs = 10` , `save_freq = 5` 

The weights will be saved to `weights` and the training history will also be saved in `./results/history0`

### Predict

Do a prediction using trained weights.

```bash
$ python predict.py --image_path=IMAGE PATH --mask_path=MASK PATH --labels=LABELS PATH --show_results=True --weights=WEIGHTS PATH
```
The defaults for the different arguments are the same ones as used before

### Evaluate

Make a prediction for a test set and measure dice and jaccard scores

```bash
$ python evaluate.py --weights=WEIGHTS PATH --test_images=IMAGES DIRECTORY PATH --test_masks=MASKS DIRECTORY PATH --results=PATH TO SAVE THE RESULTS
```

By default, `test_images` and `test_masks` are set to `./Dataset_CVAT2/JPEGImages/Test` and `./Dataset_CVAT2/SegmentationClass/Test`. `weights` is set to `./weights/cp-0100.ckpt` and restuls to `./restults/`.

After running the script, a figure containing the Dice and Jaccard scores boxplot for the test images is generated and saved in `results`, as well as a subdirectory containing the predictions of the test set. 


### [Download a pretrained model](https://drive.google.com/file/d/1S-pUZZONC3fqMSvXfBg_7tGdxtkLIs-F/view?usp=sharing)

It is also possible to download a pretrained model, which can be later used to make predictions as follows:

```python
model = tf.keras.models.load_model('Model1.h5')

Y_pred = model.predict(X)
```

## TO DO

### [Segmentation](./U-Net):

- ~~U-Net implementation to segment feet and background.~~
- U-Net implementation to segment the differentes parts of the foot.
- Create a toy dataset to compare the performance.
- ~~Data augmentation~~
- ~~Re-label the dataset (optional)~~
- ~~Improve Dice to at least 80%.~~
- ~~Closing and Opening~~
