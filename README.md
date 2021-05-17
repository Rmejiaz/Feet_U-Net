# Feet U-Net

Semantic segmentation using a U-Net network architecture and temperature reading from termographic images of feet.

## Usage

### Install the requirements

```shell
pip install -r requirements.txt
```
### Download the dataset

It is possible to download the [feet dataset](https://drive.google.com/drive/folders/11a8eyrhjsk6Mh80bxv4D49j6s8khECs_?usp=sharing) in google colab, kaggle, or a linux machine running the following script:

```shell
python U-Net/download_dataset.py --path=DOWNLOAD PATH
```
The download path is relative to the current directory, if left unspecified it will download the dataset to the current working direcory.

### Train 

### Predict

### Evaluate

### Working example in google colab

https://colab.research.google.com/drive/1ODqQGpF4-0Cf6RPxyjy4_AqOZ1WBRb-7#scrollTo=ai-YYmBNIYWp


## TO DO

### [Segmentation](./U-Net):

- U-Net implementation to segment feet and background.
- U-Net implementation to segment the differentes parts of the foot.
- Create a toy dataset to compare the performance.
- Data augmentation
- Re-label the dataset (optional)

### [GUI](./GUI):
