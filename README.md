# Project Title

Image-Super-Resolution-using-an-Efficient-Sub-Pixel-CNN

## Motivation

We are motivated by a scene in a crime movie in which a blurred picture of a criminal is turned into a clear picture

## Description

Convert a Low Resolution image into a High Resolution image using Sub-Pixel-CNN

A detailed description of the folders and files is in DETAIL.md

## Features

When training CNN neural networks, RGB channels of images were independently trained.
Therefore, each image channel passes through the neural network.

3_colors_network has 3 networks that are independently responsible for each RGB. Therefore, when extracting an HR image, the channel of the LR image enters the corresponding network and finally merges RGB to obtain the HR image.

## Screenshots
![image](https://user-images.githubusercontent.com/107605573/174060023-96efd4ed-4e26-4059-8e74-154fb996a490.png)

## Getting Started

* make data folder
* download the data from https://www.kaggle.com/datasets/jessicali9530/celeba-dataset into data folder

 Inside the data folder, there should be an img_align_celeba folder and a list_val_partition.csv file

### Dependencies

* openCV
* Numpy
* Scipy
* Pytorch
* matplotlib

### Executing program

* How to run the program
```
python *_train.py
```
```
python *_test.py
```
```
python *_print.py
```

## Authors

Hyuk Jin Kim 
hyukjin.kim.dev@gmail.com

## License

CelebFaces Attributes (CelebA) Dataset
https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

ESPCN paper
https://arxiv.org/abs/1609.05158



