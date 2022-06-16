import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

def Gaussian(x, y, sd=1.0):
    a = 1 / (2 * np.pi * sd ** 2)
    b = -((x ** 2 + y ** 2) / (2 * sd ** 2))
    c = np.exp(b)
    return a * c

def gFilter(sd=1):
    F_size = 8 * sd + 1
    center = 4 * sd

    filter = np.zeros((F_size, F_size), dtype=float)
    filter[center][center] = Gaussian(0, 0, sd)

    for row in range(0, center):
        for col in range(0, center):
            if (row == 0) and (col == 0):
                continue
            filter[center - row][center - col] = filter[center + row][center + col] = Gaussian(row, col, sd)
            filter[center + row][center - col] = filter[center - row][center + col] = Gaussian(-row, col, sd)
    return filter

def resize(img):

    img_crop = cv2.resize(img, dsize=(176, 176))
    img_blur = cv2.filter2D(img_crop, -1, kernel=gFilter(1))  # 가우시안 blur 적용

    # 44*44*3 크기로 4배 축소
    out_shape = tuple([44, 44, 3])
    input_shape = img_blur.shape

    n = (np.array(input_shape) / np.array(out_shape))
    coord = [n[i] * (np.arange(d))
             for i, d in enumerate(out_shape)]
    coord = np.meshgrid(*coord, indexing='ij')
    img_resized = ndi.map_coordinates(img_blur, coord)

    return img_resized, img_crop

def preprocess(images):
    B = images.shape[0]

    a = np.zeros((B, 44, 44, 3))
    b = np.zeros((B, 176, 176, 3))
    for idx in range(B):
        img = images[idx]
        a[idx], b[idx] = resize(img)
    return a, b
    


