import numpy as np
import cv2
import random

def MeanFilter(image):
    filter_size=9
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # create an empty array with same size as input image
    output = np.zeros(image.shape, np.uint8)
    # creat an empty variable
    result = 0
    for j in range(1, image.shape[0]-1):
        for i in range(1, image.shape[1]-1):
            for y in range(-1, 2):
                for x in range(-1, 2):
                    result = result + image[j+y, i+x]
            output[j][i] = int(result / filter_size)
            result = 0
    return output