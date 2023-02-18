import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from dataclasses import dataclass


class HarrisCornerDetection:
    """
    Harris Corner to detect corner in 
    """
    sobel_x = np.array((
        [-1, 0, -1],
        [-2, 0, 2],
        [-1, 0, 1]), dtype='int32'
    )

    sobel_y = np.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]), dtype='int32'
    )

    gauss = np.array((
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]), dtype='float64'
    )

    def __init__(self, image_path, threshold=0.6):
        self.threshold = threshold
        self.image = image_path
        self.run()

    def convolve(self, img, kernel):
        img_height = img.shape[0]
        img_width = img.shape[1]
        pad_height = kernel.shape[0] // 2
        pad_width = kernel.shape[1] // 2

        pad = ((pad_height, pad_height), (pad_width, pad_width))
        g = np.empty(img.shape, dtype=np.float64)
        img = np.pad(img, pad, mode='constant', constant_values=0)

        for i in np.arange(pad_height, img_height+pad_height):
            for j in np.arange(pad_width, img_width+pad_width):
                roi = img[i-pad_height:i+pad_height+1, j-pad_width:j+pad_width+1]
                g[i-pad_height, j-pad_width] = (roi*kernel).sum()
        
        if (g.dtype == np.float64):
            kernel = kernel / 255.0
            kernel = (kernel*255).astype(np.uint8)
        else:
            g = g + abs(np.amin(g))
            g = g / np.amax(g)
            g = (g*255.0)
        return g

    def harris(self):
        img_copy = self.image.copy()
        img1_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        dx = self.convolve(img1_gray, self.sobel_x)
        dy = self.convolve(img1_gray, self.sobel_y)
        dx2 = np.square(dx)
        dy2 = np.square(dy)
        dxdy = dx*dy
        g_dx2 = self.convolve(dx2, self.gauss)
        g_dy2 = self.convolve(dy2, self.gauss)
        g_dxdy = self.convolve(dxdy, self.gauss)

        harris = g_dx2*g_dy2 - np.square(g_dxdy) - 0.12*np.square(g_dx2 + g_dy2)
        cv2.normalize(harris, harris, 0, 1, cv2.NORM_MINMAX)

        loc = np.where(harris >= self.threshold)
        for pt in zip(*loc[::-1]):
            cv2.circle(img_copy, pt, 3, (0, 0, 255), -1)
        return img_copy, g_dx2, g_dy2, dx, dy, loc
    
    def run(self):
        corners, g_dx2, g_dy2, dx, dy, loc = self.harris()
        cv2.imshow('Origin', self.image)
        cv2.imshow('harris corner', corners)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

img_path = 'images/test_CPV.jpeg'
image = cv2.imread(img_path)
harris = HarrisCornerDetection(image, 0.85)

# plt.figure(figsize=(20, 20))
# plt.subplot(121), plt.imshow(img)
# plt.title("Raw Image"), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(corners)
# plt.title("Harris Corner Output"), plt.xticks([]), plt.yticks([])
