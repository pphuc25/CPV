import random
import cv2
import numpy as np

def add_salt_and_pepper_noise(img):
    # Getting the dimensions of the image
    row, col = img.shape[:2]

    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img

def median_filter(image):
    image = add_salt_and_pepper_noise(image)
    cv2.imshow('Salt and pepper', image)
    cv2.waitKey(0)

    kernel_half = 3 // 2
    median_filter_image = np.zeros_like(image)
    for channel in range(image.shape[2]):
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                row_start = max(0, row-kernel_half)
                row_end = min(image.shape[0]-1, row+kernel_half)
                col_start = max(0, col-kernel_half)
                col_end = min(image.shape[1]-1, col-kernel_half)
                window = image[row_start:row_end+1, col_start:col_end+1, channel]
                median_filter_image[row][col][channel] = np.median(window)
    return median_filter_image

if __name__ == "__main__":
    path_image = '/home/pphuc/Coding/Project/FPTU/CPV/CPV/images/Anh_nude_1.jpg'
    origin_image = cv2.imread(path_image)
    image = origin_image.copy()
    
    choice = int(input('Type the choice: '))
    if choice == 1:
        result = median_filter(image)

    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
