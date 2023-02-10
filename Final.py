import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math

def apply_mask(matrix, mask, fill_value):
    #Apply fill value for element correspoding False in mask
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    #Fill low value for elements in matrix smaller than low value
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    #Fill high value for elements in matrix smaller than high value
    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_color_balance(img, percent):
    #Apply condition for code to continue
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200

    #Split image into three channels RGB
    channels = cv2.split(img)

    #Init results channels
    out_channels =  []

    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = height * width
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_value = flat[math.floor(n_cols * half_percent)]
        high_value = flat[math.ceil(n_cols * (1 - half_percent))]
        print(low_value, high_value)

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_value, high_value)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    result = cv2.merge(out_channels)

    # Display the original image
    cv2.imshow('Origin', image)

    # Display the result image
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def histogram_equalization(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(image.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Using calcHist to calculate the distribution
    hist1 = cv2.calcHist([image], [0], None, [256], [0, 256])
    # Using equalizeHist to increase globalcontrast
    result = cv2.equalizeHist(image)
    hist2 = cv2.calcHist([result], [0], None, [256], [0, 256])

    # Display the before and after histogram
    plt.subplot(222), plt.plot(hist1)
    plt.subplot(224), plt.plot(hist2)
    plt.show()

    # Display the original image
    cv2.imshow('Origin', image)

    # Display the result image
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    # Add salt and pepper noise
    image = add_salt_and_pepper_noise(image)
    # Perform median filtering to remove salt and pepper noise
    result = cv2.medianBlur(image, 3)
    # Display the original image
    cv2.imshow('Origin', image)

    # Display the result image
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mean_filter(image):
    # Add salt and pepper noise
    image = add_salt_and_pepper_noise(image)
    # Perform mean filtering to remove salt and pepper noise
    result = cv2.blur(image, (3, 3))
    # Display the original image
    cv2.imshow('Origin', image)

    # Display the result image
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gaussian_smoothing(image):
    # Perform Gaussian smoothing to perform image smoothing
    result = cv2.GaussianBlur(image, (3, 3), 0)

    # Display the original image
    cv2.imshow('Origin', image)

    # Display the result image
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image = cv2.imread('chess.png')
    result = image.copy()

    while True:
        print('Menu:')
        print('1. Color balance')
        print('2. Histogram equalization')
        print('3. Median filter')
        print('4. Mean filter')
        print('5. Gaussian smoothing')
        print('6. Back to origin image')
        print('7. Quit')

        choice = int(input('Enter your choice: '))

        if choice == 1:
            # Perform color balance
            percent = float(input('Enter percent value: '))
            simplest_color_balance(result, percent)

        elif choice == 2:
            # Perform histogram equalization
            histogram_equalization(result)

        elif choice == 3:
            # Perform median filtering
            median_filter(result)

        elif choice == 4:
            # Perform mean filtering
            mean_filter(result)

        elif choice == 5:
            # Perform Gaussian smoothing
            gaussian_smoothing(result)

        elif choice == 6:
            # Back to origin image
            result = image.copy()

        elif choice == 7:
            break

        else:
            print('Invalid choice. Try again.')



