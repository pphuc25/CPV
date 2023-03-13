import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.signal import convolve2d


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
    cv2.imshow('Origin', img)

    # Display the result image
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def imhist(im):
    # calculates normalized histogram of an image
    m, n = im.shape
    h = [0.0] * 256
    for i in range(m):
        for j in range(n):
            h[im[i, j]]+=1
    return np.array(h)/(m*n)

def histeq(im):
    #calculate Histogram
    h = imhist(im)
    cdf = np.array(np.cumsum(h)) #cumulative distribution function
    sk = np.uint8(255 * cdf) #finding transfer function values
    s1, s2 = im.shape
    Y = np.zeros_like(im)
    # applying transfered values for each pixels
    for i in range(0, s1):
        for j in range(0, s2):
            Y[i, j] = sk[im[i, j]]
    H = imhist(Y)
    #return transformed image, original and new istogram,
    # and transform function
    return Y , h, H, sk

def histogram_equalization(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_img, h, new_h, sk = histeq(img)

    #Show image
    cv2.imshow('Origin', img)

    cv2.imshow('Result', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Show hist
    fig = plt.figure()
    fig.add_subplot(221)
    plt.plot(h)
    plt.title('Original histogram') # original histogram

    fig.add_subplot(222)
    plt.plot(new_h)
    plt.title('New histogram') #hist of equalized image

    plt.show()

def add_salt_and_pepper_noise(raw_img):
    img = raw_img.copy()
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

def rgb_to_gray(image):
    """
    Converts an RGB image to grayscale.
    """
    # Compute the weighted sum of the RGB channels to obtain the grayscale image
    gray = np.dot(image, [0.2989, 0.5870, 0.1140])

    return gray

def gaussian_kernel(kernel_size, sigma):
    x, y = np.meshgrid(np.linspace(-1,1,kernel_size), np.linspace(-1,1,kernel_size))
    d = np.sqrt(x*x+y*y)
    return np.exp(-((d)**2/(2.0*sigma**2)))

def gaussian_blur(image, kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma)
    image = rgb_to_gray(image)
    padded = np.pad(image, [(kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)], mode='constant')
    result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = (kernel * padded[i:i+kernel_size, j:j+kernel_size]).sum()
    result = result.astype(np.uint8)
    return result

def median_filtering(image, kernel_size):
    #Turn image into gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_half = kernel_size // 2
    padded_image = np.pad(image, kernel_half, mode='edge')
    median_filter_image = np.zeros(image.shape)

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            row_start = max(0, row-kernel_half)
            row_end = min(image.shape[0]-1, row+kernel_half)
            col_start = max(0, col-kernel_half)
            col_end = min(image.shape[1]-1, col+kernel_half)
            window = padded_image[row_start:row_end+1, col_start:col_end+1]
            median_filter_image[row, col] = np.median(window)
    median_filter_image = median_filter_image.astype(np.uint8)

    return median_filter_image

def median_filter(raw_image):
    image = raw_image
    # Add salt and pepper noise
    image = add_salt_and_pepper_noise(image)
    # Perform median filtering to remove salt and pepper noise
    result = median_filtering(image, 3)
    # Display the original image
    cv2.imshow('Origin', image)

    # Display the result image
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

def mean_filter(raw_image):
    image = raw_image
    # Add salt and pepper noise
    image = add_salt_and_pepper_noise(image)
    # Perform mean filtering to remove salt and pepper noise
    result = MeanFilter(image)
    # Display the original image
    cv2.imshow('Origin', image)

    # Display the result image
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gaussian_smoothing(raw_image):
    image = raw_image
    # Perform Gaussian smoothing to perform image smoothing
    result = gaussian_blur(image, 3, 0.000001)

    # Display the original image
    cv2.imshow('Origin', image)

    # Display the result image
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ori_image = cv2.imread('test_CPV.jpeg')

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
            simplest_color_balance(ori_image, percent)

        elif choice == 2:
            # Perform histogram equalization
            histogram_equalization(ori_image)

        elif choice == 3:
            # Perform median filtering
            median_filter(ori_image)

        elif choice == 4:
            # Perform mean filtering
            mean_filter(ori_image)

        elif choice == 5:
            # Perform Gaussian smoothing
            gaussian_smoothing(ori_image)

        elif choice == 6:
            # Back to origin image
            result = ori_image.copy()

        elif choice == 7:
            break

        else:
            print('Invalid choice. Try again.')



