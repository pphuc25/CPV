import math
import cv2
import numpy as np

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

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_value, high_value)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    result = cv2.merge(out_channels)

    # # Display the original image
    # cv2.imshow('Origin', img)
    #
    # # Display the result image
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return result

# Define a function for face preprocessing
def preprocess_img(img):

    #Color balance
    img = simplest_color_balance(img, 10)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform histogram equalization to improve contrast
    eq_gray = cv2.equalizeHist(gray)

    # Apply a Gaussian blur to smooth the image and reduce noise
    blur = cv2.GaussianBlur(eq_gray, (5, 5), 0)

    return blur
