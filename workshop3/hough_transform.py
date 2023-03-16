import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
# from scipy.misc import imread,imresize,imsave
from skimage.color import rgb2gray

def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    # Initialize output image
    lines = []

    # Get image dimensions
    height, width = image.shape[:2]

    # Compute maximum rho value
    max_rho = math.ceil(math.sqrt(height ** 2 + width ** 2))

    # Compute theta values
    thetas = np.arange(-math.pi / 2, math.pi / 2, theta)

    # Compute sin and cos of theta values
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    # Initialize accumulator array
    accumulator = np.zeros((len(thetas), max_rho * 2))

    # Find edges in the image
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Loop through all edge pixels
    for y in range(height):
        for x in range(width):
            if edges[y, x] != 0:
                # Compute rho for each theta value
                for i in range(len(thetas)):
                    rho_val = int(x * cos_thetas[i] + y * sin_thetas[i]) + max_rho
                    accumulator[i, rho_val] += 1

    # Find lines using accumulator array
    for i in range(len(thetas)):
        for j in range(max_rho * 2):
            if accumulator[i, j] > threshold:
                # Compute endpoint coordinates for each line
                rho_val = j - max_rho
                x1 = int(rho_val * cos_thetas[i] - height * sin_thetas[i])
                y1 = int(rho_val * sin_thetas[i] + height * cos_thetas[i])
                x2 = int(rho_val * cos_thetas[i] + width * sin_thetas[i])
                y2 = int(rho_val * sin_thetas[i] - width * cos_thetas[i])

                # Add line to output image
                lines.append([[x1, y1, x2, y2]])

    return lines


def draw_lines(image, lines, color=(0, 0, 255), thickness=2):
    # Copy input image to avoid modifying original
    img_copy = np.copy(image)

    # Loop through all lines
    for line in lines:
        # Get endpoints of line
        x1, y1, x2, y2 = line[0]

        # Draw line on image
        cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)

    return img_copy

def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
    """
    Hough transform for lines

    Input:
    img - 2D binary image with nonzeros representing edges
    angle_step - Spacing between angles to use every n-th angle
                 between -90 and 90 degrees. Default step is 1.
    lines_are_white - boolean indicating whether lines to be detected are white
    value_threshold - Pixel values above or below the value_threshold are edges

    Returns:
    accumulator - 2D array of the hough transform accumulator
    theta - array of angles used in computation, in radians.
    rhos - array of rho values. Max size is 2 times the diagonal
           distance of the input image.
    """
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
    lines = []
    line_width = 80

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    # (row, col) indexes to edges
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    y_idxs, x_idxs = np.nonzero(are_edges)

    # print(len(x_idxs), num_thetas)

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1
            if accumulator[rho, t_idx] > line_width:
                lines.append((rho, t_idx))

    return accumulator, thetas, rhos, lines


def draw_hough_lines(img, lines):
    """
    Draw the detected hough lines on the original image

    Input:
    img - original image
    lines - list of detected lines as (rho, theta) tuples
    """
    for line in lines:
        print(line)
        rho, theta = line
        print(rho, theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

def hough(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # accumulator, thetas, rhos, lines = hough_line(img)
    # # print('lines:', lines)
    # draw_hough_lines(img, lines[:100])
    # cv2.imshow('Hough', img)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = hough_lines(edges, 1, np.pi / 180, 150, 14, 42)
    line_img = draw_lines(img, lines)
    cv2.imshow('result', line_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


