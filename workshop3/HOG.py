import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure


def hog_visualization(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define the cell size for computing the histogram of gradients
    cell_size = (8, 8)

    # Define the block size for normalizing the histograms
    block_size = (2, 2)

    # Define the number of orientation bins for the histogram
    n_bins = 9

    # Compute the gradients of the image in the x and y directions using the Sobel filter
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)

    # Compute the magnitude and direction of the gradients
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # Create a histogram for each cell
    height, width = gray.shape
    cell_height, cell_width = cell_size
    n_cells_y = int(height / cell_height)
    n_cells_x = int(width / cell_width)
    histograms = np.zeros((n_cells_y, n_cells_x, n_bins))

    for y in range(n_cells_y):
        for x in range(n_cells_x):
            cell_magnitude = magnitude[y * cell_height:(y + 1) * cell_height,
                                    x * cell_width:(x + 1) * cell_width]
            cell_angle = angle[y * cell_height:(y + 1) * cell_height,
                            x * cell_width:(x + 1) * cell_width]
            histograms[y, x, :] = np.histogram(cell_angle,
                                                bins=n_bins,
                                                range=(0, 180),
                                                weights=cell_magnitude)[0]

    # Normalize the histograms for each block
    block_height, block_width = block_size
    n_blocks_y = n_cells_y - block_height + 1
    n_blocks_x = n_cells_x - block_width + 1
    block_vector_length = block_height * block_width * n_bins
    features = np.zeros((n_blocks_y, n_blocks_x, block_vector_length))

    for y in range(n_blocks_y):
        for x in range(n_blocks_x):
            block_histograms = histograms[y:y + block_height, x:x + block_width, :]
            block_vector = block_histograms.flatten()
            block_vector /= np.linalg.norm(block_vector) + 1e-7
            features[y, x, :] = block_vector

    # Concatenate the block feature vectors to form the final feature vector
    feature_vector = features.flatten()

    # Show the original image
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.xticks([])
    plt.yticks([])

    # Show the HOG visualization
    plt.subplot(122)
    plt.pcolor(features[:, :, 0])
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('HOG Visualization')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.show()

def HOG():
    image = color.rgb2gray(data.astronaut())

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box')
    plt.show()

# image_path = '/home/pphuc/Coding/Project/FPTU/CPV/CPV/images/sofas.jpg'
# image = cv2.imread(image_path)
# HOG()