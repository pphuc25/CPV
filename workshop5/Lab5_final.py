import os
import cv2
import numpy as np
import math
import random
from tkinter import Tk
from tkinter import ttk
from tkinter.filedialog import askdirectory

def ReadImage(ImageFolderPath):
    Images = [] # Input Images will be stored in this list.

	# Checking if path is of folder.
    if os.path.isdir(ImageFolderPath): # If path is of a folder contaning images.
        ImageNames = os.listdir(ImageFolderPath)
        ImageNames_Split = [[(os.path.splitext(os.path.basename(ImageName))[0]), ImageName] for ImageName in ImageNames]
        ImageNames_Sorted = [ImageNames_Split[i][1] for i in range(len(ImageNames_Split))]
        
        for i in range(len(ImageNames_Sorted)): # Getting all image's name present inside the folder.
            ImageName = ImageNames_Sorted[i]
            InputImage = cv2.imread(ImageFolderPath + "/" + ImageName)  # Reading images one by one.
            
            # Checking if image is read
            if InputImage is None:
                print("Not able to read image: {}".format(ImageName))
                exit(0)

            Images.append(InputImage) # Storing images.
            
    else: # If it is not folder(Invalid Path).
        print("\nEnter valid Image Folder Path.\n")
        
    if len(Images) != 2:
        print("\nNot enough images found. Please provide 2 images.\n")
        exit(1)
    
    return Images


def sift_features(images):
    """
    Use the SIFT (Scale Invariant Feature Transform) detector in order to detect features on each image
    :param images: List of images name
    :return:
    image_keypoints: Image with keypoints draw
    keypoints: Array containing keypoints
    descriptors: Array containing descriptors
    """
    gray_images = []
    for img in images:
        gray_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoint and descriptor with SIFT
    keypoints = [None] * 2
    descriptors = [None] * 2
    i = 0
    for img in gray_images:
        keypoints[i], descriptors[i] = sift.detectAndCompute(img, None)
        i += 1

    # Draw keypoints
    image_keypoints = [None] * 2
    i = 0
    for img in images:
        image_keypoints[i] = np.empty(img.shape, dtype=np.uint8)
        cv2.drawKeypoints(img, keypoints[i], image_keypoints[i])
        i += 1

    return image_keypoints, keypoints, descriptors

def calculate_matching(images, keypoints, descriptors):
    """
    Find correspondences between the features of pairs of images.
    Use of the k-Nearest Neighbors method, for each feature descriptor from the
    source image, to calculate its 2-nn (2-nearest neighbors) on the destination image
    :param images: Images exported from SIFT procedure
    :param keypoints: List containing keypoints of every image
    :param descriptors: List containing descriptors of every image
    :return:
    out_matches: List containing pairs of images with their matches draw
    good_matches: List containing good matches
    """
    out_images = []
    good_matches = []
    for i in range(0, len(images) - 1):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors[i], descriptors[i+1], k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        # Draw total matches
        img_matches = np.empty((max(images[i].shape[0], images[i+1].shape[0]), images[i].shape[1] + images[i+1].shape[1], 3),dtype=np.uint8)
        cv2.drawMatchesKnn(images[i], keypoints[i], images[i+1], keypoints[i+1], outImg=img_matches, matches1to2=good, flags=2)

        out_images.append(img_matches)
        good_matches.append(good)

    return out_images, good_matches


def ransac(src_points, dst_points, ransac_reproj_threshold=1, max_iters=1000, inlier_ratio=0.8):
    """
    Calculate the set of inlier correspondences w.r.t. homography transformation, using the
    RANSAC method.
    :param src_points: numpy.array(float), coordinates of the points in the source image
    :param dst_points: numpy.array(float), coordinates of the points in the destination image
    :param ransac_reproj_threshold: float, maximum allowed reprojection error to treat a point pair
    as an inlier
    :param max_iters: int, the maximum number of RANSAC iterations
    :param inlier_ratio: float, ratio of inliers w.r.t. total number of correspondences
    :return:
    H: numpy.array(float), the estimated homography transformation
    mask: numpy.array(uint8), mask that denotes the inlier correspondences
    """

    assert src_points.shape == dst_points.shape, print("Source and Destination dimensions have to be the same!")
    assert ransac_reproj_threshold >= 0, print("Reprojection Threshold has to be greater or equal to zero!")
    assert max_iters > 0, print("Max iterations has to be greater than zero!")
    assert (inlier_ratio >= 0) and (inlier_ratio <= 1), print("Inlier Ratio has to be in range [0,1]")

    H = []
    mask = []
    max_inliers = 0
    count = 0
    while count < max_iters:
        # Convert to Homogeneous
        temp_src = np.ones((src_points.shape[0], src_points.shape[1] + 1))
        temp_src[:, :-1] = src_points
        temp_dst = np.ones((dst_points.shape[0], dst_points.shape[1] + 1))
        temp_dst[:, :-1] = dst_points

        # 1. Select 4 random points
        random_indexes = random.sample(range(0, src_points.shape[0]), 4)
        pts1 = np.float32([src_points[random_indexes[0]], src_points[random_indexes[1]], src_points[random_indexes[2]], src_points[random_indexes[3]]])
        pts2 = np.float32([dst_points[random_indexes[0]], dst_points[random_indexes[1]], dst_points[random_indexes[2]], dst_points[random_indexes[3]]])

        # 1.1. Remove selected points
        np.delete(temp_src, random_indexes[0])
        np.delete(temp_src, random_indexes[1])
        np.delete(temp_src, random_indexes[2])
        np.delete(temp_src, random_indexes[3])

        # 2. Calculate Homography transformation
        H_temp = cv2.getPerspectiveTransform(src=pts1, dst=pts2)

        # 3. Transform rest points using H matrix
        projected = []
        for point in temp_src:
            projected.append(np.dot(H_temp, point))

        # 3.1 / 4. Calculate Euclidean distance AND Find inliers using reprojection threshold
        temp_inliers = [None] * src_points.shape[0]
        # Add the 4 selected points to mask
        temp_inliers[random_indexes[0]] = [1]
        temp_inliers[random_indexes[1]] = [1]
        temp_inliers[random_indexes[2]] = [1]
        temp_inliers[random_indexes[3]] = [1]

        inlier_count = -4
        for i in range(0, len(projected)):
            temp_inliers[i] = [0]
            # Normalize projected points
            projected[i] = projected[i] / projected[i][2]
            distance = math.sqrt((dst_points[i][0] - projected[i][0]) ** 2 + (dst_points[i][1] - projected[i][1]) ** 2)
            if distance <= ransac_reproj_threshold:
                inlier_count += 1
                temp_inliers[i] = [1]

        # 5. If greater than inlier ratio, break and return
        if inlier_count >= (inlier_ratio * src_points.shape[0]):
            H = H_temp
            mask = temp_inliers
            max_inliers = inlier_count
            break
        else:
            if inlier_count > max_inliers:
                max_inliers = inlier_count
                H = H_temp
                mask = temp_inliers

        count += 1

    return H, mask


def prepare_data_and_run_ransac(keypoints, good, images):
    """
    Calculate source and destination points and then run ransac implementations
    :param keypoints: List containing keypoints of every image
    :param good:  List containing good matches between pairs of images
    :param images: Original images
    :param option: If "mine" run custom ransac function, else run OpenCV's ransac
    :return:
    homographies: List containing homography transformation matrix for every pair of images
    inlier_images: Images with inliers draw
    """

    src = np.float32([keypoints[0][g[0].queryIdx].pt for g in good[0]])
    dst = np.float32([keypoints[1][g[0].trainIdx].pt for g in good[0]])
    H, mask = ransac(src, dst)

    # Draw inliers
    img_inliers = np.empty((max(images[0].shape[0], images[1].shape[0]), images[0].shape[1] + images[1].shape[1], 3),dtype=np.uint8)
    good_temp = np.array(good[0])
    inliers = good_temp[np.where(np.squeeze(mask) == 1)[0]]
    cv2.drawMatchesKnn(images[0], keypoints[0], images[1], keypoints[1], outImg=img_inliers, matches1to2=inliers, flags=2)

    return H, img_inliers


def main_prog():
    path = askdirectory(title='Select folder')
    print(path)
    # Reading images
    Images = ReadImage(str(path))
    
    text = ttk.Label(text="Please wait a moment")
    text.pack()
    
    # Compute sift features of images
    print("Computing SIFT Features...")
    sift_images, keypoints, descriptors = sift_features(Images)

    # Calculate matching images
    print("Finding Good Matching...")
    matching_images, good = calculate_matching(sift_images, keypoints, descriptors)

    # Calculate homographies, remove outliers of images
    print("Removing Outliers and Computing Homography Transformations...")
    homographies, inlier_images = prepare_data_and_run_ransac(keypoints, good, Images)
 
    # Blending and Aligning images
    print("Aligning Images...")
    aligned_img1 = cv2.warpPerspective(Images[0], homographies, (Images[1].shape[1], Images[1].shape[0]))    

    #cv2.imshow("Image 1", Images[0])
    #cv2.imshow("Image 2", Images[1])
    cv2.imshow("Matching Images", matching_images[0])
    cv2.imshow("Result", aligned_img1)
    cv2.waitKey()
    cv2.destroyAllWindows()

root = Tk()
root.title('Aligning')
root.geometry('600x300')

choose_button = ttk.Button(root, text='Select Folder', command=main_prog)

choose_button.pack(expand=True)

root.mainloop()






