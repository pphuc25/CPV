import numpy as np
import cv2
import math
import os
import random

# Insert the path containing the images
# Images have to be named alphabetically based on their order in panorama composition
directory = "./book/"
num_of_images = 0

def load_images():
    """
    Load images from directory
    :return:
    ret_images: List of images name
    """
    global num_of_images
    num_of_images = 0
    images = os.listdir(directory)
    images.sort()
    ret_images = []
    image_names = []
    for img in images:
        image_names.append(img[:-4])
        ret_images.append(cv2.imread(directory + img, cv2.IMREAD_UNCHANGED))
        num_of_images += 1

    return ret_images, image_names

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
    keypoints = [None] * num_of_images
    descriptors = [None] * num_of_images
    i = 0
    for img in gray_images:
        keypoints[i], descriptors[i] = sift.detectAndCompute(img, None)
        i += 1

    # Draw keypoints
    image_keypoints = [None] * num_of_images
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
    homographies = []
    inlier_images = []
    for i in range(0, num_of_images - 1):
        src = np.float32([keypoints[i][g[0].queryIdx].pt for g in good[i]])
        dst = np.float32([keypoints[i+1][g[0].trainIdx].pt for g in good[i]])

        H, mask = ransac(src, dst)
        homographies.append(H)

        # Draw inliers
        img_inliers = np.empty((max(cv2_images[i].shape[0], cv2_images[i+1].shape[0]), cv2_images[i].shape[1] + cv2_images[i+1].shape[1], 3),dtype=np.uint8)
        good_temp = np.array(good[i])
        inliers = good_temp[np.where(np.squeeze(mask) == 1)[0]]
        cv2.drawMatchesKnn(cv2_images[i], keypoints[i], cv2_images[i+1], keypoints[i+1], outImg=img_inliers, matches1to2=inliers, flags=2)
        inlier_images.append(img_inliers)

    return homographies, inlier_images

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

if __name__ == "__main__":
    #Load and plot images
    print("Loading Images...")
    cv2_images, names = load_images()

    #Compute sift features and plot images
    print("Computing SIFT Features...")
    sift_images, keypoints, descriptors = sift_features(cv2_images)

    #Calculate matching and plot images
    print("Finding Good Matching...")
    matching_images, good = calculate_matching(sift_images, keypoints, descriptors)

    #Calculate homographies, remove outliers and plot images
    print("Removing Outliers and Computing Homography Transformations...")
    homographies, inlier_images = prepare_data_and_run_ransac(keypoints, good, cv2_images)
 
    #Blending and Stiching and plot images
    print("Aligning Images...")
    # Warp image 1 to align with image 2
    aligned_img1 = cv2.warpPerspective(cv2_images[0], homographies[0], (cv2_images[1].shape[1], cv2_images[1].shape[0]))    
    cv2.imshow('Result', aligned_img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
