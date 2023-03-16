import sys
import cv2
import numpy as np
import random


# This draws matches and optionally a set of inliers in a different color
def drawMatches(img1, kp1, img2, kp2, matches, inliers = None):
    # Create a new output image that concatenates the two images together
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns, y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        inlier = False

        if inliers is not None:
            for i in inliers:
                if i.item(0) == x1 and i.item(1) == y1 and i.item(2) == x2 and i.item(3) == y2:
                    inlier = True

        # Draw a small circle at both co-ordinates
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points, draw inliers if we have them
        if inliers is not None and inlier:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1)
        elif inliers is not None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)

        if inliers is None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    return out


# Runs sift algorithm to find features
def findFeatures(img):
    print("Finding Features...")
    sift = cv2.SIFT()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    img = cv2.drawKeypoints(img, keypoints)
    cv2.imwrite('sift_keypoints.png', img)

    return keypoints, descriptors

# Matches features given a list of keypoints, descriptors, and images
def matchFeatures(kp1, kp2, desc1, desc2, img1, img2):
    print("Matching Features...")
    matcher = cv2.BFMatcher(cv2.NORM_L2, True)
    matches = matcher.match(desc1, desc2)
    matchImg = drawMatches(img1,kp1,img2,kp2,matches)
    cv2.imwrite('Matches.png', matchImg)
    return matches



# Computers a homography from 4-correspondences
def calculateHomography(correspondences):
    # Loop through correspondences and create assemble matrix
    aList = []
    for corr in correspondences:
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    matrixA = np.matrix(aList)

    # Svd composition
    u, s, v = np.linalg.svd(matrixA)

    # Reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))

    # Normalize and now we have h
    h = (1/h.item(8)) * h
    return h



# Calculate the geometric distance between estimated points and original points
def geometricDistance(correspondence, h):

    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


# Runs through RANSAC algorithm, creating homographies from random correspondences
def ransac(corr, thresh):
    maxInliers = []
    finalH = None
    for i in range(1000):
        # Find 4 random points to calculate a homography
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr4))

        # Call the homography function on those points
        h = calculateHomography(randomFour)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h

        if len(maxInliers) > (len(corr)*thresh):
            break
    return finalH, maxInliers


#
# Main parses argument list and runs the functions
#
def ransac_image_alignment(img1, img2, ransac_reproj_threshold=5.0):

    #find features and keypoints
    correspondenceList = []
    if img1 is not None and img2 is not None:
        kp1, desc1 = findFeatures(img1)
        kp2, desc2 = findFeatures(img2)
        keypoints = [kp1,kp2]
        matches = matchFeatures(kp1, kp2, desc1, desc2, img1, img2)
        for match in matches:
            (x1, y1) = keypoints[0][match.queryIdx].pt
            (x2, y2) = keypoints[1][match.trainIdx].pt
            correspondenceList.append([x1, y1, x2, y2])

        corrs = np.matrix(correspondenceList)

        #run ransac algorithm
        finalH, inliers = ransac(corrs, ransac_reproj_threshold)

        matchImg = drawMatches(img1,kp1,img2,kp2,matches,inliers)
        cv2.imwrite('InlierMatches.png', matchImg)

        # Warp image 1 to align with image 2
        aligned_img1 = cv2.warpPerspective(img1, finalH, (img2.shape[1], img2.shape[0]))

        # Combine the two images for visualization
        combined_img = cv2.hconcat([img2, aligned_img1])

        return combined_img


def on_button_click(event, x, y, flags, param):
    global combined_image
    global button_hover
    global button_text

    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x and x <= 10 + button_size[0] and 10 <= y and y <= 10 + button_size[1]:
            if button_text == 'Alignment':
                print("Alignment Image")
                button_text = 'Raw'
                combined_image = ransac_image_alignment(image1, image2, 'ORB', 'BFMatcher')
            else:
                print('Raw')
                button_text = 'Alignment'
                combined_image = cv2.hconcat([image1, image2])

# Load the two images
image1 = cv2.imread("book.png")
image2 = cv2.imread("book2.png")

# aligned_img = ransac_image_alignment(img1, img2, 'ORB', 'BFMatcher')

# Create the window
window_name = 'Alignment Image'
cv2.namedWindow(window_name)

# Combine the images horizontally
combined_image = cv2.hconcat([image1, image2])

# Get the size of the combined image
height, width, _ = combined_image.shape

# Define button properties
button_text = "Alignment"
button_font = cv2.FONT_HERSHEY_SIMPLEX
button_font_scale = 1
button_thickness = 2
button_padding = 10
button_color = (255, 0, 0)
button_hover_color = (0, 255, 0)

# Get button size and position
button_size, _ = cv2.getTextSize(button_text, button_font, button_font_scale, button_thickness)
button_rect = ((10, 10), (button_size[0] + button_padding*2, button_size[1] + button_padding*2))

# Initialize button hover state
button_hover = False

# Show the image with the button
cv2.setMouseCallback(window_name, on_button_click)

# Wait for a button click
while True:
    def aligment_image():
        combined_image = ransac_image_alignment(image1, image2, 'ORB', 'BFMatcher')
    #Display the combined image
    cv2.imshow(window_name, combined_image)

    # Draw button
    if button_hover:
        button_text_color = button_hover_color
    else:
        button_text_color = button_color
    cv2.rectangle(combined_image, button_rect[0], button_rect[1], button_text_color, -1)
    cv2.putText(combined_image, button_text,
                (button_rect[0][0] + button_padding, button_rect[0][1] + button_size[1] + button_padding), button_font,
                button_font_scale, (255, 255, 255), button_thickness, cv2.LINE_AA)

    # Check for key events
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Close the window
cv2.destroyAllWindows()






