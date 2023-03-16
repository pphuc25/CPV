import sys
import cv2
import numpy as np

def ransac_image_alignment(img1, img2, feature_detector='ORB', feature_matcher='BFMatcher',
                           ransac_reproj_threshold=5.0):

    # Initialize the feature detector and matcher
    if feature_detector == 'ORB':
        detector = cv2.ORB_create()
    elif feature_detector == 'SIFT':
        detector = cv2.SIFT_create()
    elif feature_detector == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()
    else:
        raise ValueError("Unsupported feature detector type")

    if feature_matcher == 'BFMatcher':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif feature_matcher == 'FLANNMatcher':
        matcher = cv2.FlannBasedMatcher_create()
    else:
        raise ValueError("Unsupported feature matcher type")

    # Detect features in both images
    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

    # Match features between the images
    matches = matcher.match(descriptors1, descriptors2)

    # Convert the keypoints to numpy arrays
    points1 = np.array([keypoints1[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
    points2 = np.array([keypoints2[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)

    # Estimate the alignment transformation matrix using RANSAC
    M, mask = cv2.findHomography(points1, points2, cv2.RANSAC, ransac_reproj_threshold)

    # Warp image 1 to align with image 2
    aligned_img1 = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))

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
image1 = cv2.imread("book1.png")
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






