import os
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC, SVC


def face_detection(image):
    # Define paths to positive and negative image folders
    pos_folder = 'Pos'
    neg_folder = 'Neg'

    # Load positive and negative images
    pos_imgs = []
    for filename in os.listdir(pos_folder):
        if filename.endswith('.jpg'):
            img_path = os.path.join(pos_folder, filename)
            img = cv2.imread(img_path)
            pos_imgs.append(img)

    neg_imgs = []
    for filename in os.listdir(neg_folder):
        if filename.endswith('.jpg'):
            img_path = os.path.join(neg_folder, filename)
            img = cv2.imread(img_path)
            neg_imgs.append(img)

    # Define HOG parameters
    win_size = (64, 128)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9

    # Compute HOG features for positive and negative training images
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    pos_feats = [hog.compute(cv2.resize(img, win_size)) for img in pos_imgs]
    neg_feats = [hog.compute(cv2.resize(img, win_size)) for img in neg_imgs]

    # Concatenate positive and negative features and labels
    X = np.concatenate((np.array(pos_feats), np.array(neg_feats)), axis=0)
    y = np.concatenate((np.ones(len(pos_feats)), np.zeros(len(neg_feats))), axis=0)

    # Train SVM classifier with probability estimates
    clf = LinearSVC()
    clf.fit(X, y)

    # Wrap the model in CalibratedClassifierCV to enable probability estimates
    calibrated_clf = CalibratedClassifierCV(clf, cv=5, method='sigmoid')
    calibrated_clf.fit(X, y)

    # Load test image and perform sliding window object detection
    test_image = image
    window_size = (64, 128)
    stride = 8
    threshold = 0.905
    detections = []
    for y in range(0, test_image.shape[0] - window_size[1], stride):
        for x in range(0, test_image.shape[1] - window_size[0], stride):
            window = test_image[y:y + window_size[1], x:x + window_size[0]]
            hog_feat = hog.compute(cv2.resize(window, win_size))
            class_probabilities = calibrated_clf.predict_proba([hog_feat])
            print(class_probabilities)
            if np.max(class_probabilities) > threshold:
                detections.append((x, y, window_size[0], window_size[1]))

    # Draw bounding boxes around detected objects
    for detection in detections:
        x, y, w, h = detection
        cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return test_image

def on_button_click(event, x, y, flags, param):
    global img
    global button_hover
    global button_
    global button_text

    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x and x <= 10 + button_size[0] and 10 <= y and y <= 10 + button_size[1]:
            if button_text == 'DETECT FACE':
                print("Face Detect")
                img = face_detection(img)

# Load the input image
raw_img = cv2.imread('hoahau.jpg')
img = raw_img

# Create the window
window_name = 'Face Detecter'
cv2.namedWindow(window_name)

# Get the size of the combined image
height, width, _ = img.shape

# Define button properties
button_text = "DETECT FACE"
name_d = 'haarcascade_frontalface_default.xml'
button_font = cv2.FONT_ITALIC
button_font_scale = 0.6
button_thickness = 1
button_ = cv2.data.haarcascades + name_d
button_padding = 10
button_color = (255, 0, 0)
button_hover_color = (0, 255, 0)

# Get button size and position
button_size, _ = cv2.getTextSize(button_text, button_font, button_font_scale, button_thickness)
button_rect = ((0, 0), (button_size[0] + button_padding*2, button_size[1] + button_padding*2))

# Initialize button hover state
button_hover = False

# Show the image with the button
cv2.setMouseCallback(window_name, on_button_click)

# Wait for a button click
while True:
    #Display the combined image
    cv2.imshow(window_name, img)

    # Draw button
    if button_hover:
        button_text_color = button_hover_color
    else:
        button_text_color = button_color
    cv2.rectangle(img, button_rect[0], button_rect[1], button_text_color, -1)
    cv2.putText(img, button_text,
                (button_rect[0][0] + button_padding, button_rect[0][1] + button_size[1] + button_padding), button_font,
                button_font_scale, (255, 255, 255), button_thickness, cv2.LINE_AA)

    # Check for key events
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Close the window
cv2.destroyAllWindows()