import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

def face_detection(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the image to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply a Sobel filter to the blurred image to detect edges
    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    edges = (edges / np.max(edges)) * 255
    edges = edges.astype(np.uint8)

    # Threshold the edges to obtain a binary image
    threshold_value, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Erode and dilate the binary image to remove noise and fill gaps
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Find the minimum rectangle that can fit around the contour
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Draw the rectangle around the face
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

    return image

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
raw_img = cv2.imread('face.jpg')
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