import cv2
import numpy as np

def face_detection(image):

    face_kernel = np.array([
        [ 1,  1,  1,  1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1,  1,  1,  1,  1],
    ])

    # Load the input image
    # image = cv2.imread("workshop7/hoahau.jpg")

    # Convert the input image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the window size and the stride of the kernel
    win_w, win_h = face_kernel.shape
    stride = 2

    # Initialize an array to store the scores for each window
    scores = np.zeros_like(gray, dtype=np.float32)

    # Calculate the integral image
    int_img = cv2.integral(gray)

    face_cascade = cv2.CascadeClassifier(button_)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)
    for y in range(0, gray.shape[0] - win_h, stride):
        for x in range(0, gray.shape[1] - win_w, stride):
            
            # Extract the window from the integral image
            i = y + win_h
            j = x + win_w
            window_sum = int_img[i, j] - int_img[i, x] - int_img[y, j] + int_img[y, x]
            window = gray[y:y+win_h, x:x+win_w]
            
            # Calculate the score for the window using the Haar Cascade kernel
            score = np.sum(window * face_kernel) / np.sqrt(np.sum(window ** 2))
            scores[y, x] = score

    # Find the local maxima in the scores
    maxima = cv2.dilate(scores, np.ones((3,3))) == scores
    maxima = maxima.astype(np.uint8)

    # Draw rectangles around the detected faces in the original color image
    faces = []
    for y, x in np.argwhere(maxima):
        faces.append((x, y, win_w, win_h))


    # Return face direction
    for (x, y, w, h) in face:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

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
raw_img = cv2.imread('workshop7/hoahau.jpg')
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