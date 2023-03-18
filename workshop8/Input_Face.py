import re
import subprocess
import cv2
import os
import numpy as np
from pre_process import preprocess_img


# Define the cropping coordinates (left, top, right, bottom)
crop_coords = (1080//2 - 360, 1920//2 - 480, 1080//2 + 360, 1920//2 + 480)

def input_image():

    # Initalize face list
    face_images = []
    size = (64, 64)  # desired size of the resized images
    j = 0

    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Load the face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read a frame from the camera
    ret, frame = cap.read()

    #Crop frame into image size
    frame = frame[crop_coords[0] : crop_coords[2], crop_coords[1] : crop_coords[3]]

    # Loop over frames from the webcam and save 150 images
    while j < 1:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        # If the frame cannot be captured, break out of the loop
        if not ret:
            break

        # Convert the image to grayscale
        gray = preprocess_img(frame)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

        if faces is None:
            continue

        for (x, y, w, h) in faces:
            # Crop the image around the face
            face_img = gray[y:y + h, x:x + w]

            # Resize the face image
            face_img = np.array(cv2.resize(face_img, size))

            # Store face of each studentq
            face_images.append(face_img)

            # Count face
            j += 1

        # Gap between image
        cv2.imshow('Image', frame)
        cv2.waitKey(1)

    # Loop through each face and save it
    for i in range(len(face_images)):
        image_path = os.path.join('D:\FPT\SPRING23\CPV301\CPV301_code\CPV\workshop8\Test', f'face_{i}.png')
        print(image_path)
        cv2.imwrite(image_path, face_images[i])

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

input_image()
