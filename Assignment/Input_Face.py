import re
import subprocess
import cv2
import os
import numpy as np
from Assignment.pre_process import preprocess_img
from CameraIP import ip, crop_coords

def input_image():

    # Create a directory to save the images and labels
    os.makedirs('face_data', exist_ok=True)

    # Get the existing names and IDs of the students
    names = []
    IDs = []
    for directory in os.listdir("face_data"):
        if os.path.isdir(os.path.join("face_data", directory)):
            name, id = directory.split("_")
            names.append(name)
            IDs.append(id)

    # Get the name and id of the student from the user
    name = input("Enter the name of student: ")
    while True:
        id = input("Enter the ID of student (format: SE******): ")
        if re.match(r'^SE\d{6}$', id):
            break
        else:
            print("Invalid format. Please enter the ID in the format SE******.")

    # Check if the ID is already in the folder
    if id in IDs:
        print("You have already input face for this ID.")
        return
    else:
        print('Look at the Camera')

        # Create a directory for the student's images
        if not os.path.exists(f"face_data/{name}_{id}"):
            os.makedirs(f"face_data/{name}_{id}")

    student = name + '_' + id

    # Initalize face list
    face_images = []
    face_labels = []
    size = (64, 64)  # desired size of the resized images
    j = 0

    # Open the webcam
    cap = cv2.VideoCapture(ip)

    # Load the face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read a frame from the camera
    ret, frame = cap.read()

    #Crop frame into image size
    frame = frame[crop_coords[0] : crop_coords[2], crop_coords[1] : crop_coords[3]]

    # Loop over frames from the webcam and save 50 images
    while j < 50:
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
            face_labels.append(student)

            # Count face
            j += 1

        # Gap between image
        cv2.imshow('Image', frame)
        cv2.waitKey(1)

    # Loop through each face and save it with its corresponding label
    for i in range(len(face_images)):
        # Create a subdirectory with the label name
        label_dir = os.path.join('face_data', str(face_labels[i]))
        os.makedirs(label_dir, exist_ok=True)

        # Save the face image in the label subdirectory
        image_path = os.path.join(label_dir, f'face_{i}.png')
        cv2.imwrite(image_path, face_images[i])

    #Rerun the Train
    subprocess.call(['python', 'Train.py'])

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

input_image()
