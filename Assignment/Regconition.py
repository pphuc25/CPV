import cv2
import numpy as np
import pandas as pd
import datetime
import os
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pickle

# Define the path to the saved face images and labels
data_dir = 'face_data'

# Load the images and labels from the saved files
face_images = []
face_labels = []
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    if os.path.isdir(label_dir):
        for image_file in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            face_images.append(image)
            face_labels.append(label)

# Convert the image and label lists to numpy arrays
face_images = np.array(face_images)
face_labels = np.array(face_labels)

#Flatten array
face_images = face_images.reshape(face_images.shape[0], -1)

X = face_images
y = face_labels

# Perform PCA to reduce the dimensionality of the face embeddings
n_components = 10  # Number of principal components to keep
pca = PCA(n_components=n_components, whiten=True, random_state=42)
X_pca = pca.fit_transform(X)

# Train an SVM classifier on the face embeddings
svm = SVC(kernel='rbf', C=10, gamma=0.001, random_state=42)
svm.fit(X_pca, y)

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop through each face and recognize it
    for (x, y, w, h) in faces:
        # Extract the face region from the frame and resize it to the same size as the training data
        face_img = cv2.resize(gray[y:y+h, x:x+w], (64, 64))

        # Flatten the face image to a 1D array
        face_img_flat = face_img.reshape(1, -1)

        # Transform the face image using the same PCA object used for training
        face_img_pca = pca.transform(face_img_flat)

        # Predict the label of the face using the SVM classifier
        label = svm.predict(face_img_pca)

        # Draw a rectangle around the face and label it with the predicted name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, str(label[0]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with the recognized faces
    cv2.imshow('Face Recognition', frame)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
