import csv
import cv2
import numpy as np
import pandas as pd
import os
import pickle
from Assignment.pre_process import preprocess_img

ip = "rtsp://admin:ZSXNWK@192.168.225.155:554/H.264"

cap = cv2.VideoCapture(ip)
cap.set(cv2.CAP_PROP_FPS, 10)  # set frame rate to 10 fps

# Define the cropping coordinates (left, top, right, bottom)
crop_coords = (1080//2 - 360, 1920//2 - 480, 1080//2 + 360, 1920//2 + 480)

# load the trained model from file
with open('SVM.pkl', 'rb') as f:
    svm = pickle.load(f)

with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

# Define the path to the saved face images and labels
data_dir = 'face_data'

# Create a new CSV file for attendance
attendance_file = open('attendance.csv', mode='w', newline='')
attendance_writer = csv.writer(attendance_file)

# Write headers to the CSV file
attendance_writer.writerow(['Name', 'ID', 'Attendance'])

# Load the images and labels from the saved files
for label in os.listdir(data_dir):
    name, id = label.split('_')
    attendance_writer.writerow([name, id, 'Absent'])

# Close the CSV file
attendance_file.close()

# Load CSV file into dataframe
df = pd.read_csv('attendance.csv')

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start the webcam
# cap = cv2.VideoCapture(0)

# Set the threshold for recognizing unknown faces
threshold = 0.8

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Crop frame into image size
    frame = frame[crop_coords[0]: crop_coords[2], crop_coords[1]: crop_coords[3]]

    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = preprocess_img(frame)
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
        confidence_scores = svm.predict_proba(face_img_pca)
        print(confidence_scores)
        label = svm.predict(face_img_pca)

        if np.max(confidence_scores) >= threshold:
            # Draw a rectangle around the face and label it with the predicted name
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(label[0]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            name, id = label[0].split('_')
            # Find row corresponding to student ID
            row_index = df.loc[df['ID'] == id].index[0]

            # Update attendance for that row
            df.at[row_index, 'Attendance'] = 'Present'
        else:
            # Draw a rectangle around the face and label it as "unknown"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame with the recognized faces
    cv2.imshow('Face Recognition', frame)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1000) & 0xFF == ord('q'):

        # Save updated dataframe back to CSV file
        df.to_csv('attendance.csv', index=False)
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
