import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import ConfusionMatrixDisplay

# Define the directory where the student images are stored
input_dir = "Student_Image"

# Get a list of all the student names in the input directory
students = os.listdir(input_dir)
data = []

# Loop over all the student images and recognize faces
for student in students:
    # Get a list of all the image files for the current student
    images = os.listdir(os.path.join(input_dir, student))
    name, id = student.split('_')

    # Loop over all the image files and recognize faces
    for image_file in images:
        # Load the image
        image_path = os.path.join(input_dir, student, image_file)
        image = cv2.imread(image_path)
        #Store image and student
        data.append((image, student))

    # Print the student name and predicted label
    print(f"Student: {name}, ID: {id}")

#Initalize face list
faces_list = []
size = (64, 64)  # desired size of the resized images


for image, student in data:
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

    # Loop over the detected faces and extract each one
    for (x, y, w, h) in faces:
        # Crop the image around the face
        face_img =gray[y:y + h, x:x + w]

        # Resize the face image
        face_img = np.array(cv2.resize(face_img, size))

        #Store face of each studentq
        faces_list.append((face_img, student))

        # Display the extracted face
        # cv2.imshow(student, face_img)
        # cv2.waitKey(0)

# Create an empty list to store the face images and their corresponding labels
face_images = []
face_labels = []

#Load face and labels
for face, label in faces_list:
    # Append the image and its label to the lists
    face_images.append(face)
    face_labels.append(label)

# Create a directory to save the images and labels
os.makedirs('face_data', exist_ok=True)

# Loop through each face and save it with its corresponding label
for i in range(len(face_images)):
    # Create a subdirectory with the label name
    label_dir = os.path.join('face_data', str(face_labels[i]))
    os.makedirs(label_dir, exist_ok=True)

    # Save the face image in the label subdirectory
    image_path = os.path.join(label_dir, f'face_{i}.png')
    cv2.imwrite(image_path, face_images[i])

# face_images = np.array(face_images)
# face_labels = np.array(face_labels)

# print(face_images.shape)

def plot_gallery(images, titles, h=480, w=640, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()

# plot_gallery(face_images, face_labels)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(face_images, face_labels, test_size=0.2, random_state=42)
#
# # Flatten the 2D images to 1D arrays
# X_train_flat = X_train.reshape(X_train.shape[0], -1)
# X_test_flat = X_test.reshape(X_test.shape[0], -1)
#
# # Scale the data using the same scaler used on the training set
# scaler = StandardScaler()
# X_test_scaled = scaler.fit_transform(X_test_flat)
#
# # Perform PCA to reduce the dimensionality of the data
# n_components = 2  # Number of principal components to keep
# pca = PCA(n_components=n_components, whiten=True, random_state=42)
# X_train_pca = pca.fit_transform(X_train_flat)
# X_test_pca = pca.transform(X_test_flat)
#
# # Train an SVM classifier on the training data
# svm = SVC(kernel='rbf', C=10, gamma=0.001, random_state=42)
# svm.fit(X_train_pca, y_train)
#
# # Evaluate the performance of the model on the testing data
# y_pred = svm.predict(X_test_pca)
# print(classification_report(y_test, y_pred))



