import os
import pickle
import cv2
import numpy as np
from scipy.stats import loguniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

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

# Flatten the array
face_images = face_images.reshape(face_images.shape[0], -1)

# Split the data into training and testing sets
X = face_images
y = face_labels

# Shuffle X and y
indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# Perform PCA to reduce the dimensionality of the face embeddings
n_components = 50  # Number of principal components to keep
pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True)
X_pca = pca.fit_transform(X)

#Init Linear SVM
svm = LinearSVC(max_iter=10000)

# Define the hyperparameter grid to search over
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
}

# Create the GridSearchCV object
grid_search = GridSearchCV(svm, param_grid, cv=5)

# Fit the GridSearchCV object to the data
grid_search.fit(X_pca, y)

# Print the best score and best parameters found by the grid search
print(f"Best score: {grid_search.best_score_:.3f}")
print(f"Best parameters: {grid_search.best_params_}")

# Save the best model
svm = LinearSVC(penalty=grid_search.best_params_['penalty'], C=grid_search.best_params_['C'], max_iter=10000)
# Wrap the model in CalibratedClassifierCV to enable probability estimates
svm = CalibratedClassifierCV(svm, cv=5, method='sigmoid')
svm.fit(X_pca, y)

# Save the trained model and PCA object to a file
with open('SVM.pkl', 'wb') as f:
    pickle.dump(svm, f)

with open('pca.pkl', 'wb') as f:
    pickle.dump(pca, f)
