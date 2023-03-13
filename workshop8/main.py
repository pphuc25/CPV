import cv2
import os
from eigenface import calculation, recogniser

# Read images
dataset_path = 'Dataset/'
dataset_dir  = os.listdir(dataset_path)

# *** ALL OF THE IMAGES IN THE DATASET HAVE TO BE IN THE SAME SIZE IN ORDER TO CALCULATE ***
# Each image in the dataset is crop to the size of 195 x 231 (width x height) pixels 
# and each pixel uses 8 bits for grayscale
width  = 195
height = 231

# Choose which images is used for training
train_image_names = ['subject01.normal.jpg', 'subject02.normal.jpg', 
                     'subject03.normal.jpg', 'subject07.normal.jpg', 
                     'subject10.normal.jpg', 'subject11.normal.jpg', 
                     'subject14.normal.jpg', 'subject15.normal.jpg']

mean_face, proj_data, w = calculation(width, height, dataset_path, train_image_names)
recogniser('subject11.happy.jpg', train_image_names, mean_face, proj_data, w)
cv2.waitKey(0)
cv2.destroyAllWindows()

# IF THE COLOR OF THE TEXT IN THE RESULT WINDOW IS GREEN THEN THE RESULT IS CORRECT,
# OTHERWISE (RED) THEN IT IS INCORRECT