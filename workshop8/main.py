import cv2
import os
from tkinter import *
from tkinter import Tk
from tkinter import ttk
from tkinter.filedialog import askdirectory
from eigenface import calculation, recogniser

def ReadImage(ImageFolderPath):
    Images = [] # Input Images will be stored in this list.

	# Checking if path is of folder.
    if os.path.isdir(ImageFolderPath): # If path is of a folder contaning images.
        ImageNames = os.listdir(ImageFolderPath)
        ImageNames_Split = [[(os.path.splitext(os.path.basename(ImageName))[0]), ImageName] for ImageName in ImageNames]
        ImageNames_Sorted = [ImageNames_Split[i][1] for i in range(len(ImageNames_Split))]
        
        for i in range(len(ImageNames_Sorted)): # Getting all image's name present inside the folder.
            ImageName = ImageNames_Sorted[i]
            InputImage = ImageName  # Reading images one by one.
            
            # Checking if image is read
            if InputImage is None:
                print("Not able to read image: {}".format(ImageName))
                exit(0)

            Images.append(InputImage) # Storing images.
            
    else: # If it is not folder(Invalid Path).
        print("\nEnter valid Image Folder Path.\n")
        
    if len(Images) < 2:
        print("\nNot enough images found. Please provide 2 or more images.\n")
        exit(1)
    
    return Images


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

def choose():
    global dataset_path
    path = askdirectory(title='Select folder')
    dataset_path = str(path)
    # Reading images
    Images = ReadImage(str(path))
    text = Text(root, width=30, height=15)
    text.pack()
    for image in Images:
        text.insert(END, image + ' \n')
    
def main_prog():
    global dataset_path
    fname = file_name.get()
    dataset_path = dataset_path + '/'
    dataset_dir  = os.listdir(dataset_path)
    mean_face, proj_data, w = calculation(width, height, dataset_path, train_image_names)
    recogniser(fname, train_image_names, mean_face, proj_data, w)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


root = Tk()
root.title('Eigenface')
root.geometry('400x300')

file_name = ttk.Entry(root)
comment = Label(root, text="After choosing a folder,\n input a file name from the list that pop up below\n Then hit RUN!!!")
choose_button = ttk.Button(root, text='Select Folder', command=choose)
run_button = ttk.Button(root, text='RUN!!!', command=main_prog)

choose_button.pack(expand=True, pady=30)
comment.pack()
file_name.pack()
run_button.pack()

root.mainloop()

# IF THE COLOR OF THE TEXT IN THE RESULT WINDOW IS GREEN THEN THE RESULT IS CORRECT,
# OTHERWISE (RED) THEN IT IS INCORRECT