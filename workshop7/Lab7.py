import cv2
import os
from tkinter import *
from tkinter import Tk
from tkinter import ttk
from tkinter.filedialog import askdirectory

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
    
    return Images

def face_detection(image):

    # Load the face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the input image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

    # Draw rectangles around the detected faces in the original color image
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image

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
    raw_img = cv2.imread(dataset_path + fname)
    cv2.imshow('Original', raw_img)
    img = face_detection(raw_img)
    cv2.imshow('Detected', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


root = Tk()
root.title('Face Detection')
root.geometry('400x300')

file_name = ttk.Entry(root)
comment = Label(root, text="After choosing a folder,\n input a file name from the list that pop up below\n Then hit DETECT FACES!!!")
choose_button = ttk.Button(root, text='Select Folder', command=choose)
run_button = ttk.Button(root, text='DETECT FACES!!!', command=main_prog)

choose_button.pack(expand=True, pady=30)
comment.pack()
file_name.pack()
run_button.pack()

root.mainloop()