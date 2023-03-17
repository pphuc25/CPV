import cv2
import numpy as np
drawing = False
ix, iy = -1, -1


def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix = x
        iy = y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 255), -1)


def rotation(file, deg):
    img = cv2.imread(file)
    rows = img.shape[0]
    cols = img.shape[1]
    img_center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(img_center, deg, 1)
    rotated_image = cv2.warpAffine(img, M, (cols, rows), borderValue=(255,255,255))
    cv2.imshow('Rotated', rotated_image)


def translation(file):
    image = cv2.imread(file)
    #get the width and height of the image
    height, width = image.shape[:2]
    tx, ty = width / 4, height / 4
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty]], dtype=np.float32)
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height), borderValue=(255,255,255))
    cv2.imshow('Translated', translated_image)


def scale(file, scale_percent):
    image = cv2.imread(file)
    #calculate the percent of original dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    #resize image
    output = cv2.resize(image, (width, height))
    cv2.imshow('Scale', output)


img = np.zeros((512,512,1),dtype=np.uint8)
img.fill(255)
filename = 'org.png'
#change to your computer path for the code to work

cv2.namedWindow("Draw Rectangle")
cv2.setMouseCallback("Draw Rectangle", draw_rectangle)


#display the window
while True:
    cv2.imshow("Draw Rectangle", img)
    EditedImage = cv2.imwrite(filename, img)
    tx = int(input('Enter tx:'))
    ty = int(input('Enter ty:'))
    translation(filename)
    rotation(filename, 30) #change the degree
    scale(filename, 50) #change the percent
    if cv2.waitKey(10) == 27: #use escape key to close all window
        break
cv2.destroyAllWindows()
#there are 4 window
#drag the window away from each other to see all of them