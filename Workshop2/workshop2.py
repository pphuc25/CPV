import cv2
import random

def BrightnessContrast(brightness=0):
	# getTrackbarPos returns the current position (value) of the trackbar
	brightness = cv2.getTrackbarPos('Brightness','Original')
	contrast = cv2.getTrackbarPos('Contrast','Original')
	effect = controller(img, brightness,contrast)
	cv2.imshow('Effect', effect)

def controller(img, brightness=255, contrast=127):
	# Set the value of original image to be the center value (brightness = 255, contrast = 127)
	brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
	contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

	# IMAGE PROCESSING ALGORITHMS: CONTRAST ADJUSTMENT
	# https://stackoverflow.com/questions/39510072/algorithm-for-adjustment-of-image-levels/48859502#48859502
	if brightness != 0:
		if brightness > 0:
			shadow = brightness
			max = 255
		else:
			shadow = 0
			max = 255 + brightness
		al_pha = (max - shadow) / 255
		ga_mma = shadow
		cal = cv2.addWeighted(img, al_pha, img, 0, ga_mma)
	else:
		cal = img

	# IMAGE PROCESSING ALGORITHMS: CONTRAST ADJUSTMENT
	# https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
	if contrast != 0:
		Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
		Gamma = 127 * (1 - Alpha)
		cal = cv2.addWeighted(cal, Alpha,cal, 0, Gamma)
	return cal


def color_balance(original):		
	cv2.namedWindow('Original')
	cv2.imshow('Original', cv2.resize(original, (new_width, new_height)))

	# createTrackbar(trackbarName, windowName, value, count, onChange)
	# Brightness range -255 to 255
	cv2.createTrackbar('Brightness', 'Original', 255, 2 * 255, BrightnessContrast)
		
	# Contrast range -127 to 127
	cv2.createTrackbar('Contrast', 'Original', 127, 2 * 127, BrightnessContrast)

def median_filter(img, ksize=3):
	median = cv2.medianBlur(img, ksize)
	cv2.imshow('Original',img)
	cv2.imshow('Median filter', median)

def mean_filter(img, m=3, n=3):
	ksize = (m, n)
	mean = cv2.blur(img, ksize)
	cv2.imshow('Original',img)
	cv2.imshow('Mean filter', mean)

def gaussian_smooth(img, m=3, n=3):
	ksize = (m, n)
	gaussian = cv2.GaussianBlur(img, ksize, cv2.BORDER_DEFAULT)
	cv2.imshow('Original',img)
	cv2.imshow('Gaussian Smoothing', gaussian)

def create_salt_pepper_noise(img):
	row = img.shape[0]
	col = img.shape[1]
      
    # Randomly pick some pixels in the image for coloring them white
    # Pick a random number between 300 and 10000
	number_of_pixels = random.randint(300, 10000)
	for i in range(number_of_pixels):
        # Pick a random y coordinate
		y_coord=random.randint(0, row - 1)
        # Pick a random x coordinate
		x_coord=random.randint(0, col - 1)
        # Color that pixel to white
		img[y_coord][x_coord] = 255
    
	# Coloring black
	number_of_pixels = random.randint(300 , 10000)
	for i in range(number_of_pixels):
		y_coord=random.randint(0, row - 1)
		x_coord=random.randint(0, col - 1)
		img[y_coord][x_coord] = 0

	return img
  

original = cv2.imread("1259119.jpg")
p = 0.25
new_width = int(original.shape[1] * p)
new_height = int(original.shape[0] * p)
img = cv2.resize(original, (new_width, new_height))
#Function1
# color_balance(original)
#Function3
# median_filter(create_salt_pepper_noise(img))
#Function4
# mean_filter(create_salt_pepper_noise(img))
#Function5
# gaussian_smooth(img)
cv2.waitKey(0)