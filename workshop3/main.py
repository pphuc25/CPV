# import Canny
from harris_corner_p1 import HarrisCornerDetection
from Canny import Canny_detector
import cv2
from hough_transform import hough
from HOG import hog_visualization


if __name__ == "__main__":
    image_path = 'bookcase-chairs-clean-decor.jpeg'
    image = cv2.imread(image_path)
    chess = cv2.imread('chess.png')

    # Define menu options
    options = ["1. Harris Corner Detector", "2. HOG Feature Description", "3. Canny Edge Detection", "4. Hough Transform for Line Detection", "5. Exit"]

    # Print menu and prompt for user input
    while True:
        print("Select an option:")
        for option in options:
            print(option)
        choice = input("Enter option number: ")

        # Execute selected function or exit
        if choice == "1":
            HarrisCornerDetection(image, 0.65)
        elif choice == "2":
            hog_visualization(image)
        elif choice == "3":
            canny_img = Canny_detector(image, 30, 50)
            cv2.imshow('Origin', image)
            cv2.imshow('Canny', canny_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice == "4":
            hough(chess)
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select a valid option.")