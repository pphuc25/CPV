# import Canny
from harris_corner_p1 import HarrisCornerDetection
from Canny import Canny_detector
import cv2
from CPV_LinesDetection import hough_transform
from Func2_Lab3_CPV import task2


if __name__ == "__main__":
    image_path = '/home/pphuc/Coding/Project/FPTU/CPV/CPV/images/bookcase-chairs-clean-decor.jpeg'
    image = cv2.imread(image_path)


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
            HarrisCornerDetection(image, 0.85)
        elif choice == "2":
            result, theta = task2(image)
            cv2.imshow('Origin', image)
            cv2.imshow('Canny', result)
            cv2.waitKey(0)
        elif choice == "3":
            canny_img = Canny_detector(image, 30, 50)
            cv2.imshow('Origin', image)
            cv2.imshow('Canny', canny_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice == "4":
            hough_transform(image)
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select a valid option.")