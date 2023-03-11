import cv2
import os

def input_image():
    # Get the name and id of the student from the user
    name = input("Enter the name of student: ")
    id = input("Enter the ID of student: ")
    print('Look at the Camera')

    # Create a directory for the student's images
    if not os.path.exists(f"Student_Image/{name}_{id}"):
        os.makedirs(f"Student_Image/{name}_{id}")

    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Loop over frames from the webcam and save 100 images
    for j in range(100):
        # Capture a frame from the webcam
        ret, frame = cap.read()

        # If the frame cannot be captured, break out of the loop
        if not ret:
            break

        # Save the image to a file
        file_path = f"Student_Image/{name}_{id}/{name}_{id}_{j}.jpg"
        cv2.imwrite(file_path, frame)

        # Display the image
        cv2.imshow('image', frame)
        cv2.waitKey(50)

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

input_image()
