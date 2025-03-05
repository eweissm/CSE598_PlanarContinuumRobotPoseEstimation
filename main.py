
import cv2
import numpy as np
video_path = 'Videos/FiducialTest_1.MOV'  # Replace with the actual path to your video file
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()


scale_x = 0.5  # Reduce width to 50%
scale_y = 0.5  # Reduce height to 50%

#load aruco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()

# Create the ArUco detector
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

while True:
    ret, frame = cap.read()

    # If frame is not read correctly, break the loop
    if not ret:
        break

    new_width = int(frame.shape[1] * scale_x)
    new_height = int(frame.shape[0] * scale_y)
    new_size = (new_width, new_height)

    frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    # Process the frame (e.g., display it)
    cv2.imshow('Frame', frame)


    # Break the loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Load the image
image = cv2.imread('Videos/FiduicialTest_3.png')

scale_x = 0.2  # Reduce width to 50%
scale_y = 0.2  # Reduce height to 50%

new_width = int(image.shape[1] * scale_x)
new_height = int(image.shape[0] * scale_y)
new_size = (new_width, new_height)

resized_img = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

# Convert the image to grayscale
gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

# Detect the markers
corners, ids, rejected = detector.detectMarkers(gray)

# Print the detected markers
print("Detected markers:", ids)
Marked_image = cv2.aruco.drawDetectedMarkers(resized_img, corners, ids)

if ids is not None:

    # Show result
    cv2.imshow('Overlayed Image', Marked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

