
import cv2
import numpy as np

# Load the image
image = cv2.imread('C:/Users/Ericw/CSE598_PlanarContinuumRobotPoseEstimation/Videos/FiduicialTest_3.png')

scale_x = 0.2  # Reduce width to 50%
scale_y = 0.2  # Reduce height to 50%

new_width = int(image.shape[1] * scale_x)
new_height = int(image.shape[0] * scale_y)
new_size = (new_width, new_height)

resized_img = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

# Convert the image to grayscale
gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)


#load aruco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()

# Create the ArUco detector
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

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

