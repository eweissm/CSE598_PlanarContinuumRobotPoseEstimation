
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Load the image
image = cv2.imread('C:/Users/Ericw/CSE598_PlanarContinuumRobotPoseEstimation/Videos/FiduicialTest_3.png')

scale_x = 0.2  # Reduce width to 50%
scale_y = 0.2  # Reduce height to 50%

new_width = int(image.shape[1] * scale_x)
new_height = int(image.shape[0] * scale_y)
new_size = (new_width, new_height)

resized_img = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
bright_image = cv2.convertScaleAbs(resized_img, alpha=2.0, beta=50)  # Increase brightness

# Convert the image to grayscale
gray = cv2.cvtColor(bright_image, cv2.COLOR_BGR2GRAY)


#load aruco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()

# Create the ArUco detector
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Detect the markers
corners, ids, rejected = detector.detectMarkers(gray)

# Print the detected markers
print("Detected markers:", ids)
Marked_image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
# plt.figure()
# plt.imshow(Marked_image, origin="upper")

if ids is not None:
#     for i in range(len(ids)):
#         c = corners[i][0]
#         plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "+", label = "id={0}".format(ids[i]))
# # plt.legend()
# # plt.show()
#     cv2.imshow('Detected Markers', Marked_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    mask = cv2.subtract(Marked_image, image)  # Find the drawn areas
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)  # Convert to binary mask

    only_markers = cv2.bitwise_and(Marked_image, Marked_image, mask=mask_binary)

    # Add the markers to the original image
    overlay = cv2.add(image, only_markers)

    # Show result
    cv2.imshow('Overlayed Image', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

