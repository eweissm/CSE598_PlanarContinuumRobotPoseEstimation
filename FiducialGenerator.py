"""
Generates a list of fiducials for use on the planar robot

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# lets used the 4x4 dictionary since we dont need many fiducials
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

marker_id = 10
marker_size = 200  # Size in pixels
marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

cv2.imwrite('Fiducials/marker_10.png', marker_image)
plt.imshow(marker_image, cmap='gray', interpolation='nearest')
plt.axis('off')  # Hide axes
plt.title(f'ArUco Marker {marker_id}')
plt.show()