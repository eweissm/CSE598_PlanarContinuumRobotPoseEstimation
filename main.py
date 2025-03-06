
import cv2
import numpy as np
video_path = 'Videos/MVI_0698.MOV'  # Replace with the actual path to your video file
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

scale_x = 0.5  # Reduce width to 50%
scale_y = 0.5  # Reduce height to 50%

numMarkers=8

#load aruco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

#arUco Parameters
parameters = cv2.aruco.DetectorParameters()
# parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
# parameters.adaptiveThreshWinSizeMin = 3
# parameters.adaptiveThreshWinSizeMax = 23
# parameters.adaptiveThreshWinSizeStep = 10
# parameters.adaptiveThreshConstant = 7
# parameters.polygonalApproxAccuracyRate = 0.02
# parameters.minMarkerPerimeterRate = 0.02
# parameters.minCornerDistanceRate = 0.05
# parameters.minMarkerDistanceRate = 0.05

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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 10)

    # Detect the markers
    corners, ids, rejected = detector.detectMarkers(gray)

    #draw on detected markers
    frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    if len(ids) == numMarkers:

        for i in range(numMarkers):
            ThisCorner = corners[i]
            center_x = int(np.mean(ThisCorner[0][:, 0]))
            center_y = int(np.mean(ThisCorner[0][:, 1]))

            # Draw a circle at the center
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(ids[i]), (center_x, center_y),cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2, cv2.LINE_AA)
    # Process the frame (e.g., display it)
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

