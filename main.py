
import cv2
import numpy as np
import math


video_path = 'Videos/MVI_0698.MOV'  # Replace with the actual path to your video file
cap = cv2.VideoCapture(video_path)

scale_x = 0.8  # Reduce width to 50%
scale_y = 0.8  # Reduce height to 50%

numMarkers=8

SensorPairs = np.array([[2,5],[1,9],[5,6],[6,4],[9,3],[3,10]]) #Fiducial IDs for pairs of sensors arranged as

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

#load aruco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

#arUco Parameters
parameters = cv2.aruco.DetectorParameters()


# Create the ArUco detector
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

badFrames=0 #num frames where 8 fiducials are not found

SensorValues=[] #distances of the 6 sensors

while True:
    ret, frame = cap.read()

    # If frame is not read correctly, break the loop
    if not ret:
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 10)

    # Detect the markers
    corners, ids, rejected = detector.detectMarkers(gray)

    #draw on detected markers
    frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    #create list to store fiducial locations. made 11 long so i can just use ID as the index
    center_x = [0]*11
    center_y = [0]*11

    if len(ids) == numMarkers:

        for i in range(numMarkers):
            ThisCorner = corners[i]

            center_x[ids[i][0]] = int(np.mean(ThisCorner[0][:, 0]))
            center_y[ids[i][0]] = int(np.mean(ThisCorner[0][:, 1]))

            # Draw a circle at the center
            cv2.circle(frame, (center_x[ids[i][0]], center_y[ids[i][0]]), 5, (0, 0, 255), -1)
            # cv2.putText(frame, str(ids[i]), (center_x, center_y),cv2.FONT_HERSHEY_SIMPLEX,
            #                     1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        badFrames=badFrames+1

    thisFramesSensorValues=[0]*len(SensorPairs)
    for i in range(len(SensorPairs)):
        x1=center_x[SensorPairs[i][0]]
        x2=center_x[SensorPairs[i][1]]
        y1=center_y[SensorPairs[i][0]]
        y2=center_y[SensorPairs[i][1]]
        thisFramesSensorValues[i] =math.sqrt((x1-x2)**2+(y1-y2)**2)

    SensorValues.append(thisFramesSensorValues)

    new_width = int(frame.shape[1] * scale_x)
    new_height = int(frame.shape[0] * scale_y)
    new_size = (new_width, new_height)

    frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()


print(badFrames)