import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import csv

video_path = 'Videos/MVI_0767.MOV'  # Replace with the actual path to your video file
cap = cv2.VideoCapture(video_path)

scale_x = .8  # Reduce width to 50%
scale_y = .8  # Reduce height to 50%
# print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


numMarkers=8 #num fiducials

SensorPairs = np.array([[2,5],[1,9],[5,6],[6,4],[9,3],[3,10]]) #Fiducial IDs for pairs of sensors arranged as

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #number of video frames

#load aruco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

#arUco Parameters
parameters = cv2.aruco.DetectorParameters()

# Create the ArUco detector
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

#counters
NumBadFrames=0 #num frames where 8 fiducials are not found
badFrames=[]
FrameNum=0 #current video frame

SensorValues=np.zeros((len(SensorPairs),frame_count)) #distances of the 6 sensors


#specify color HSV bounds for the spine
# lower boundary RED color range values; Hue (0 - 10)
lower1 = np.array([0, 30, 120])
upper1 = np.array([20, 255, 255])

# upper boundary RED color range values; Hue (160 - 180)
lower2 = np.array([160, 30, 120])
upper2 = np.array([179, 255, 255])

#green color bounds
GreenLower = np.array([50, 125, 130])
GreenUpper = np.array([95, 180, 255])


def fit_polynomial_curve(ThisMask, degree):
    x=[]
    y=[]
    for i in range(ThisMask.shape[0]):
        for j in range(mask.shape[1]):

            if ThisMask[i,j] != 0:
                x.append(j)
                y.append(i)

    # Fit a polynomial (degree can be adjusted)
    poly_coeffs = np.polyfit(x, y, degree)
    poly_func = np.poly1d(poly_coeffs)  # Create a polynomial function
    print(np.max(x))
    # Generate smooth curve points
    x_smooth = np.linspace(np.min(x), np.max(x), 100)
    y_smooth = poly_func(x_smooth)

    return np.column_stack((x_smooth, y_smooth)), poly_coeffs   # Return smoothed centerline points

output_csv = "StoredData.csv"

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ["FrameNum"] + [f"Sensor_{i+1}" for i in range(len(SensorPairs))] + [f"Coeff_{i}" for i in range(6)]
    writer.writerow(header)
    while True:
        ret, frame = cap.read()
        # If frame is not read correctly, break the loop
        if not ret:
            break

        frame[0:200, 1195:1400]= [0,0,0]
        frame = frame[0:1960, 500:1400 ]

        ###################################################################################################################
        #Process Fiducials
        ###################################################################################################################
        #make frame grayscale and apply gaussian blur to improve reading
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 10)

        # Detect the markers
        corners, ids, rejected = detector.detectMarkers(gray)

        #create list to store fiducial locations. made 11 long so i can just use ID as the index
        center_x = [0]*11
        center_y = [0]*11

        if len(ids) == numMarkers:
            #find center of the fiducials
            for i in range(numMarkers):
                ThisCorner = corners[i]
                center_x[ids[i][0]] = int(np.mean(ThisCorner[0][:, 0]))
                center_y[ids[i][0]] = int(np.mean(ThisCorner[0][:, 1]))
                # Draw a circle at the center
                cv2.circle(frame, (center_x[ids[i][0]], center_y[ids[i][0]]), 5, (0, 0, 0), -1)
        else:
            NumBadFrames=NumBadFrames+1
            badFrames.append(FrameNum)

        #calculate the distances between the fiducials
        thisFramesSensorValues=np.zeros((1,len(SensorPairs)))
        for i in range(len(SensorPairs)):
            x1=center_x[SensorPairs[i][0]]
            x2=center_x[SensorPairs[i][1]]
            y1=center_y[SensorPairs[i][0]]
            y2=center_y[SensorPairs[i][1]]
            thisFramesSensorValues[0,i] =math.sqrt((x1-x2)**2+(y1-y2)**2)
        SensorValues[:, FrameNum] = thisFramesSensorValues

        ###################################################################################################################
        # Process Spine
        ###################################################################################################################
        frame = cv2.transpose(frame)
        #get hsv colors of the frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 10)
        #find color masks
        #red
        # lower_mask = cv2.inRange(hsv, lower1, upper1)
        # upper_mask = cv2.inRange(hsv, lower2, upper2)
        # mask = lower_mask + upper_mask
        mask = cv2.inRange(hsv, GreenLower, GreenUpper)
        # define kernel size
        kernel = np.ones((7, 7), np.uint8)

        # Remove unnecessary noise from mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


        # finds contours from colors
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if np.any(mask) and len(ids) == numMarkers:
            try:
                degree= 5
                centerline, poly_coeffs = fit_polynomial_curve(mask.copy(), degree)
                writer.writerow([FrameNum] + list(thisFramesSensorValues[0]) + list(poly_coeffs))
            except:
                print('error fitting line/ finding markers/ writing to csv at frame: ', FrameNum )
        else:
            print('error fitting line/ finding markers/ writing to csv at frame: ', FrameNum)



        # array of center points of contours
        C = np.empty([len(contours), 2], 'i')

        # Draw contour on original image
        output = cv2.drawContours(frame, contours, -1, (0, 255, 0  ), 2)

        if len(contours) > 0:
            # largest_contour = max(contours, key=cv2.contourArea)

            # Fit a polynomial curve to the contour
            # centerline = fit_polynomial_curve(contours, degree=3)

            for i in range(len(contours) ):
                # pt1 = (int(centerline[i][0]), int(centerline[i][1]))
                # pt2 = (int(centerline[i + 1][0]), int(centerline[i + 1][1]))
                # cv2.line(frame, pt1, pt2, (255, 0, 0), 2)  # Blue centerline

                M = cv2.moments(contours[i])
                C[i, 0] = int(M['m10'] / M['m00'])  # cx
                C[i, 1] = int(M['m01'] / M['m00'])  # cy
                output[C[i, 1] - 2:C[i, 1] + 2, C[i, 0] - 2:C[i, 0] + 2] = [255, 255, 255]

            for i in range(len(centerline)-1):
                pt1 = (int(centerline[i][0]), int(centerline[i][1]))
                pt2 = (int(centerline[i + 1][0]), int(centerline[i + 1][1]))
                cv2.line(frame, pt1, pt2, (255, 0, 255), 2)  # Blue centerline

        frame = cv2.transpose(frame)
        ###################################################################################################################
        # Output video
        ###################################################################################################################
        # draw on detected markers
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # resize frame to fit on my screen
        new_width = int(frame.shape[1]* scale_x)
        new_height = int(frame.shape[0] * scale_y)
        new_size = (new_width, new_height)
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

        cv2.imshow('Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        FrameNum = FrameNum+1

    # Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

#remove frame where we do not have all fiducials in view
SensorValues= np.delete(SensorValues,badFrames,1)

for i in range(len(SensorPairs)):
    plt.plot(SensorValues[i,:])
plt.show()

print(NumBadFrames)