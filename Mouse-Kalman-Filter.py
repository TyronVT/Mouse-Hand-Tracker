import cv2
from cvzone.HandTrackingModule import HandDetector
import mouse
import numpy as np
import pyautogui
from PIL import Image
import time

# Capture the video
cap = cv2.VideoCapture(0)

#cam_w, cam_h = 640, 480
cam_w, cam_h = 1280, 720
cap.set(3, cam_w)
cap.set(4, cam_h)

frameR = 180


#FRAME_START_X_OFFSET = -cam_w//8
#FRAME_START_Y_OFFSET = cam_h//4
#FRAME_START_X = cam_w//2 + FRAME_START_X_OFFSET 
#FRAME_START_Y = cam_h//2 + FRAME_START_Y_OFFSET
#
#FRAME_END_X_OFFSET = cam_w//8
#FRAME_END_Y_OFFSET = cam_h//2 - 10
#FRAME_END_X = cam_w//2 + FRAME_END_X_OFFSET
#FRAME_END_Y = cam_h//2 + FRAME_END_Y_OFFSET


FRAME_START_X_OFFSET = cam_w//4
FRAME_START_Y_OFFSET = cam_h//4
FRAME_START_X = cam_w//2 + FRAME_START_X_OFFSET - 30
FRAME_START_Y = cam_h//2 + FRAME_START_Y_OFFSET

FRAME_END_X_OFFSET = -cam_w//16
FRAME_END_Y_OFFSET = cam_h//2 - 10
FRAME_END_X = cam_w + FRAME_END_X_OFFSET - 30
FRAME_END_Y = cam_h//2 + FRAME_END_Y_OFFSET
print(FRAME_START_X, FRAME_START_Y)
print(FRAME_END_X, FRAME_END_Y)

detector = HandDetector(detectionCon=0.8, maxHands=1)

HAND_HEIGHT = 200
HAND_WIDTH = 200
handOverlay = cv2.resize(cv2.imread("resources/hand.png"), (HAND_WIDTH, HAND_HEIGHT))

img2gray = cv2.cvtColor(handOverlay, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

hand_placement_duration = 5  # Hand placement duration in seconds
start_time = None
hand_placed = False


# Create a Kalman filter object
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 0.03

# Smoothing parameters
alpha = 0.7  # Smoothing factor
prev_x, prev_y = 0, 0


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    # Create a rectangle that represents the screen.
    #cv2.rectangle(img, (frameR, frameR), (cam_w - frameR, cam_h - frameR), (255, 0, 255), 2)
    cv2.rectangle(img, (FRAME_START_X, FRAME_START_Y), (FRAME_END_X, FRAME_END_Y), (255, 0, 255), 2)
    cv2.circle(img,(FRAME_START_X, FRAME_START_Y),3,color=(255,255,255),thickness=2)
    cv2.circle(img,(FRAME_END_X, FRAME_END_Y),3,color=(255,255,255),thickness=2)
    if not hand_placed:
        if hands and all(detector.fingersUp(hands[0])):
            if start_time is None:
                start_time = time.time()

            elapsed_time = time.time() - start_time
            remaining_time = max(0, hand_placement_duration - elapsed_time)

            if elapsed_time >= hand_placement_duration:
                hand_placed = True
                start_time = None
                print("Hand placed!")

            # Draw countdown timer
            cv2.putText(img, f"Place hand for {int(remaining_time)}s", (0, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        else:
            start_time = None

    if hand_placed:
        if hands:
            # Get the values for the hands.
            lmlist = hands[0]['lmList']

            # Get the x and y coordinates of the wrist.
            ind_x, ind_y = lmlist[0][0], lmlist[0][1]

            # Apply Kalman filter to predict the state
            predicted_state = kalman.predict()

            # Update the measurement based on the observed coordinates
            measurement = np.array([[ind_x], [ind_y]], dtype=np.float32)
            corrected_state = kalman.correct(measurement)

            smoothed_x = corrected_state[0][0]
            smoothed_y = corrected_state[1][0]

            prev_x, prev_y = smoothed_x, smoothed_y

            # Draw a circle on the index finger using the smoothed coordinates.
            cv2.circle(img, (int(smoothed_x), int(smoothed_y)), 5, (0, 255, 255), 2)

            # Convert the smoothed coordinates to the screen resolution.
            conv_x = int(np.interp(smoothed_x, (FRAME_START_X, FRAME_END_X), (0, 1920)))
            conv_y = int(np.interp(smoothed_y, (FRAME_START_Y, FRAME_END_Y), (0, 1080)))

            # Move mouse.
            mouse.move(conv_x, conv_y)

            # Create the fingers object.
            fingers = detector.fingersUp(hands[0])

            indexThumbDistance = detector.findDistance(lmlist[4][:2], lmlist[8][:2])
            print(indexThumbDistance[0])

            LEFT_CLICK_DISTANCE_THRESHOLD = 25
            if indexThumbDistance[0] < LEFT_CLICK_DISTANCE_THRESHOLD:
                pyautogui.leftClick()
            # 4 = pinky, 3 = ring, 2 = middle, 1 = index, 0 = thumb

    cv2.imshow("Camera Feed", img)
    cv2.waitKey(1)