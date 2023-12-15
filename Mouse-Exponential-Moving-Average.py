import cv2
from cvzone.HandTrackingModule import HandDetector
import mouse
import numpy as np
import pyautogui
from PIL import Image
import time

def calcAngles(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Capture the video
cap = cv2.VideoCapture(0)

#cam_w, cam_h = 640, 480
cam_w, cam_h = 1280, 720
cap.set(3, cam_w)
cap.set(4, cam_h)

screenWidth, screenHeight = 1920, 1080

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


hand_placement_duration = 5  # Hand placement duration in seconds
start_time = None
hand_placed = False

# Smoothing parameters
alpha = 0.8  # Smoothing factor
prev_x, prev_y = 0, 0

# Scroll and Mouse Mode
isMouseMode = True

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img,flipType=True)
    
    if not hand_placed:
        handDetectionRectangle = cv2.rectangle(img, (0,0), (cam_w,cam_h//16), (0,0,255), -1)
        cv2.putText(handDetectionRectangle, "TRACKING NO HAND. HAND DETECTED WILL MOVE THE MOUSE.", (cam_w//24,cam_h//24), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0),3)
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
            cv2.rectangle(img, (cam_w//2 - 110, cam_h - 100), (cam_w//2 + 210, cam_h + 150), (0,0,0), -1)
            cv2.rectangle(img, (cam_w//2 - 110, cam_h - 100), (cam_w//2 + 210, cam_h + 150), (255,255,255), 2)
            cv2.putText(img, f"Place hand for {int(remaining_time)}s", (cam_w//2 - 100, cam_h - 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        else:
            start_time = None

    if hand_placed:
        if hands and isMouseMode:
            if hands[0]['type'] == "Left":
                FRAME_START_X_OFFSET = cam_w//4
                FRAME_START_X = cam_w//2 + FRAME_START_X_OFFSET - 30
                FRAME_END_X_OFFSET = -cam_w//16
                FRAME_END_X = cam_w + FRAME_END_X_OFFSET - 30

            elif hands[0]['type'] == "Right":
                FRAME_START_X_OFFSET = 70
                FRAME_START_X = cam_w//16 + FRAME_START_X_OFFSET
                FRAME_END_X = cam_w//4 + FRAME_START_X_OFFSET
            
            cv2.rectangle(img, (FRAME_START_X, FRAME_START_Y), (FRAME_END_X, FRAME_END_Y), (255, 0, 255), 4)
            cv2.putText(img, "Screen Mapping", (FRAME_START_X, FRAME_START_Y - 10),cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,255), 1)
            ANCHOR_POINT_X_START = FRAME_START_X
            ANCHOR_POINT_X_END = FRAME_END_X
            ANCHOR_POINT_Y_START = FRAME_START_Y
            ANCHOR_POINT_Y_END = FRAME_END_Y
            # Get the values for the hands.
            lmlist = hands[0]['lmList']

            # Get the x and y coordinates of the wrist.
            ind_x, ind_y = lmlist[0][0], lmlist[0][1]

            # Apply smoothing using exponential moving average (EMA)
            smoothed_x = alpha * prev_x + (1 - alpha) * ind_x
            smoothed_y = alpha * prev_y + (1 - alpha) * ind_y

            prev_x, prev_y = smoothed_x, smoothed_y

            # Draw a circle on the index finger using the smoothed coordinates.
            cv2.circle(img, (int(smoothed_x), int(smoothed_y)), 5, (0, 255, 255), 2)

            # Convert the smoothed coordinates to the screen resolution.
            conv_x = int(np.interp(smoothed_x, (ANCHOR_POINT_X_START, ANCHOR_POINT_X_END), (0, 1920)))
            conv_y = int(np.interp(smoothed_y, (ANCHOR_POINT_Y_START, ANCHOR_POINT_Y_END), (0, 1080)))

            # Move mouse.
            mouse.move(conv_x, conv_y)

            # Create the fingers object.
            fingers = detector.fingersUp(hands[0])


            LEFT_CLICK_DISTANCE_THRESHOLD = 25
            indexThumbDistance = detector.findDistance(lmlist[4][:2], lmlist[8][:2])

            RIGHT_CLICK_DISTANCE_THRESHOLD = 25
            middleThumbDistance = detector.findDistance(lmlist[4][:2], lmlist[12][:2])

            DOUBLE_CLICK_ANGLE_THRESHOLD = 11
            indexFingerTip = lmlist[8]
            indexFingerBase = lmlist[5]
            middleFingerTip = lmlist[12]
            doubleClickAngle = calcAngles(indexFingerTip,indexFingerBase,middleFingerTip)

            # FOR LEFT CLICK
            if indexThumbDistance[0] < LEFT_CLICK_DISTANCE_THRESHOLD:
                pyautogui.leftClick()
            # FOR RIGHT CLICK
            elif middleThumbDistance[0] < RIGHT_CLICK_DISTANCE_THRESHOLD:
                pyautogui.rightClick()
            # FOR DOUBLE LEFT CLICK
            elif doubleClickAngle < DOUBLE_CLICK_ANGLE_THRESHOLD:
                pyautogui.doubleClick()
            elif not fingers[4] and not fingers[3] and not fingers[2] and not fingers[1]:
                print("Scroll Mode")
                isMouseMode = False

            # 4 = pinky, 3 = ring, 2 = middle, 1 = index, 0 = thumb
        elif hands and isMouseMode == False:
            fingers = detector.fingersUp(hands[0])
            if fingers[0]:
                mouse.wheel(delta=1)
            elif fingers[1]:
                mouse.wheel(delta=-1)
            if all(detector.fingersUp(hands[0])):
                isMouseMode = True

        else:
            hand_placed = False
    cv2.imshow("Camera Feed", img)
    cv2.waitKey(1)
