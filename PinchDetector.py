from HandT import hand_detector
import mediapipe as mp
import numpy as np
import osascript as osa
import cv2
import time

 

def tracking_points(img, point_list, h_range, v_range):
    thumb, index = point_list[4], point_list[8]

    cx = (thumb.x + index.x) // 2
    cy = (thumb.y + index.y) // 2

    #Finger points
    cv2.circle(img, (thumb.x, thumb.y), 7, (255, 255, 255), cv2.FILLED)
    cv2.circle(img, (index.x, index.y), 7, (255, 255, 255), cv2.FILLED)

    #Secant line
    cv2.circle(img, (cx, cy), 15, (255, 255, 255), cv2.FILLED)
    cv2.line(img, (thumb.x, thumb.y), (index.x, index.y), (255, 0, 255), 5)

    #Secant line -> volume math
    length_sec = np.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)

    if length_sec < 50:
        cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)


    # Using osascript for changing *mac* volume, it is very slow as the thread waits for the script to execute (lots of frame loss)
    """
    vol = int(np.interp(length_sec, [h_range[0], h_range[1]], [v_range[0], v_range[1]]))
    cv2.putText(img, str(vol), (0, 160), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    osa.osascript("set volume output volume " + str(vol)) # Super slow; python waits for the script to execute which causes a lot of frame loss
    """

def main():
    cap = cv2.VideoCapture(0)
    time.sleep(2)

    detector = hand_detector()

    preTime = 0
    curTime = 0

    h_range = [50, 300] # Hand Range
    v_range = [0, 100] # Volume Range

    while cap.isOpened():
        success, img = cap.read()

        if success:

            img = detector.detect_hands(img)
            lm_list = detector.detect_position(img, draw=False)

            if lm_list:
                tracking_points(img, lm_list, h_range, v_range)

            curTime = time.time()
            fps = 1 / (curTime - preTime)
            preTime = curTime

            cv2.putText(img, str(int(fps)), (0, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 8)
            
            cv2.imshow("Image", img)

            if cv2.waitKey(1) == 27:
                break
        else:
            print("Problem Reading")

if __name__ == "__main__":
    main()
