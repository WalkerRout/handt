
import cv2
import mediapipe as mp
import time
import numpy as np


class position():
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y


class hand_detector():
    def __init__(self, mode=False, num_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.num_hands = num_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(mode, num_hands, detection_confidence, tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils


    def detect_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)

        return img


    def detect_position(self, img, draw=True, hand_num=0):
        lm_list = []
        
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_num]

            for id, lm in enumerate(hand.landmark):
                height, width, colour_depth = img.shape
                center_x, center_y = int(lm.x * width), int(lm.y * height)
                lm_list.append(position(id, center_x, center_y))

                if draw:
                    cv2.circle(img, (center_x, center_y), 10, (255, 0, 255), cv2.FILLED)

            return lm_list


def main():
    preTime = 0
    curTime = 0

    cap = cv2.VideoCapture(0)
    
    detector = hand_detector()

    while cap.isOpened():
        success, img = cap.read()

        if success:
            img = detector.detect_hands(img)
            lm_list = detector.detect_position(img)

            if lm_list:
                thumb = lm_list[4]
                index = lm_list[8]

            curTime = time.time()
            fps = 1 / (curTime - preTime)
            preTime = curTime

            cv2.putText(img, str(int(fps)), (0, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 8)

            cv2.imshow("Image", img)

            if cv2.waitKey(1) == 27: #waits for q key
                break
                
        else:
            print("Problem Reading!")


if __name__ == "__main__":
    main()