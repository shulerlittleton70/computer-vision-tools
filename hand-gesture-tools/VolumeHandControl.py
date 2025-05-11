#control the volume of your comnpute via hand gesture

import cv2
import time
import numpy as np
from base_modules import HandDetector
from constants import TEXT_COLOR, CONNECTION_COLOR, LANDMARK_COLOR

#####################

wCam, hCam = 1280, 720

#####################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = HandDetector()



while True:
    success, img = cap.read()
    detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[4],lmList[4])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cv2.circle(img, (x1, y1), 15, LANDMARK_COLOR, cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, LANDMARK_COLOR, cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), CONNECTION_COLOR, cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(
        img,
        str(int(fps)),
        (10, 70),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        TEXT_COLOR,
        3
    )

    cv2.imshow('Img', img)
    cv2.waitKey(1)

