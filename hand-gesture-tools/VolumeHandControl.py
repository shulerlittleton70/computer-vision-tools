#control the volume of your comnpute via hand gesture

import cv2
import time
import numpy as np
from base_modules import HandDetector
from constants import TEXT_COLOR

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

