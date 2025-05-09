import mediapipe as mp
import time
import cv2
from constants import TEXT_COLOR, CONNECTION_COLOR, LANDMARK_COLOR


class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.8,trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                self.mpDraw.draw_landmarks(img, handLms, self.mphands.HAND_CONNECTIONS,
                                      self.mp.solutions.drawing_utils.DrawingSpec(color=LANDMARK_COLOR, thickness=8),
                                     self.mp.solutions.drawing_utils.DrawingSpec(color=CONNECTION_COLOR, thickness=4))

        return img

                #for id, lm in enumerate(handLms.landmark):
                    # print(id,lm)
                    #h, w, c = img.shape
                    #cx, cy = int(lm.x * w), int(lm.y * h)
                    #print(id, cx, cy)
                    # if id==0:
                    # cv2.circle(img, (cx, cy), 25, TEXT_COLOR, cv2.FILLED)


def main()
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        detector.findHands(img)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,TEXT_COLOR,3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)


if __name__ == "__main__":
    main()

