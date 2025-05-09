import mediapipe as mp
import time
import cv2
from constants import TEXT_COLOR, CONNECTION_COLOR, LANDMARK_COLOR


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.8, trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackingCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        handLms,
                        self.mphands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=LANDMARK_COLOR, thickness=8),
                        self.mpDraw.DrawingSpec(color=CONNECTION_COLOR, thickness=4)
                    )
        return img

        # Optional: Landmark positions (uncomment if needed)
        # for id, lm in enumerate(handLms.landmark):
        #     h, w, c = img.shape
        #     cx, cy = int(lm.x * w), int(lm.y * h)
        #     if id == 0:
        #         cv2.circle(img, (cx, cy), 25, TEXT_COLOR, cv2.FILLED)


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
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

        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()