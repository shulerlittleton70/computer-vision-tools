import time
import cv2

from base_modules import FaceMeshDetector
from base_modules.handtrackingmodule import HandDetector
from constants import TEXT_COLOR

def main():
    cap = cv2.VideoCapture(0)
    face_detector = FaceMeshDetector()
    hand_detector = HandDetector()

    pTime = 0

    while True:
        success, img = cap.read()

        # Detect face landmarks
        img = face_detector.findFaceMesh(img)
        face_lms = face_detector.findPosition(img, draw=False)

        # Detect hand landmarks
        img = hand_detector.findHands(img)
        hand_lms = hand_detector.findPosition(img, draw=False)

        # Optional: print one landmark from each for debug
        if face_lms:
            print(f"Face landmark 1: {face_lms[1]}")
        if hand_lms:
            print(f"Hand landmark 4: {hand_lms[4]}")

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime

        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (10, 70),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            TEXT_COLOR,
            3
        )

        cv2.imshow("Face + Hands", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()