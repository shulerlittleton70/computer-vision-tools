import cv2
import time
from base_modules import PoseDetector
from constants import TEXT_COLOR

def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    pTime = 0

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)

        if lmList:
            print(f"Nose: {lmList[0]}")  # ID 0 is the nose landmark

        # FPS calculation
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

        cv2.imshow("Pose Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()