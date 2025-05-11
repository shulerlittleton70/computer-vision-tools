import time
import cv2
from base_modules import FaceMeshDetector
from constants import TEXT_COLOR

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector()

    pTime = 0

    while True:
        success, img = cap.read()
        img = detector.findFaceMesh(img)
        lmList = detector.findPosition(img)

        if lmList:
            print(f"Detected {len(lmList)} landmarks. Point 1: {lmList[1]}")

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

        cv2.imshow("Face Mesh Live", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()