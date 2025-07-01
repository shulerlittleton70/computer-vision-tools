import cv2
import time
from base_modules import HolisticDetector
from CONSTANTS import TEXT_COLOR

def main():
    cap = cv2.VideoCapture(0)
    detector = HolisticDetector()

    pTime = 0

    while True:
        success, img = cap.read()

        # Run holistic detection
        img = detector.findHolistic(img)
        landmarks = detector.getLandmarks(img)

        # Debug output
        if landmarks["face"]:
            print(f"Face Point 1: {landmarks['face'][1]}")
        if landmarks["pose"]:
            print(f"Nose: {landmarks['pose'][0]}")
        if landmarks["left_hand"]:
            print(f"Left hand thumb tip: {landmarks['left_hand'][4]}")
        if landmarks["right_hand"]:
            print(f"Right hand thumb tip: {landmarks['right_hand'][4]}")

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, TEXT_COLOR, 3)

        cv2.imshow("Holistic Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()