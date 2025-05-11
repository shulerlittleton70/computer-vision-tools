import mediapipe as mp
import time
import cv2
from constants import TEXT_COLOR, CONNECTION_COLOR, LANDMARK_COLOR


class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=1, detectionCon=0.8, trackingCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackingCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(color=LANDMARK_COLOR, thickness=5, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        faceLms,
                        self.mpFaceMesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.drawSpec,
                        connection_drawing_spec=self.mpDraw.DrawingSpec(color=CONNECTION_COLOR, thickness=1)
                    )
        return img

    def findPosition(self, img, faceNo=0, draw=False):
        lmList = []
        if self.results.multi_face_landmarks:
            myFace = self.results.multi_face_landmarks[faceNo]
            h, w, _ = img.shape
            for id, lm in enumerate(myFace.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 2, TEXT_COLOR, cv2.FILLED)
        return lmList


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img = detector.findFaceMesh(img)
        lmList = detector.findPosition(img)

        if lmList:
            print(f"Point 1: {lmList[1]}")  # You can print other points by ID

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
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

        cv2.imshow("FaceMesh", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()