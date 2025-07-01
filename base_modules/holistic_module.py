import cv2
import mediapipe as mp
from CONSTANTS import TEXT_COLOR, LANDMARK_COLOR, CONNECTION_COLOR


class HolisticDetector:
    def __init__(self, staticMode=False, modelComplexity=1, smoothLandmarks=True, detectionCon=0.5, trackingCon=0.5):
        self.staticMode = staticMode
        self.modelComplexity = modelComplexity
        self.smoothLandmarks = smoothLandmarks
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpHolistic = mp.solutions.holistic
        self.holistic = self.mpHolistic.Holistic(
            static_image_mode=self.staticMode,
            model_complexity=self.modelComplexity,
            smooth_landmarks=self.smoothLandmarks,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackingCon
        )
        self.mpDraw = mp.solutions.drawing_utils

        # All drawing specs come from constants
        self.landmarkSpec = self.mpDraw.DrawingSpec(color=LANDMARK_COLOR, thickness=1, circle_radius=1)
        self.connectionSpec = self.mpDraw.DrawingSpec(color=CONNECTION_COLOR, thickness=1)

    def findHolistic(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.holistic.process(imgRGB)

        if draw:
            if self.results.face_landmarks:
                self.mpDraw.draw_landmarks(
                    img,
                    self.results.face_landmarks,
                    self.mpHolistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.landmarkSpec,
                    connection_drawing_spec=self.connectionSpec
                )
            if self.results.left_hand_landmarks:
                self.mpDraw.draw_landmarks(
                    img,
                    self.results.left_hand_landmarks,
                    self.mpHolistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.landmarkSpec,
                    connection_drawing_spec=self.connectionSpec
                )
            if self.results.right_hand_landmarks:
                self.mpDraw.draw_landmarks(
                    img,
                    self.results.right_hand_landmarks,
                    self.mpHolistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.landmarkSpec,
                    connection_drawing_spec=self.connectionSpec
                )
            if self.results.pose_landmarks:
                self.mpDraw.draw_landmarks(
                    img,
                    self.results.pose_landmarks,
                    self.mpHolistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.landmarkSpec,
                    connection_drawing_spec=self.connectionSpec
                )

        return img

    def getLandmarks(self, img, draw=False):
        """
        Returns a dict with keys: face, pose, left_hand, right_hand
        Each maps to a list of [id, cx, cy]
        """
        landmark_data = {"face": [], "pose": [], "left_hand": [], "right_hand": []}
        h, w, _ = img.shape

        def extract_landmarks(results_attr, output_key):
            landmark_list = getattr(self.results, results_attr)
            if landmark_list:
                for i, lm in enumerate(landmark_list.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_data[output_key].append([i, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 2, TEXT_COLOR, cv2.FILLED)

        extract_landmarks("face_landmarks", "face")
        extract_landmarks("pose_landmarks", "pose")
        extract_landmarks("left_hand_landmarks", "left_hand")
        extract_landmarks("right_hand_landmarks", "right_hand")

        return landmark_data


def main():
    import time
    cap = cv2.VideoCapture(0)
    detector = HolisticDetector()
    pTime = 0

    while True:
        success, img = cap.read()
        img = detector.findHolistic(img)
        landmarks = detector.getLandmarks(img)

        if landmarks["pose"]:
            print(f"Pose Nose: {landmarks['pose'][0]}")  # ID 0 = nose in pose

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, TEXT_COLOR, 3)

        cv2.imshow("Holistic View", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()