import cv2
import time
import numpy as np
import math
import platform
from collections import deque

from base_modules import HandDetector
from CONSTANTS import TEXT_COLOR, DISTANCE_COLOR, NULL_COLOR

# OS detection
IS_MAC = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

# Volume control setup
if IS_WINDOWS:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
    minVol, maxVol, _ = volume_interface.GetVolumeRange()

    def set_volume(percent):
        vol = np.interp(percent, [0, 100], [minVol, maxVol])
        volume_interface.SetMasterVolumeLevel(vol, None)

elif IS_MAC:
    import subprocess

    def set_volume(percent):
        percent = max(0, min(100, int(percent)))
        subprocess.run(["osascript", "-e", f"set volume output volume {percent}"])

else:
    def set_volume(percent):
        print("⚠ Volume control not supported on this OS.")


# Constants
VOLUME_BAR_X, VOLUME_BAR_Y = 50, 150
VOLUME_BAR_HEIGHT = 300
CAM_WIDTH, CAM_HEIGHT = 1280, 720
MUTE_THRESHOLD = 30
MAX_DISTANCE = 250
SMOOTHING_WINDOW = 5
VOLUME_TOLERANCE = 1  # Only set volume if change is significant

# Camera and detection setup
cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)
pTime = 0

detector = HandDetector()
volume_history = deque(maxlen=SMOOTHING_WINDOW)
last_set_volume = -1  # Init outside valid range

print("✅ Gesture-based volume control started")

while True:
    success, img = cap.read()
    detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    is_muted = False
    volume_percent = last_set_volume

    if len(lmList) != 0:
        # Thumb tip and index tip
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)

        if length <= MUTE_THRESHOLD:
            is_muted = True
            if last_set_volume != 0:
                set_volume(0)
                last_set_volume = 0
            cv2.circle(img, (cx, cy), 15, NULL_COLOR, cv2.FILLED)
        else:
            mapped_volume = np.interp(length, [MUTE_THRESHOLD, MAX_DISTANCE], [0, 100])
            volume_history.append(mapped_volume)
            smoothed_volume = int(np.mean(volume_history))

            if abs(smoothed_volume - last_set_volume) >= VOLUME_TOLERANCE:
                set_volume(smoothed_volume)
                last_set_volume = smoothed_volume

            volume_percent = smoothed_volume

        # Draw hand landmarks
        cv2.circle(img, (x1, y1), 15, DISTANCE_COLOR, cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, DISTANCE_COLOR, cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), DISTANCE_COLOR, 4)
        cv2.circle(img, (cx, cy), 15, DISTANCE_COLOR, cv2.FILLED)

    # Draw volume bar
    filled_height = int((volume_percent / 100) * VOLUME_BAR_HEIGHT)
    cv2.rectangle(img, (VOLUME_BAR_X, VOLUME_BAR_Y), (VOLUME_BAR_X + 50, VOLUME_BAR_Y + VOLUME_BAR_HEIGHT), (200, 200, 200), 2)

    if not is_muted:
        cv2.rectangle(
            img,
            (VOLUME_BAR_X, VOLUME_BAR_Y + VOLUME_BAR_HEIGHT - filled_height),
            (VOLUME_BAR_X + 50, VOLUME_BAR_Y + VOLUME_BAR_HEIGHT),
            DISTANCE_COLOR,
            cv2.FILLED
        )
        cv2.putText(img, f"{volume_percent}%", (VOLUME_BAR_X, VOLUME_BAR_Y - 20), cv2.FONT_HERSHEY_PLAIN, 2, TEXT_COLOR, 2)
    else:
        cv2.putText(img, "MUTED", (VOLUME_BAR_X, VOLUME_BAR_Y + VOLUME_BAR_HEIGHT + 40), cv2.FONT_HERSHEY_PLAIN, 2, NULL_COLOR, 2)

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, TEXT_COLOR, 3)

    cv2.imshow('Img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()