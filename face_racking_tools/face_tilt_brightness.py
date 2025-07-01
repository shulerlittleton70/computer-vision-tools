import cv2
import time
import platform
import subprocess
import shutil
import math
import numpy as np
from base_modules.face_mesh_module import FaceMeshDetector
from CONSTANTS import TEXT_COLOR

IS_MAC = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

def set_brightness(percent):
    percent = max(0, min(100, int(percent)))

    if IS_MAC:
        if shutil.which("brightness"):
            try:
                subprocess.run(["brightness", str(percent / 100)], check=True)
            except subprocess.CalledProcessError:
                print("⚠ Tried to set brightness but it failed.")
        else:
            print("⚠ `brightness` command not found. Install with: brew install brightness")
    elif IS_WINDOWS:
        try:
            import screen_brightness_control as sbc
            sbc.set_brightness(percent)
        except ImportError:
            print("⚠ Install `screen_brightness_control` for Windows support.")
        except Exception as e:
            print(f"⚠ Windows brightness control failed: {e}")
    else:
        print("⚠ Brightness control not supported on this OS.")

def calculate_face_angle(lx, ly, rx, ry):
    dx = rx - lx
    dy = ry - ly
    radians = math.atan2(dy, dx)
    return math.degrees(radians)

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector()
    pTime = 0
    last_brightness = -1

    # Calibration
    baseline_angle = None
    calibration_frames = 30
    calibration_data = []

    print("🛠 Calibrating... Hold your head neutral and face the camera.")

    while True:
        success, img = cap.read()
        img = detector.findFaceMesh(img)
        lmList = detector.findPosition(img)

        if len(lmList) >= 468:
            _, lx, ly = lmList[33]   # Left eye
            _, rx, ry = lmList[263]  # Right eye

            angle = calculate_face_angle(lx, ly, rx, ry)

            if len(calibration_data) < calibration_frames:
                calibration_data.append(angle)
                cv2.putText(img, "Calibrating...", (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, TEXT_COLOR, 2)
            else:
                if baseline_angle is None:
                    baseline_angle = sum(calibration_data) / len(calibration_data)
                    print(f"✅ Calibration complete: baseline angle = {baseline_angle:.2f}°")

                # Relative tilt
                tilt = angle - baseline_angle
                brightness = np.interp(tilt, [-25, 25], [0, 100])  # full range
                brightness = int(brightness)

                if abs(brightness - last_brightness) >= 3:
                    set_brightness(brightness)
                    print(f"🌓 Tilt: {tilt:.2f}° → Brightness: {brightness}%")
                    last_brightness = brightness

                # Brightness bar
                bar_x, bar_y = 50, 150
                bar_height = 300
                filled_height = int((brightness / 100) * bar_height)
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + 50, bar_y + bar_height), (200, 200, 200), 2)
                cv2.rectangle(img, (bar_x, bar_y + bar_height - filled_height),
                              (bar_x + 50, bar_y + bar_height), TEXT_COLOR, cv2.FILLED)
                cv2.putText(img, f"{brightness}%", (bar_x, bar_y - 20), cv2.FONT_HERSHEY_PLAIN, 2, TEXT_COLOR, 2)

                # Debug: angle
                cv2.putText(img, f"Tilt: {tilt:.1f} deg", (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, TEXT_COLOR, 2)

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, TEXT_COLOR, 3)

        cv2.imshow("Face Tilt Brightness Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()