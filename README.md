# Computer Vision Tools with MediaPipe

This repository is a modular, cross-platform toolkit for real-time computer vision applications using [MediaPipe](https://google.github.io/mediapipe/). Built entirely in Python with OpenCV integration, it supports hand tracking, face mesh detection, pose estimation, and holistic body tracking — all organized in reusable class-based modules.

---

## 🔍 Why I Built This

This project began as a personal exploration into computer vision, gesture recognition, and real-time desktop interaction. Inspired by the responsiveness of gesture-controlled interfaces, I developed a series of clean, modular tools that interpret human posture and hand or face movement using MediaPipe — then apply those inputs to change system state, like audio or screen brightness.

---

## 🧠 What Is MediaPipe?

[MediaPipe](https://google.github.io/mediapipe/) is a framework developed by Google for building multimodal (video/audio) perception pipelines. It's commonly used for tasks such as:
- Hand, pose, and face tracking
- Gesture recognition
- Augmented reality filters
- Background segmentation

MediaPipe provides highly efficient, pre-trained models that run in real time on CPU and GPU — making it ideal for interactive applications.

---

## 🗂 Project Structure

### 🔧 `base_modules/`
Reusable, class-based MediaPipe modules for various trackers.

| File                        | Description |
|-----------------------------|-------------|
| `handtrackingmodule.py`     | Detects 21 hand landmarks and draws hand skeleton |
| `facemeshmodule.py`         | Detects 468 face landmarks using FaceMesh |
| `facedetectionmodule.py`    | Lightweight bounding box face detector |
| `poseTrackingModule.py`     | Detects 33 body landmarks for full-body tracking |
| `holisticModule.py`         | All-in-one detector combining face, pose, and hands |

---

### 🎨 `constants.py`
Defines shared drawing colors for landmarks, connections, and text overlays.

---

### 🛠 Tooling Scripts

#### `hand-gesture-tools/VolumeHandControl.py`
Uses `HandDetector` to:
- Detect distance between thumb tip and index finger
- Maps this to system volume level (0–100%)
- Mutes if distance is very small

Supports macOS (via `osascript`) and Windows (via `pycaw`).

---

#### `face-tracking-tools/face-tilt-brightness.py`
Uses `FaceMeshDetector` to:
- Calculate roll angle of face based on eye landmarks
- Maps head tilt left/right to brightness percentage
- Sets system brightness (macOS: `brightness`, Windows: `screen_brightness_control`)
- Includes calibration and a visual brightness bar

---

### 📁 `Call Examples/`
Basic exploratory scripts for face, hand, pose, and holistic tracking to demonstrate individual module outputs.

---

## 🧩 Functions and Classes

### 📦 `base_modules/handtrackingmodule.py`
- `class HandDetector`
  - `findHands(img, draw=True)`
  - `findPosition(img, handNo=0, draw=True)`

### 📦 `base_modules/facemeshmodule.py`
- `class FaceMeshDetector`
  - `findFaceMesh(img, draw=True)`
  - `findPosition(img, faceNo=0, draw=False)`

### 📦 `base_modules/facedetectionmodule.py`
- `class FaceDetector`
  - `findFaces(img, draw=True)`
  - `findBoundingBox(img)`

### 📦 `base_modules/poseTrackingModule.py`
- `class PoseDetector`
  - `findPose(img, draw=True)`
  - `findPosition(img, draw=False)`

### 📦 `base_modules/holisticModule.py`
- `class HolisticDetector`
  - `findHolistic(img, draw=True)`
  - `getLandmarks(img, draw=False)`

---

## 🛠 Requirements

Install dependencies using:
```bash
pip install -r requirements.txt