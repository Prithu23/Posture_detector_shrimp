# Are You Shrimpin'? 🦐

A posture detection web app that uses your webcam to catch you shrimping — entirely in the browser using MediaPipe.

## Usage

```bash
cd Posture_detector_shrimp
python3 -m http.server 8000
```

Open `http://localhost:8000/posture.html` in Chrome. Click START, sit straight for 3 seconds to calibrate, and let it do its thing.

If you shrimp, it will honk at you until you fix your posture.

## Python Version

```bash
pip install opencv-python mediapipe numpy
python posture_detector.py
```
