"""
Posture Detector - "Are You Shrimpin'?"
Real-time posture detection using OpenCV + MediaPipe.

Setup:
    pip install opencv-python mediapipe numpy

Usage:
    cd Posture_detector_shrimp
    python posture_detector.py

Controls:
    q - Quit
    c - Recalibrate
    1 - Strict sensitivity
    2 - Normal sensitivity (default)
    3 - Relaxed sensitivity
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import math
import subprocess
import threading
import asyncio
import json
import websockets

# ---------------------------------------------------------------------------
# WebSocket Server
# ---------------------------------------------------------------------------

_ws_clients = set()
_ws_loop = None

async def _ws_handler(websocket):
    _ws_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        _ws_clients.discard(websocket)

async def _ws_broadcast(data):
    if _ws_clients:
        msg = json.dumps(data)
        await asyncio.gather(*[c.send(msg) for c in list(_ws_clients)], return_exceptions=True)

def _run_ws_server():
    global _ws_loop
    _ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_ws_loop)
    async def _serve():
        async with websockets.serve(_ws_handler, "localhost", 8765):
            print("WebSocket server running on ws://localhost:8765")
            await asyncio.Future()
    _ws_loop.run_until_complete(_serve())

def broadcast(data):
    if _ws_loop and _ws_clients:
        asyncio.run_coroutine_threadsafe(_ws_broadcast(data), _ws_loop)


# ---------------------------------------------------------------------------
# Sound Alert
# ---------------------------------------------------------------------------

def play_alert_sound():
    """Play a non-blocking alert sound. macOS uses afplay; fallback is terminal bell."""
    def _play():
        try:
            subprocess.run(
                ["afplay", "/System/Library/Sounds/Sosumi.aiff"],
                check=False, timeout=3,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("\a", end="", flush=True)

    threading.Thread(target=_play, daemon=True).start()


# ---------------------------------------------------------------------------
# Landmark indices
# ---------------------------------------------------------------------------

# Eyes
LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER = 1, 2, 3
RIGHT_EYE_INNER, RIGHT_EYE, RIGHT_EYE_OUTER = 4, 5, 6
EYE_INDICES = [1, 2, 3, 4, 5, 6]

# Other upper-body landmarks
NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24

TRACKED_INDICES = [NOSE] + EYE_INDICES + [
    LEFT_EAR, RIGHT_EAR,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
]

# Connections to draw
SKELETON_CONNECTIONS = [
    (LEFT_EAR, LEFT_SHOULDER),
    (RIGHT_EAR, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (NOSE, LEFT_EAR),
    (NOSE, RIGHT_EAR),
    (LEFT_EYE, NOSE),
    (RIGHT_EYE, NOSE),
]


# ---------------------------------------------------------------------------
# Sensitivity presets  (entry_offset, exit_offset, required_frames)
# Offsets are added to the baseline ratio — larger ratio = more hunched
# ---------------------------------------------------------------------------

SENSITIVITY = {
    "strict":  (0.05, 0.02, 8),
    "normal":  (0.08, 0.04, 10),
    "relaxed": (0.12, 0.06, 15),
}


# ---------------------------------------------------------------------------
# PostureDetector
# ---------------------------------------------------------------------------

class PostureDetector:
    def __init__(self, sensitivity="normal"):
        # Calibration state
        self.calibrated = False
        self.calibration_data = []
        self.calibration_tilt_data = []
        self.calibration_wait_start = None
        self.baseline_ratio = None
        self.baseline_tilt = None

        # Thresholds (set after calibration)
        self.entry_threshold = 999.0
        self.exit_threshold = 999.0
        self.required_frames = 10

        # Detection state
        self.is_shrimping = False
        self.consecutive_bad = 0
        self.shrimp_count = 0
        self.bad_posture_start = None
        self.last_alert_time = 0.0

        # Timing
        self.session_start = None
        self.total_bad_seconds = 0.0

        self.set_sensitivity(sensitivity)

    # --- sensitivity ---

    def set_sensitivity(self, level):
        self.sensitivity = level
        entry_off, exit_off, req = SENSITIVITY[level]
        self.required_frames = req
        if self.baseline_ratio is not None:
            self.entry_threshold = self.baseline_ratio + entry_off
            self.exit_threshold = self.baseline_ratio + exit_off

    def reset_calibration(self):
        self.calibrated = False
        self.calibration_data = []
        self.calibration_tilt_data = []
        self.calibration_wait_start = None
        self.baseline_ratio = None
        self.baseline_tilt = None
        self.entry_threshold = 999.0
        self.exit_threshold = 999.0
        self.is_shrimping = False
        self.consecutive_bad = 0
        self.bad_posture_start = None

    # --- posture metrics ---

    def _calc_shrimp_ratio(self, lm):
        """eyesToNose / noseToShoulder — increases when hunching forward."""
        nose_y = lm[NOSE].y
        eye_avg_y = (lm[LEFT_EYE].y + lm[RIGHT_EYE].y) / 2.0
        shoulder_avg_y = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2.0
        eyes_to_nose = nose_y - eye_avg_y
        nose_to_shoulder = shoulder_avg_y - nose_y
        if nose_to_shoulder <= 0.001:
            return 999.0
        return eyes_to_nose / nose_to_shoulder

    def _calc_sideways_tilt(self, lm):
        """Absolute ear y-difference — increases when leaning sideways."""
        return abs(lm[LEFT_EAR].y - lm[RIGHT_EAR].y)

    # --- calibration ---

    CALIBRATION_WAIT = 3.0
    CALIBRATION_FRAMES = 45

    def calibrate_frame(self, landmarks):
        """Feed one frame during calibration. Returns True when done."""
        now = time.time()
        if self.calibration_wait_start is None:
            self.calibration_wait_start = now

        elapsed = now - self.calibration_wait_start
        if elapsed < self.CALIBRATION_WAIT:
            return False

        self.calibration_data.append(self._calc_shrimp_ratio(landmarks))
        self.calibration_tilt_data.append(self._calc_sideways_tilt(landmarks))

        if len(self.calibration_data) >= self.CALIBRATION_FRAMES:
            self.baseline_ratio = sum(self.calibration_data) / len(self.calibration_data)
            self.baseline_tilt = sum(self.calibration_tilt_data) / len(self.calibration_tilt_data)
            entry_off, exit_off, _ = SENSITIVITY[self.sensitivity]
            self.entry_threshold = self.baseline_ratio + entry_off
            self.exit_threshold = self.baseline_ratio + exit_off
            self.calibrated = True
            self.session_start = time.time()
            return True

        return False

    def calibration_progress(self):
        """Returns (countdown_remaining, frames_collected, frames_needed)."""
        if self.calibration_wait_start is None:
            return self.CALIBRATION_WAIT, 0, self.CALIBRATION_FRAMES
        elapsed = time.time() - self.calibration_wait_start
        countdown = max(0.0, self.CALIBRATION_WAIT - elapsed)
        return countdown, len(self.calibration_data), self.CALIBRATION_FRAMES

    # --- analysis ---

    def analyze_posture(self, landmarks):
        """Analyze one frame. Returns a metrics dict."""
        ratio = self._calc_shrimp_ratio(landmarks)
        tilt = self._calc_sideways_tilt(landmarks)

        # Bad posture = forward hunch OR sideways lean
        entry_off, exit_off, _ = SENSITIVITY[self.sensitivity]
        tilt_threshold = self.baseline_tilt + entry_off if self.is_shrimping else self.baseline_tilt + entry_off
        forward_bad = ratio > (self.exit_threshold if self.is_shrimping else self.entry_threshold)
        sideways_bad = tilt > (self.baseline_tilt + exit_off if self.is_shrimping else self.baseline_tilt + entry_off)
        is_bad = forward_bad or sideways_bad

        if is_bad:
            self.consecutive_bad += 1
        else:
            self.consecutive_bad = 0

        # Transition to shrimping
        if not self.is_shrimping and self.consecutive_bad >= self.required_frames:
            self.is_shrimping = True
            self.consecutive_bad = 0
            self.shrimp_count += 1
            self.bad_posture_start = time.time()

        # Transition to good posture
        if self.is_shrimping and self.consecutive_bad == 0 and ratio <= self.exit_threshold:
            if self.bad_posture_start:
                self.total_bad_seconds += time.time() - self.bad_posture_start
            self.is_shrimping = False
            self.bad_posture_start = None

        bad_seconds = 0.0
        if self.is_shrimping and self.bad_posture_start:
            bad_seconds = time.time() - self.bad_posture_start

        return {
            "ratio": ratio,
            "baseline_ratio": self.baseline_ratio,
            "is_shrimping": self.is_shrimping,
            "shrimp_count": self.shrimp_count,
            "bad_seconds": bad_seconds,
            "total_bad_seconds": self.total_bad_seconds + bad_seconds,
        }

    # --- drawing ---

    def draw_skeleton(self, frame, landmarks, analysis):
        """Draw tracked landmarks and connections on the frame."""
        h, w = frame.shape[:2]
        is_bad = analysis["is_shrimping"] if analysis else False

        color_point = (0, 0, 220) if is_bad else (0, 200, 0)
        color_line = (0, 0, 180) if is_bad else (0, 180, 0)
        color_outline = (255, 255, 255)

        # Draw connections
        for (i, j) in SKELETON_CONNECTIONS:
            a, b = landmarks[i], landmarks[j]
            if a.visibility > 0.5 and b.visibility > 0.5:
                pt1 = (int(a.x * w), int(a.y * h))
                pt2 = (int(b.x * w), int(b.y * h))
                cv2.line(frame, pt1, pt2, color_line, 2, cv2.LINE_AA)

        # Draw points
        for idx in TRACKED_INDICES:
            lm = landmarks[idx]
            if lm.visibility > 0.5:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 6, color_point, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 6, color_outline, 1, cv2.LINE_AA)

    def draw_overlay(self, frame, analysis):
        """Draw the HUD overlay (stats, alerts, controls help)."""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        is_bad = analysis["is_shrimping"]

        # --- red tint when shrimping ---
        if is_bad:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

        # --- top-left stats ---
        y = 30
        self._put_text(frame, f"Shrimp Count: {analysis['shrimp_count']}", (15, y), font, 0.65, (255, 255, 255))
        y += 28
        self._put_text(frame, f"Ratio: {analysis['ratio']:.2f} (baseline: {analysis['baseline_ratio']:.2f})",
                       (15, y), font, 0.55, (200, 200, 200))
        y += 25
        status_text = "SHRIMPING!" if is_bad else "GOOD POSTURE"
        status_color = (0, 0, 255) if is_bad else (0, 255, 0)
        self._put_text(frame, f"Status: {status_text}", (15, y), font, 0.6, status_color)

        if is_bad and analysis["bad_seconds"] > 0:
            y += 25
            self._put_text(frame, f"Bad posture for {analysis['bad_seconds']:.0f}s", (15, y), font, 0.55, (0, 100, 255))

        # --- pulsing center alert ---
        if is_bad and time.time() % 1.0 < 0.6:
            text = "STRAIGHTEN UP!"
            text_size = cv2.getTextSize(text, font, 1.2, 3)[0]
            tx = (w - text_size[0]) // 2
            ty = (h + text_size[1]) // 2
            self._put_text(frame, text, (tx, ty), font, 1.2, (0, 0, 255), thickness=3)

        # --- bottom-left: sensitivity ---
        labels = {"strict": "1:STRICT", "normal": "2:NORMAL", "relaxed": "3:RELAXED"}
        parts = []
        for key, label in labels.items():
            if key == self.sensitivity:
                parts.append(f"[{label}]")
            else:
                parts.append(f" {label} ")
        sens_text = "  ".join(parts)
        self._put_text(frame, sens_text, (15, h - 15), font, 0.45, (200, 200, 200))

        # --- bottom-right: controls ---
        self._put_text(frame, "q:Quit  c:Recalibrate", (w - 230, h - 15), font, 0.45, (160, 160, 160))

    def draw_calibration_overlay(self, frame):
        """Draw calibration UI (countdown + progress bar)."""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        countdown, collected, needed = self.calibration_progress()

        if countdown > 0:
            text = f"Sit in your best posture... {int(countdown) + 1}"
            text_size = cv2.getTextSize(text, font, 0.85, 2)[0]
            tx = (w - text_size[0]) // 2
            ty = (h // 2)
            self._put_text(frame, text, (tx, ty), font, 0.85, (0, 255, 255), thickness=2)
        else:
            text = "Calibrating..."
            text_size = cv2.getTextSize(text, font, 0.75, 2)[0]
            tx = (w - text_size[0]) // 2
            self._put_text(frame, text, (tx, h // 2 - 20), font, 0.75, (0, 255, 255), thickness=2)

            # Progress bar
            bar_w = 300
            bar_h = 20
            bx = (w - bar_w) // 2
            by = h // 2 + 10
            progress = collected / needed if needed > 0 else 0
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (100, 100, 100), 2)
            fill_w = int(bar_w * progress)
            cv2.rectangle(frame, (bx, by), (bx + fill_w, by + bar_h), (0, 255, 0), -1)

    @staticmethod
    def _put_text(frame, text, org, font, scale, color, thickness=2):
        """Draw text with a dark shadow for readability."""
        cv2.putText(frame, text, (org[0] + 1, org[1] + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    detector = PostureDetector(sensitivity="normal")

    # Start WebSocket server in background thread
    ws_thread = threading.Thread(target=_run_ws_server, daemon=True)
    ws_thread.start()

    # Initialize MediaPipe Pose Landmarker
    print("Loading pose model...")
    base_options = python.BaseOptions(model_asset_path="pose_landmarker_heavy.task")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("=== Are You Shrimpin'? ===")
    print("Sit in your best posture for calibration...")
    print("Controls: q=quit  c=recalibrate  1/2/3=sensitivity")

    calibrating = True
    calibrated_flash_until = 0.0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        # Convert and detect
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = pose_landmarker.detect(mp_image)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]

            if calibrating:
                # Draw skeleton in green during calibration
                detector.draw_skeleton(frame, landmarks, None)
                done = detector.calibrate_frame(landmarks)
                detector.draw_calibration_overlay(frame)

                if done:
                    calibrating = False
                    calibrated_flash_until = time.time() + 2.0
                    print(f"Calibrated! Baseline ratio: {detector.baseline_ratio:.3f}")
                    print(f"  Entry threshold: {detector.entry_threshold:.3f}")
                    print(f"  Exit threshold:  {detector.exit_threshold:.3f}")
                broadcast({"status": "calibrating", "calibrated": False, "shrimp_count": 0, "is_shrimping": False})
            else:
                analysis = detector.analyze_posture(landmarks)
                detector.draw_skeleton(frame, landmarks, analysis)
                detector.draw_overlay(frame, analysis)
                broadcast({
                    "status": "shrimping" if analysis["is_shrimping"] else "good",
                    "calibrated": True,
                    "is_shrimping": analysis["is_shrimping"],
                    "shrimp_count": analysis["shrimp_count"],
                    "ratio": round(analysis["ratio"], 3),
                    "baseline_ratio": round(analysis["baseline_ratio"], 3),
                    "bad_seconds": round(analysis["bad_seconds"], 1),
                })

                # "Calibrated!" flash
                if time.time() < calibrated_flash_until:
                    h, w = frame.shape[:2]
                    text = "Calibrated!"
                    sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                    detector._put_text(frame, text, ((w - sz[0]) // 2, h // 2),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        else:
            # No pose detected
            detector._put_text(frame, "No pose detected - make sure you're visible",
                               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
            broadcast({"status": "no_pose", "calibrated": detector.calibrated, "is_shrimping": False, "shrimp_count": detector.shrimp_count})

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            detector.reset_calibration()
            calibrating = True
            print("Recalibrating... sit in your best posture.")
        elif key == ord("1"):
            detector.set_sensitivity("strict")
            print("Sensitivity: STRICT")
        elif key == ord("2"):
            detector.set_sensitivity("normal")
            print("Sensitivity: NORMAL")
        elif key == ord("3"):
            detector.set_sensitivity("relaxed")
            print("Sensitivity: RELAXED")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession stats: {detector.shrimp_count} shrimp events, "
          f"{detector.total_bad_seconds:.0f}s total bad posture")


if __name__ == "__main__":
    main()
