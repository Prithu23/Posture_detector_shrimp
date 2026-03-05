import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Download the pose landmarker model first
# You'll need to download it from: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

# Initialize pose detector
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break
    
    # Convert to RGB and create MediaPipe Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect pose
    detection_result = detector.detect(mp_image)
    
    # Draw landmarks if detected
    if detection_result.pose_landmarks:
        for landmark in detection_result.pose_landmarks[0]:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    cv2.imshow("Posture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()