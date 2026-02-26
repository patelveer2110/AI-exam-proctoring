import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

# ----------------------------
# Load Model
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

# Iris indices
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Eye corner landmarks
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263


def get_iris_center(landmarks, iris_indices, w, h):
    x = 0
    y = 0
    for idx in iris_indices:
        lm = landmarks[idx]
        x += lm.x * w
        y += lm.y * h
    return int(x / len(iris_indices)), int(y / len(iris_indices))


def detect_gaze(frame):

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if not result.face_landmarks:
        return "No Face"

    landmarks = result.face_landmarks[0]

    # ---- LEFT EYE ----
    left_iris_x, left_iris_y = get_iris_center(landmarks, LEFT_IRIS, w, h)

    left_outer = landmarks[LEFT_EYE_OUTER]
    left_inner = landmarks[LEFT_EYE_INNER]

    left_outer_x = int(left_outer.x * w)
    left_inner_x = int(left_inner.x * w)

    # ---- RIGHT EYE ----
    right_iris_x, right_iris_y = get_iris_center(landmarks, RIGHT_IRIS, w, h)

    right_outer = landmarks[RIGHT_EYE_OUTER]
    right_inner = landmarks[RIGHT_EYE_INNER]

    right_outer_x = int(right_outer.x * w)
    right_inner_x = int(right_inner.x * w)

    # Draw iris centers
    cv2.circle(frame, (left_iris_x, left_iris_y), 4, (0,255,0), -1)
    cv2.circle(frame, (right_iris_x, right_iris_y), 4, (0,0,255), -1)

    # ---- Calculate gaze ratio ----
    left_ratio = (left_iris_x - left_outer_x) / (left_inner_x - left_outer_x)
    right_ratio = (right_iris_x - right_outer_x) / (right_inner_x - right_outer_x)

    gaze_ratio = (left_ratio + right_ratio) / 2

    print("Gaze Ratio:", gaze_ratio)

    # ---- Direction Logic ----
    if gaze_ratio < 0.4:
        return "Looking Left"
    elif gaze_ratio > 0.6:
        return "Looking Right"
    else:
        return "Looking Center"