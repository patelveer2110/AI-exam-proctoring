import cv2
import numpy as np
import mediapipe as mp
import time
from pathlib import Path
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


# ---------- Load Model ----------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)
mp_image_module = mp.Image


class HeadPoseEstimator:
    def __init__(self):
        self.prev_yaw = 0
        self.alpha = 0.3  # less smoothing = faster response

        self.last_direction = "Straight"
        self.direction_start_time = time.time()

        self.stable_count = 0
        self.last_raw_direction = "Straight"

    def smooth(self, current, previous):
        return self.alpha * current + (1 - self.alpha) * previous

    def get_direction(self, yaw):
        # 🔥 Strong dead zone (ignore small movements)
        if -15 < yaw < 15:
            return "Straight"
        elif yaw <= -18:
            return "Looking Left"
        elif yaw >= 18:
            return "Looking Right"
        return "Straight"

def detect(self, landmarks, frame_shape):
    h, w = frame_shape

    if landmarks is None:
        return {
            "yaw": 0,
            "pitch": 0,
            "roll": 0,
            "direction": "No Face",
            "suspicion": 0
        }

    face_2d = []
    landmark_ids = [1, 152, 33, 263, 61, 291]

    for idx in landmark_ids:
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        face_2d.append([x, y])

    face_2d = np.array(face_2d, dtype=np.float64)

    face_3d = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0]
    ], dtype=np.float64)

    focal_length = w
    cam_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ])

    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vec, _ = cv2.solvePnP(
        face_3d, face_2d, cam_matrix, dist_matrix
    )

    rmat, _ = cv2.Rodrigues(rotation_vec)
    angles, *_ = cv2.RQDecomp3x3(rmat)

    pitch, yaw, roll = angles

    yaw = self.smooth(yaw, self.prev_yaw)
    pitch = self.smooth(pitch, self.prev_pitch)
    roll = self.smooth(roll, self.prev_roll)

    self.prev_yaw, self.prev_pitch, self.prev_roll = yaw, pitch, roll

    direction = self.get_direction(yaw, pitch)

    return {
        "yaw": round(yaw, 2),
        "pitch": round(pitch, 2),
        "roll": round(roll, 2),
        "direction": direction,
        "suspicion": 0,
        "landmarks": landmarks
    }