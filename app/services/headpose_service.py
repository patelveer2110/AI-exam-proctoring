import cv2
import numpy as np
import mediapipe as mp
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
    num_faces=1,
)

detector = vision.FaceLandmarker.create_from_options(options)
mp_image_module = mp.Image


class HeadPoseEstimator:
    def __init__(self):
        self.prev_yaw = 0
        self.prev_pitch = 0
        self.prev_roll = 0
        self.alpha = 0.3

    def smooth(self, current, previous):
        return self.alpha * current + (1 - self.alpha) * previous

    def get_direction(self, yaw):
        if -15 < yaw < 15:
            return "Straight"
        if yaw <= -18:
            return "Looking Left"
        if yaw >= 18:
            return "Looking Right"
        return "Straight"

    def detect(self, frame):
        if frame is None:
            return {
                "yaw": 0,
                "pitch": 0,
                "roll": 0,
                "direction": "No Face",
                "suspicion": 0,
                "landmarks": None,
            }

        h, w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_image_module(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        if not detection_result.face_landmarks:
            return {
                "yaw": 0,
                "pitch": 0,
                "roll": 0,
                "direction": "No Face",
                "suspicion": 0,
                "landmarks": None,
            }

        landmarks = detection_result.face_landmarks[0]

        face_2d = []
        landmark_ids = [1, 152, 33, 263, 61, 291]

        try:
            for idx in landmark_ids:
                lm = landmarks[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                face_2d.append([x, y])
        except (IndexError, TypeError):
            return {
                "yaw": 0,
                "pitch": 0,
                "roll": 0,
                "direction": "No Face",
                "suspicion": 0,
                "landmarks": None,
            }

        face_2d = np.array(face_2d, dtype=np.float64)

        face_3d = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, -330.0, -65.0],
                [-225.0, 170.0, -135.0],
                [225.0, 170.0, -135.0],
                [-150.0, -150.0, -125.0],
                [150.0, -150.0, -125.0],
            ],
            dtype=np.float64,
        )

        cam_matrix = np.array(
            [
                [w, 0, w / 2],
                [0, w, h / 2],
                [0, 0, 1],
            ]
        )
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        if not success:
            return {
                "yaw": 0,
                "pitch": 0,
                "roll": 0,
                "direction": "No Face",
                "suspicion": 0,
                "landmarks": None,
            }

        rmat, _ = cv2.Rodrigues(rotation_vec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, roll = angles

        yaw = self.smooth(yaw, self.prev_yaw)
        pitch = self.smooth(pitch, self.prev_pitch)
        roll = self.smooth(roll, self.prev_roll)
        self.prev_yaw, self.prev_pitch, self.prev_roll = yaw, pitch, roll

        direction = self.get_direction(yaw)

        return {
            "yaw": round(yaw, 2),
            "pitch": round(pitch, 2),
            "roll": round(roll, 2),
            "direction": direction,
            "suspicion": 0.3 if direction != "Straight" else 0,
            "landmarks": landmarks,
        }
