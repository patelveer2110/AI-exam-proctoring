import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# ---------- Load Face Landmarker (same model as gaze) ----------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

mp_image_module = mp.Image


# ---------- Head Pose Detection Function ----------
def detect_head_pose(frame):
    """
    Returns:
        dict {
            yaw: float,
            pitch: float,
            roll: float,
            direction: str
        }
    """

    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp_image_module(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    if not result.face_landmarks:
        return {
            "yaw": 0,
            "pitch": 0,
            "roll": 0,
            "direction": "No Face"
        }

    face_landmarks = result.face_landmarks[0]

    # Select important landmark indices
    # Nose tip, chin, left eye, right eye, left mouth, right mouth
    landmark_ids = [1, 152, 33, 263, 61, 291]

    face_2d = []
    face_3d = []

    for idx in landmark_ids:
        lm = face_landmarks[idx]

        x, y = int(lm.x * w), int(lm.y * h)

        face_2d.append([x, y])
        face_3d.append([x, y, lm.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    # Camera matrix
    focal_length = w
    cam_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ])

    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vec, translation_vec = cv2.solvePnP(
        face_3d, face_2d, cam_matrix, dist_matrix
    )

    rmat, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch = angles[0] * 360
    yaw = angles[1] * 360
    roll = angles[2] * 360

    # -------- Direction Classification --------
    direction = "Straight"

    if yaw < -10:
        direction = "Looking Left"
    elif yaw > 10:
        direction = "Looking Right"
    elif pitch < -10:
        direction = "Looking Down"
    elif pitch > 10:
        direction = "Looking Up"

    return {
        "yaw": round(yaw, 2),
        "pitch": round(pitch, 2),
        "roll": round(roll, 2),
        "direction": direction
    }