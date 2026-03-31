import time
import numpy as np

class FacePresenceDetector:
    def __init__(self):
        self.absence_start_time = None

    def detect(self, face_landmarks):
        current_time = time.time()

        # ---------- FACE PRESENT ----------
        is_face_present = False

        if face_landmarks is not None:
            if isinstance(face_landmarks, (list, tuple)) and len(face_landmarks) > 0:
                is_face_present = True
            elif isinstance(face_landmarks, np.ndarray) and face_landmarks.size > 0:
                is_face_present = True

        if is_face_present:
            self.absence_start_time = None
            return {
                "face_present": True,
                "duration": 0,
                "suspicion": 0
            }

        # ---------- FACE MISSING ----------
        if self.absence_start_time is None:
            self.absence_start_time = current_time

        duration = current_time - self.absence_start_time

        if duration < 1:
            suspicion = 0.2
        elif duration < 3:
            suspicion = 0.5
        elif duration < 6:
            suspicion = 0.8
        else:
            suspicion = 1.0

        return {
            "face_present": False,
            "duration": round(duration, 2),
            "suspicion": suspicion
        }