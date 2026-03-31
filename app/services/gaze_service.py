import numpy as np
import time

# Iris indices
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263


class GazeEstimator:
    def __init__(self):
        self.prev_ratio = 0.5
        self.alpha = 0.4  # smoother but responsive

        self.last_direction = "Center"
        self.direction_start_time = time.time()

    def smooth(self, current, previous):
        return self.alpha * current + (1 - self.alpha) * previous

    def get_iris_center(self, landmarks, iris_indices, w, h):
        x, y = 0, 0
        for idx in iris_indices:
            lm = landmarks[idx]
            x += lm.x * w
            y += lm.y * h
        return x / len(iris_indices), y / len(iris_indices)

    def get_direction(self, ratio):
        # 🔥 more stable thresholds
        if ratio < 0.32:
            return "Looking Left"
        elif ratio > 0.68:
            return "Looking Right"
        return "Center"

    # 🔥 IMPORTANT: NOW ACCEPT LANDMARKS (NOT FRAME)
    def detect(self, landmarks, w, h):
        if landmarks is None:
            return {
                "gaze": "No Face",
                "ratio": 0.5,
                "duration": 0,
                "suspicion": 0
            }

        # ---- LEFT EYE ----
        left_iris_x, _ = self.get_iris_center(landmarks, LEFT_IRIS, w, h)
        left_outer_x = landmarks[LEFT_EYE_OUTER].x * w
        left_inner_x = landmarks[LEFT_EYE_INNER].x * w

        # ---- RIGHT EYE ----
        right_iris_x, _ = self.get_iris_center(landmarks, RIGHT_IRIS, w, h)
        right_outer_x = landmarks[RIGHT_EYE_OUTER].x * w
        right_inner_x = landmarks[RIGHT_EYE_INNER].x * w

        if (left_inner_x - left_outer_x) == 0 or (right_inner_x - right_outer_x) == 0:
            return {"gaze": "Unknown", "ratio": 0.5, "duration": 0, "suspicion": 0}

        left_ratio = (left_iris_x - left_outer_x) / (left_inner_x - left_outer_x)
        right_ratio = (right_iris_x - right_outer_x) / (right_inner_x - right_outer_x)

        gaze_ratio = (left_ratio + right_ratio) / 2

        gaze_ratio = self.smooth(gaze_ratio, self.prev_ratio)
        self.prev_ratio = gaze_ratio

        direction = self.get_direction(gaze_ratio)

        current_time = time.time()

        if direction == self.last_direction:
            duration = current_time - self.direction_start_time
        else:
            self.last_direction = direction
            self.direction_start_time = current_time
            duration = 0

        suspicion = 0
        if direction != "Center":
            if duration > 2:
                suspicion = 0.3
            if duration > 4:
                suspicion = 0.6

        return {
            "gaze": direction,
            "ratio": round(gaze_ratio, 3),
            "duration": round(duration, 2),
            "suspicion": suspicion
        }