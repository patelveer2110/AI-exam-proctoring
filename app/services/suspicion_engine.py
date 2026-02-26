# def calculate_score(face_count, gaze_status, objects, audio_status, head_direction):

#     score = 0

#     # Multiple faces
#     if face_count > 1:
#         score += 50

#     # Only consider gaze if head is straight
#     if head_direction == "Straight":
#         if gaze_status != "Looking Center":
#             score += 20

#     # Head turned
#     if head_direction != "Straight":
#         score += 20

#     # Suspicious objects
#     suspicious_objects = ["cell phone", "book", "laptop"]

#     for obj in objects:
#         if obj in suspicious_objects:
#             score += 40

#     # Audio
#     if audio_status == "High Noise":
#         score += 30

#     return score

import time


class SuspicionEngine:

    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time

        self.total_duration = 0

        self.head_turn_time = 0
        self.gaze_away_time = 0
        self.multiple_face_time = 0
        self.object_time = 0
        self.audio_time = 0

    def update(self, face_count, gaze_status, objects, audio_status, head_direction):

        current_time = time.time()
        delta = current_time - self.last_time
        self.last_time = current_time

        self.total_duration += delta

        # Head turned
        if head_direction != "Straight":
            self.head_turn_time += delta

        # Gaze away (only if head straight)
        if head_direction == "Straight":
            if gaze_status != "Looking Center":
                self.gaze_away_time += delta

        # Multiple faces
        if face_count > 1:
            self.multiple_face_time += delta

        # Suspicious objects
        suspicious_objects = ["cell phone", "book", "laptop"]
        if any(obj in suspicious_objects for obj in objects):
            self.object_time += delta

        # Audio
        if audio_status == "High Noise":
            self.audio_time += delta

    def get_live_score(self):
        score = 0

        score += min(self.head_turn_time * 5, 100)
        score += min(self.gaze_away_time * 4, 80)
        score += min(self.multiple_face_time * 10, 100)
        score += min(self.object_time * 8, 100)
        score += min(self.audio_time * 6, 100)

        return int(score)
    def get_suspicion_level(self):
        score = self.get_live_score()

        if score < 30:
            return "LOW"
        elif score < 80:
            return "MEDIUM"
        elif score < 150:
            return "HIGH"
        else:
            return "CRITICAL"

    def get_final_report(self):
        return {
            "total_duration_sec": round(self.total_duration, 2),
            "head_turn_time": round(self.head_turn_time, 2),
            "gaze_away_time": round(self.gaze_away_time, 2),
            "multiple_face_time": round(self.multiple_face_time, 2),
            "object_time": round(self.object_time, 2),
            "audio_time": round(self.audio_time, 2),
            "final_score": self.get_live_score(),
            "suspicion_level": self.get_suspicion_level()
        }