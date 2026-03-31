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

# import time


# class SuspicionEngine:

#     def __init__(self):
#         self.start_time = time.time()
#         self.last_time = self.start_time

#         self.total_duration = 0

#         self.head_turn_time = 0
#         self.gaze_away_time = 0
#         self.multiple_face_time = 0
#         self.object_time = 0
#         self.audio_time = 0

#     def update(self, face_count, gaze_status, objects, audio_status, head_direction):

#         current_time = time.time()
#         delta = current_time - self.last_time
#         self.last_time = current_time

#         self.total_duration += delta

#         # Head turned
#         if head_direction != "Straight":
#             self.head_turn_time += delta

#         # Gaze away (only if head straight)
#         if head_direction == "Straight":
#             if gaze_status != "Looking Center":
#                 self.gaze_away_time += delta

#         # Multiple faces
#         if face_count > 1:
#             self.multiple_face_time += delta

#         # Suspicious objects
#         suspicious_objects = ["cell phone", "book", "laptop"]
#         if any(obj in suspicious_objects for obj in objects):
#             self.object_time += delta

#         # Audio
#         if audio_status == "High Noise":
#             self.audio_time += delta

#     def get_live_score(self):
#         score = 0

#         score += min(self.head_turn_time * 5, 100)
#         score += min(self.gaze_away_time * 4, 80)
#         score += min(self.multiple_face_time * 10, 100)
#         score += min(self.object_time * 8, 100)
#         score += min(self.audio_time * 6, 100)

#         return int(score)
#     def get_suspicion_level(self):
#         score = self.get_live_score()

#         if score < 30:
#             return "LOW"
#         elif score < 80:
#             return "MEDIUM"
#         elif score < 150:
#             return "HIGH"
#         else:
#             return "CRITICAL"

#     def get_final_report(self):
#         return {
#             "total_duration_sec": round(self.total_duration, 2),
#             "head_turn_time": round(self.head_turn_time, 2),
#             "gaze_away_time": round(self.gaze_away_time, 2),
#             "multiple_face_time": round(self.multiple_face_time, 2),
#             "object_time": round(self.object_time, 2),
#             "audio_time": round(self.audio_time, 2),
#             "final_score": self.get_live_score(),
#             "suspicion_level": self.get_suspicion_level()
        
#         }
import time


class SuspicionEngine:
    def __init__(self):
        self.last_time = time.time()

        self.metrics = {
            "head": 0.0,
            "gaze": 0.0,
            "face": 0.0,
            "object": 0.0,
            "audio": 0.0
        }

        self.window_time = 20.0
        self.total_time = 0.0

        self.weights = {
            "head": 0.25,
            "gaze": 0.25,
            "face": 0.2,
            "object": 0.2,
            "audio": 0.1
        }

        # 🔥 Face grace logic
        self.face_missing_time = 0.0
        self.face_grace = 3.0

        # 🔥 Cached values (performance boost)
        self.cached_score = 0.0
        self.cached_level = "LOW"

    def update(self, face_present, gaze_data, object_data, audio_data, head_data):
        now = time.time()
        delta = now - self.last_time
        self.last_time = now

        self.total_time += delta

        # ---------- SLIDING WINDOW ----------
        if self.total_time > self.window_time:
            scale = self.window_time / self.total_time
            for k in self.metrics:
                self.metrics[k] *= scale
            self.total_time = self.window_time

        # ---------- EXTRACT VALUES (ONCE) ----------
        gaze_dir = gaze_data.get("gaze", "Unknown")
        head_dir = head_data.get("direction", "Unknown")
        obj_score = object_data.get("max_suspicion", 0)
        audio_score = audio_data.get("suspicion", 0)

        # ---------- FACE TIMER ----------
        if not face_present:
            self.face_missing_time += delta
        else:
            self.face_missing_time = 0

        # ---------- HEAD ----------
        if head_dir not in ("Straight", "No Face"):
            self.metrics["head"] += delta

        # ---------- GAZE ----------
        if head_dir == "Straight" and gaze_dir not in ("Center", "Unknown"):
            self.metrics["gaze"] += delta

        # ---------- FACE ----------
        if self.face_missing_time > self.face_grace:
            self.metrics["face"] += delta

        # ---------- OBJECT ----------
        if obj_score > 0.3:
            self.metrics["object"] += delta * obj_score

        # ---------- AUDIO ----------
        if audio_score > 0.2:
            self.metrics["audio"] += delta * audio_score

        # ---------- SMART DECAY (ONLY WHEN SAFE) ----------
        if face_present and gaze_dir == "Center" and head_dir == "Straight":
            for k in self.metrics:
                self.metrics[k] *= 0.997  # slower decay = more stable

        # ---------- RECOVERY BOOST ----------
        if face_present and gaze_dir == "Center":
            self.metrics["face"] *= 0.97
            self.metrics["gaze"] *= 0.97

        # ---------- CACHE SCORE ----------
        self._update_cached_score()

    def _update_cached_score(self):
        score = 0.0

        for k in self.metrics:
            ratio = self.metrics[k] / self.window_time
            score += ratio * self.weights[k]

        score = min(score * 100, 100)
        self.cached_score = round(score, 2)

        # ---------- LEVEL ----------
        if score < 20:
            self.cached_level = "LOW"
        elif score < 40:
            self.cached_level = "MEDIUM"
        elif score < 65:
            self.cached_level = "HIGH"
        else:
            self.cached_level = "CRITICAL"

    def get_live_output(self):
        return {
            "score": self.cached_score,
            "level": self.cached_level,
            "metrics": {k: round(v, 2) for k, v in self.metrics.items()}
        }