import time


class AudioDetector:
    def __init__(self):
        self.last_check = 0
        self.cached_result = {
            "audio_state": "Normal",
            "volume": 0,
            "duration": 0,
            "suspicion": 0,
        }

    def detect(self, audio_level=None, audio_state=None):
        current_time = time.time()

        # Keep lightweight caching to reduce per-frame CPU work.
        if current_time - self.last_check < 2:
            return self.cached_result

        self.last_check = current_time

        volume = 0.0
        suspicion = 0.0
        normalized_state = "Normal"

        if isinstance(audio_level, (int, float)):
            volume = max(0.0, min(float(audio_level), 1.0))
            if volume > 0.2:
                suspicion = 0.4
                normalized_state = "Noise"

        if isinstance(audio_state, str) and audio_state.strip():
            state = audio_state.strip().lower()
            if state in {"noise", "high noise", "talking", "speaking"}:
                suspicion = max(suspicion, 0.4)
                normalized_state = "Noise"
            elif state in {"normal", "silent", "quiet"}:
                normalized_state = "Normal"
                suspicion = min(suspicion, 0.1)

        self.cached_result = {
            "audio_state": normalized_state,
            "volume": round(volume, 5),
            "duration": 0,
            "suspicion": suspicion,
        }

        return self.cached_result
