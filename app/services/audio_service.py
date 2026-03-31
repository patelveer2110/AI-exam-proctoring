import sounddevice as sd
import numpy as np
import time
class AudioDetector:
    def __init__(self):
        self.sample_rate = 16000
        self.duration = 0.5

        self.last_check = 0
        self.cached_result = {
            "audio_state": "Normal",
            "volume": 0,
            "duration": 0,
            "suspicion": 0
        }

    def detect(self):
        current_time = time.time()

        # 🔥 run every 2 seconds only
        if current_time - self.last_check < 2:
            return self.cached_result

        self.last_check = current_time

        recording = sd.rec(
            int(self.sample_rate * self.duration),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()

        volume = np.linalg.norm(recording) / len(recording)

        suspicion = 0
        if volume > 0.02:
            suspicion = 0.4

        self.cached_result = {
            "audio_state": "Noise" if suspicion > 0 else "Normal",
            "volume": round(volume, 5),
            "duration": 0,
            "suspicion": suspicion
        }

        return self.cached_result