import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
DURATION = 1  # seconds
THRESHOLD = 0.02  # adjust sensitivity


def detect_audio():
    recording = sd.rec(
        int(SAMPLE_RATE * DURATION),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()

    volume_norm = np.linalg.norm(recording) / len(recording)

    if volume_norm > THRESHOLD:
        return "High Noise"
    else:
        return "Normal"