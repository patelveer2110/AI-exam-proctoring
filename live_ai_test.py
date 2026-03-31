import cv2
import logging
import time

from app.services.gaze_service import GazeEstimator
from app.services.headpose_service import HeadPoseEstimator
from app.services.object_service import ObjectDetector
from app.services.audio_service import AudioDetector
from app.services.face_service import FacePresenceDetector
from app.services.suspicion_engine import SuspicionEngine


# ---------------- INIT ---------------- #
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

gaze_detector = GazeEstimator()
head_detector = HeadPoseEstimator()
object_detector = ObjectDetector()
audio_detector = AudioDetector()
face_detector = FacePresenceDetector()

engine = SuspicionEngine()

frame_count = 0
object_data = {"objects": [], "max_suspicion": 0}
audio_data = {"audio_state": "Normal", "suspicion": 0}

print("✅ Optimized AI Proctoring Started...")
print("Press 'q' to exit\n")
y = 30

def draw(text):
    global y
    cv2.putText(frame, text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 2)
    y += 25


# ---------------- LOOP ---------------- #
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera not detected")
            break

        frame = cv2.resize(frame, (640, 480))
        frame_count += 1

        # 🔥 RESET TEXT POSITION EVERY FRAME
        y = 30

        # ---------- LIGHT TASKS ----------
        gaze_data = gaze_detector.detect(frame)
        head_data = head_detector.detect(frame)

        face_present = head_data["direction"] != "No Face"

        # ---------- HEAVY TASKS ----------
        if frame_count % 10 == 0:
            object_data = object_detector.detect(frame)

        if frame_count % 15 == 0:
            audio_data = audio_detector.detect()

        # ---------- ENGINE ----------
        engine.update(
            face_present=face_present,
            gaze_data=gaze_data,
            object_data=object_data,
            audio_data=audio_data,
            head_data=head_data
        )

        output = engine.get_live_output()

        # ---------- DRAW FUNCTION ----------
        def draw(frame, text, y):
            cv2.putText(frame, text, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 2)
            return y + 25
        # ---------- DISPLAY ----------
        y = 30

        y = draw(frame, f"Gaze: {gaze_data['gaze']}", y)
        y = draw(frame, f"Head: {head_data['direction']}", y)
        y = draw(frame, f"Face: {'Present' if face_present else 'Missing'}", y)

        y = draw(frame, f"Audio: {audio_data['audio_state']}", y)
        y = draw(frame, f"Objects: {', '.join([o['object'] for o in object_data['objects']]) if object_data['objects'] else 'None'}", y)

        y = draw(frame, "---- METRICS ----", y)

        for k, v in output["metrics"].items():
            y = draw(frame, f"{k.upper()}: {v:.2f}", y)

        y = draw(frame, "------------------", y)
        y = draw(frame, f"SCORE: {output['score']}%", y)
        y = draw(frame, f"LEVEL: {output['level']}", y)

        # 🔥 THIS WAS MISSING
        cv2.imshow("AI Proctor", frame)

        # 🔥 THIS IS REQUIRED
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print("Error:", str(e))


# ---------------- CLEANUP ---------------- #
finally:
    cap.release()
    cv2.destroyAllWindows()

    print("\n📊 FINAL REPORT")
    print(output)