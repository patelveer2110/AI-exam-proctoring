import cv2
import logging
import time
from datetime import datetime

from app.services.gaze_service import detect_gaze
from app.services.headpose_service import detect_head_pose
from app.services.object_service import detect_objects
from app.services.audio_service import detect_audio
from app.services.suspicion_engine import SuspicionEngine


# ---------------- LOGGING SETUP ---------------- #
logging.basicConfig(
    filename="suspicion_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# ---------------- INITIALIZATION ---------------- #
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

engine = SuspicionEngine()

frame_count = 0
objects = []
audio_status = "Silent"

session_start_time = time.time()
score_history = []

print("✅ AI Proctoring System Started...")
print("Press 'q' to stop.\n")


# ---------------- MAIN LOOP ---------------- #
try:
    logging.info("SESSION STARTED")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frame_count += 1

        # --------- AI DETECTIONS --------- #
        gaze_status = detect_gaze(frame)
        headpose_data = detect_head_pose(frame)

        # Run heavy models every 5 frames
        if frame_count % 5 == 0:
            objects = detect_objects(frame)
            audio_status = detect_audio()

        face_count = 1  # Replace with real face detector if needed

        # --------- SCORE CALCULATION --------- #
        engine.update(
            face_count,
            gaze_status,
            objects,
            audio_status,
            headpose_data["direction"]
        )

        suspicion_score = engine.get_live_score()

        score_history.append(suspicion_score)

        # --------- LOGGING --------- #
        logging.info(
            f"Score={suspicion_score}, "
            f"Gaze={gaze_status}, "
            f"Yaw={headpose_data['yaw']}, "
            f"Audio={audio_status}, "
            f"Objects={objects}, "
            f"HeadPose={headpose_data['direction']}"
        )

        # --------- DISPLAY --------- #
        cv2.putText(frame, f"Gaze: {gaze_status}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Head: {headpose_data['direction']}",
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Yaw: {round(headpose_data['yaw'], 2)}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Audio: {audio_status}",
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Objects: {', '.join(objects) if objects else 'None'}",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Score: {suspicion_score}",
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

        cv2.imshow("AI Proctor System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    logging.error(f"Error during session: {str(e)}")
    print(f"An error occurred: {str(e)}")


# ---------------- SESSION SUMMARY ---------------- #
# ---------------- SESSION SUMMARY ---------------- #
finally:
    cap.release()
    cv2.destroyAllWindows()

    report = engine.get_final_report()

    print("\n📊 FINAL AI PROCTOR REPORT")
    print("====================================")
    print(f"Total Session Duration: {report['total_duration_sec']} sec")
    print(f"Head Turn Time: {report['head_turn_time']} sec")
    print(f"Gaze Away Time: {report['gaze_away_time']} sec")
    print(f"Multiple Face Time: {report['multiple_face_time']} sec")
    print(f"Suspicious Object Time: {report['object_time']} sec")
    print(f"High Noise Time: {report['audio_time']} sec")
    print("------------------------------------")
    print(f"FINAL SUSPICION SCORE: {report['final_score']}")
    print("====================================")

    logging.info("SESSION ENDED")
    logging.info(str(report))