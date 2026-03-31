
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import base64
import numpy as np
import cv2
import time
from concurrent.futures import ThreadPoolExecutor

from app.services.gaze_service import GazeEstimator
from app.services.headpose_service import HeadPoseEstimator
from app.services.object_service import ObjectDetector
from app.services.audio_service import AudioDetector
from app.services.face_service import FacePresenceDetector
from app.services.suspicion_engine import SuspicionEngine

router = APIRouter()

# ---------------- GLOBAL MODELS ---------------- #
gaze = GazeEstimator()
head = HeadPoseEstimator()
obj = ObjectDetector()
audio = AudioDetector()
face = FacePresenceDetector()

# 🔥 GLOBAL THREAD POOL (IMPORTANT)
executor = ThreadPoolExecutor(max_workers=4)

# ---------------- SESSION STORAGE ---------------- #
sessions = {}
warnings = {}
history = {}
last_state = {}

# ---------------- NUMPY FIX ---------------- #
def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj


# ---------------- RESET ---------------- #
@router.post("/reset")
async def reset_session(data: dict):
    user_id = data.get("user_id")

    sessions.pop(user_id, None)
    warnings.pop(user_id, None)
    history.pop(user_id, None)
    last_state.pop(user_id, None)

    return {"status": "reset"}


# ---------------- MAIN ANALYZE ---------------- #
@router.post("/analyze")
async def analyze(data: dict):
    try:
        user_id = data.get("user_id", "default_user")

        # ---------- INIT ----------
        if user_id not in sessions:
            sessions[user_id] = SuspicionEngine()
            warnings[user_id] = 0
            history[user_id] = []
            last_state[user_id] = {
                "gaze": None,
                "head": None,
                "face": None,
                "object": None
            }

        engine = sessions[user_id]

        # ---------- DECODE ----------
        img_data = base64.b64decode(data["image"])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse({"error": "Invalid frame"}, status_code=400)

        # 🔥 RESIZE → HUGE SPEED BOOST
        frame = cv2.resize(frame, (320, 240))

        current_time = time.time()

        # ---------- PARALLEL PROCESSING ----------
        future_gaze = executor.submit(gaze.detect, frame)
        future_head = executor.submit(head.detect, frame)
        future_object = executor.submit(obj.detect, frame)
        future_audio = executor.submit(audio.detect)

        gaze_data = future_gaze.result()
        head_data = future_head.result()
        object_data = future_object.result()
        audio_data = future_audio.result()

        # ---------- FACE ----------
        face_data = face.detect(head_data.get("landmarks", None))
        face_present = face_data["face_present"]

        # ---------- ENGINE ----------
        engine.update(
            face_present=face_present,
            gaze_data=gaze_data,
            object_data=object_data,
            audio_data=audio_data,
            head_data=head_data
        )

        output = engine.get_live_output()

        gaze_state = gaze_data.get("gaze", "Unknown")
        head_state = head_data.get("direction", "Unknown")
        object_detected = object_data.get("label")

        # ---------- LOGGING ----------
        def log_event(t, v):
            history[user_id].append({
                "type": t,
                "value": v,
                "timestamp": float(current_time)
            })

        if last_state[user_id]["gaze"] != gaze_state:
            log_event("gaze", gaze_state)

        if last_state[user_id]["head"] != head_state:
            log_event("head", head_state)

        if last_state[user_id]["face"] != face_present:
            log_event("face", face_present)

        if last_state[user_id]["object"] != object_detected:
            log_event("object", object_detected)

        last_state[user_id] = {
            "gaze": gaze_state,
            "head": head_state,
            "face": face_present,
            "object": object_detected
        }

        # ---------- REASON ----------
        level = output.get("level", "LOW")
        reason = None

        if level in ["HIGH", "CRITICAL"]:
            if object_detected:
                reason = "Mobile/Laptop Detected"
            elif not face_present:
                reason = "Face Not Visible"
            elif gaze_state != "Center":
                reason = "Looking Away"
            elif audio_data.get("audio_state") == "High Noise":
                reason = "Talking"
            else:
                reason = "Suspicious Behaviour"

        # ---------- WARNINGS ----------
        if level in ["HIGH", "CRITICAL"] and reason:
            warnings[user_id] += 1
        else:
            warnings[user_id] = max(0, warnings[user_id] - 1)

        auto_submit = warnings[user_id] >= 5 and output.get("score", 0) > 45

        # ---------- RESPONSE ----------
        return JSONResponse(convert_numpy({
            "score": output.get("score", 0),
            "level": level,
            "reason": reason,
            "warnings": warnings[user_id],
            "auto_submit": auto_submit,

            "gaze": gaze_state,
            "head": head_state,
            "face_present": face_present,
            "face_duration": face_data["duration"],
            "object_detected": object_detected,
            "audio": audio_data,

            "history": history[user_id][-10:]
        }))

    except Exception as e:
        print("🔥 ERROR:", str(e))
        return JSONResponse({
            "error": str(e),
            "level": "LOW",
            "score": 0
        }, status_code=200)