from ultralytics import YOLO
import time
import cv2

# ---------- LOAD LIGHT MODEL ----------
model = YOLO("yolov8n.pt")  # keep nano

SUSPICIOUS_OBJECTS = {"cell phone", "book", "laptop", "person"}


class ObjectDetector:
    def __init__(self):
        self.start_time = {}
        self.last_result = {
            "label": None,
            "max_suspicion": 0
        }
        self.last_run_time = 0

        self.DETECTION_INTERVAL = 2.0  # 🔥 run YOLO every 2 sec only

    def detect(self, frame):
        current_time = time.time()

        # ---------- ⛔ SKIP HEAVY YOLO ----------
        if current_time - self.last_run_time < self.DETECTION_INTERVAL:
            return self.last_result

        self.last_run_time = current_time

        # ---------- ⚡ RESIZE FRAME ----------
        small_frame = cv2.resize(frame, (320, 320))

        # ---------- YOLO INFERENCE ----------
        results = model(
            small_frame,
            imgsz=320,       # 🔥 smaller input
            conf=0.5,
            verbose=False,
            device="cpu"     # change to "cuda" if GPU available
        )

        detected_now = {}
        person_count = 0

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]

                if label in SUSPICIOUS_OBJECTS:
                    detected_now[label] = conf
                    if label == "person":
                        person_count += 1

        max_suspicion = 0
        detected_label = None

        # ---------- TEMPORAL LOGIC ----------
        for obj, conf in detected_now.items():
            if obj not in self.start_time:
                self.start_time[obj] = current_time

            duration = current_time - self.start_time[obj]

            suspicion = 0

            if obj == "person":
                if person_count > 1:
                    suspicion = 0.9
            else:
                if duration > 2:
                    suspicion = 0.4
                if duration > 5:
                    suspicion = 0.7

            if suspicion > max_suspicion:
                max_suspicion = suspicion
                detected_label = obj

        # ---------- CLEANUP ----------
        for obj in list(self.start_time.keys()):
            if obj not in detected_now:
                del self.start_time[obj]

        # ---------- CACHE RESULT ----------
        self.last_result = {
            "label": detected_label,
            "max_suspicion": max_suspicion
        }

        return self.last_result