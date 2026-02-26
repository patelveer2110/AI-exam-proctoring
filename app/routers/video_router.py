from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np
from app.services.face_service import detect_faces
from app.services.gaze_service import detect_gaze
from app.services.object_service import detect_objects
from app.services.suspicion_engine import calculate_score

router = APIRouter(prefix="/video", tags=["Video"])

@router.post("/analyze")
async def analyze_video_frame(file: UploadFile = File(...)):
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    face_count = detect_faces(frame)
    gaze_status = detect_gaze(frame)
    objects = detect_objects(frame)

    score = calculate_score(face_count, gaze_status, objects, "Normal")

    return {
        "faces": face_count,
        "gaze": gaze_status,
        "objects": objects,
        "suspicion_score": score
    }