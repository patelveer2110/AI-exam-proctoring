from fastapi import APIRouter, UploadFile, File
from app.services.audio_service import detect_noise

router = APIRouter(prefix="/audio", tags=["Audio"])

@router.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):

    path = "temp.wav"

    with open(path, "wb") as f:
        f.write(await file.read())

    status = detect_noise(path)

    return {"audio_status": status}