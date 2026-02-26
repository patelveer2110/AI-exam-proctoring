from fastapi import FastAPI
from app.routers import video_router, audio_router, exam_router

app = FastAPI(title="AI Proctoring Backend")

app.include_router(video_router.router)
app.include_router(audio_router.router)
app.include_router(exam_router.router)

@app.get("/")
def root():
    return {"message": "AI Proctoring Server Running"}