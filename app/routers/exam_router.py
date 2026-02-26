from fastapi import APIRouter

router = APIRouter(prefix="/exam", tags=["Exam"])

@router.get("/health")
def exam_health():
    return {"status": "Exam router working"}