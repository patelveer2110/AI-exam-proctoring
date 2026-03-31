# import uvicorn

# if __name__ == "__main__":
#     uvicorn.run("app.main:app", reload=True)
# from fastapi import FastAPI
# from app.api.routes import router

# app = FastAPI(title="AI Proctoring Backend")

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=3   # 🔥 important
    )


# gunicorn -k uvicorn.workers.UvicornWorker app.main:app --workers 3 --bind 0.0.0.0:8000