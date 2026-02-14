from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from api.websocket import eeg_stream_endpoint

app = FastAPI(
    title="Neural Dream Workshop - ML Service",
    description="EEG signal processing, sleep staging, emotion classification, and dream detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

# WebSocket for real-time EEG streaming
app.websocket("/ws/eeg-stream")(eeg_stream_endpoint)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "neural-dream-ml"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
