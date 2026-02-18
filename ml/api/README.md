# API — FastAPI Routes

The ML backend API. All endpoints live in `routes.py` (2017 lines). WebSocket streaming in `websocket.py`.

## routes.py Line-Number Map

Use this to navigate the large file:

| Lines | Section |
|-------|---------|
| 1-50 | Imports + Pydantic models |
| 50-80 | Model initialization (all 16 models loaded here) |
| 80-230 | Request/response schemas |
| 233-348 | `/analyze-eeg` — full EEG analysis (all models) |
| 349-402 | `/simulate-eeg` — synthetic EEG generation |
| 403-487 | `/models/status`, `/models/benchmarks` |
| 489-557 | `/analyze-wavelet`, `/clean-signal` |
| 559-617 | Neurofeedback endpoints (protocols, start, evaluate, stop) |
| 619-715 | Session management (start, stop, list, trends, compare, export) |
| 716-775 | Training data collection |
| 777-825 | Calibration (basic) |
| 826-890 | Connectivity analysis + anomaly detection |
| 882-979 | Device management (list, connect, disconnect, stream) |
| 981-1031 | Dataset downloads |
| 1032-1222 | Health integration (ingest, insights, trends, export) |
| 1223-1269 | Signal quality + confidence |
| 1270-1369 | Advanced calibration (per-user) |
| 1371-1439 | State engine + feedback + personalization |
| 1441-1537 | `/analyze-eeg-accurate` — enhanced analysis |
| 1538-1725 | Spiritual analysis (chakras, aura, consciousness, etc.) |
| 1726-1793 | Emotion shift detection |
| 1795-1907 | Cognitive model endpoints (drowsiness, attention, stress, etc.) |
| 1908-2017 | Denoising + artifact classification + datasets list |

## WebSocket Protocol

`websocket.py` handles real-time EEG streaming:

```
Client connects to ws://localhost:8000/ws/eeg-stream
    ├─▶ Server sends EEG data frames as JSON
    ├─▶ Each frame: { channels: float[][], timestamp: float }
    └─▶ Client can send control messages: { action: "start" | "stop" }
```

## Adding New Endpoints

1. Find the right category section in `routes.py` using the line-number map above
2. Add your endpoint in that section (don't just append to the end)
3. Use `_numpy_safe()` to wrap any response containing numpy types
4. Add the Pydantic request model near the top of the file (lines 80-230)
