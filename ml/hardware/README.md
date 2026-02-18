# Hardware — EEG Device Management

Manages connections to physical EEG headsets via the BrainFlow library.

## Files

| File | Purpose |
|------|---------|
| `brainflow_manager.py` | BrainFlow abstraction layer — discover, connect, stream, disconnect EEG devices |

## Supported Devices

Primary target is **Muse 2** (4-channel consumer EEG), but BrainFlow supports many boards:
- Muse 2 / Muse S
- OpenBCI (Cyton, Ganglion)
- NeuroSky MindWave
- Synthetic board (for testing without hardware)

## How It Works

```
API request (/devices/connect)
    │
    └─▶ brainflow_manager.py
            ├─ Lazy-init BrainFlow (only when first connection requested)
            ├─ Create BoardShim with board_id
            ├─ Start session + streaming
            └─ Return raw EEG data to processing pipeline
```

**Important**: BrainFlow is lazy-loaded to avoid import errors on machines without it installed. The API works fine without hardware — simulation mode generates synthetic EEG data.

## API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /api/devices` | List available EEG devices |
| `POST /api/devices/connect` | Connect to a device |
| `POST /api/devices/disconnect` | Disconnect |
| `GET /api/devices/status` | Current connection status |
| `POST /api/devices/start-stream` | Start EEG data streaming |
| `POST /api/devices/stop-stream` | Stop streaming |
