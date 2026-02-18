# Health — External Health Data Integration

Connects EEG brain data with health metrics from wearable devices and phones.

## Files

| File | Purpose |
|------|---------|
| `apple_health.py` | Imports and parses Apple Health XML exports (heart rate, sleep, steps, HRV) |
| `google_fit.py` | Google Fit REST API integration (heart rate, activity, sleep) |
| `correlation_engine.py` | Correlates health metrics with EEG patterns — finds relationships between sleep quality, heart rate, stress, and brain states |

## Supported Metrics

- Heart rate + HRV (heart rate variability)
- Sleep duration + quality
- Daily steps + activity
- Stress indicators
- Blood oxygen (SpO2)

## How It Connects

```
Apple Health / Google Fit
    │
    ├─▶ apple_health.py / google_fit.py — parse + normalize data
    │
    └─▶ correlation_engine.py — correlate with EEG sessions
            │
            └─▶ API endpoints: /health/ingest, /health/insights, /health/trends
```

Health data is ingested via the `/api/health/ingest` endpoint and correlated with brain data to generate insights like "your sleep quality drops when your stress EEG score exceeds 0.7."
