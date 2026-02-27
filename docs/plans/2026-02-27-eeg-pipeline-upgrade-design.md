# EEG Pipeline Upgrade — Design Document

**Date:** 2026-02-27
**Status:** Approved

---

## Problem

Three separate issues block experiments starting tonight:

1. **Muse never connects on production** — the hardcoded default ML backend URL is a dead ngrok tunnel. Any user who hasn't manually set their URL in Settings gets a silent "Request failed (0)" error.
2. **No error visibility** — when things go wrong in production (Render cold start, BrainFlow crash, WebSocket drop), we have no way to see it. Datadog is already partially wired but not activated.
3. **Stress and Flow models are below the 60% activation threshold** — they are disabled in the live inference path. Pilot experiments will have missing stress/flow data.

---

## Goals

- Muse connects reliably on production Vercel deploy without any user setup
- Render cold-start handled gracefully (warm-up ping, not an error)
- Datadog sees every error, inference latency, and model accuracy live
- Stress model ≥ 65% CV (currently 59.64% — below threshold)
- Flow model ≥ 62% CV (currently 57% — below threshold)
- All upgrades deployed before experiments start

---

## Architecture

### Muse Connection Fix

```
Current broken path:
  Browser → getMLApiUrl() → dead ngrok URL → "Request failed (0)"

Fixed path:
  Browser → getMLApiUrl() → VITE_ML_API_URL (set in Vercel dashboard)
                            → https://neural-dream-ml.onrender.com

  On muse-pair screen load:
  → warmUpBackend() ping /health (silent, 10s timeout)
  → "ML backend is waking up…" spinner if >1s
  → Pair button enabled once /health returns 200
```

### Datadog

```
Datadog APM (already imported in main.py, just needs DD_API_KEY in Render):
  FastAPI request → ddtrace middleware → Datadog APM

Datadog Browser RUM (new, add to index.html):
  Frontend error → DD RUM SDK → Datadog dashboard

Datadog Metrics (wire existing accuracy reporting):
  Auto-retraining loop → statsd.gauge("model.accuracy", value) → Datadog
```

### Model Retraining

```
DREAMER dataset (Zenodo, consumer Emotiv EPOC — closest to Muse 2):
  ml/training/train_dreamer.py already exists → just needs download + run

TSception architecture:
  ml/training/train_tsception.py already exists → needs proper 4-channel config

Training pipeline:
  python ml/training/train_dreamer.py → saves to ml/models/saved/
  python ml/training/train_tsception.py → saves eegnet_emotion_4ch.pt
```

---

## Tech Stack

- **Frontend**: React 18, TypeScript, Vite (`VITE_ML_API_URL` env var)
- **ML Backend**: FastAPI on Render.com (free tier, Docker), ddtrace 2.x
- **Datadog**: APM (Python ddtrace), Browser RUM (JS SDK), Metrics (statsd)
- **Datasets**: DREAMER (Zenodo), FACED (already have train_faced.py), SEED-V
- **Models**: LightGBM (current), TSception CNN (4-channel, in train_tsception.py)

---

## Non-Goals

- No Datadog account setup (user does this manually — 5 min)
- No EEGPT integration (requires GPU, out of scope for tonight)
- No new dataset downloads besides DREAMER (already have train script)
- No changes to the Express/Vercel API layer

---

## Success Criteria

- `https://dream-analysis.vercel.app/study/session` → Muse pairs without errors
- Datadog dashboard shows live FastAPI traces within 5 min of deploy
- Stress model CV ≥ 65% (re-run benchmark after retraining)
- Flow model CV ≥ 62%
- Zero "Request failed (0)" errors in production Datadog

