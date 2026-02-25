# ML Backend — Neural Dream Workshop

The Python machine-learning service that powers all EEG analysis. Takes raw EEG signals (or simulated data), processes them through a signal pipeline, runs 16 classification models, and returns results via a FastAPI REST + WebSocket API.

## Tech Stack

| Dependency | Purpose |
|-----------|---------|
| FastAPI | REST API + WebSocket server |
| uvicorn | ASGI server |
| scikit-learn | Feature extraction, base classifiers |
| LightGBM | Primary emotion classifier (74.21% CV, 9 datasets) |
| XGBoost | Alternative classifier |
| PyTorch | Neural network models |
| ONNX Runtime | Optimized model inference |
| NumPy / SciPy | Signal processing + math |
| BrainFlow | EEG hardware abstraction (Muse 2) |
| MNE-Python | EEG data handling (optional) |

## Quick Start

```bash
cd ml
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# API docs at http://localhost:8000/docs
```

## The 16 Models

| Model | File | What It Does (Plain English) |
|-------|------|------------------------------|
| Emotion Classifier | `emotion_classifier.py` | Detects 6 emotions (happy, sad, angry, fear, surprise, neutral) + valence/arousal. LightGBM mega, 74.21% CV (9 datasets) |
| Sleep Staging | `sleep_staging.py` | Classifies sleep into Wake, N1, N2, N3, REM stages |
| Dream Detector | `dream_detector.py` | Detects when someone is dreaming during sleep |
| Flow State Detector | `flow_state_detector.py` | Measures how "in the zone" you are (0-1 score) |
| Creativity Detector | `creativity_detector.py` | Detects creative thinking patterns from alpha/theta ratios |
| Memory Encoding | `creativity_detector.py` | Predicts how well a memory is being encoded (same file) |
| Drowsiness Detector | `drowsiness_detector.py` | Detects sleepiness from theta power increases |
| Cognitive Load | `cognitive_load_estimator.py` | Estimates mental workload (low/medium/high) |
| Attention Classifier | `attention_classifier.py` | Classifies attention level from beta/theta ratios |
| Stress Detector | `stress_detector.py` | Detects stress from beta asymmetry + heart rate |
| Lucid Dream Detector | `lucid_dream_detector.py` | Identifies gamma bursts during REM (lucid dreaming) |
| Meditation Classifier | `meditation_classifier.py` | Classifies meditation depth from alpha coherence |
| Anomaly Detector | `anomaly_detector.py` | Flags unusual EEG patterns (Isolation Forest) |
| Artifact Classifier | `artifact_classifier.py` | Identifies artifact types: eye blink, muscle, electrode |
| Denoising Autoencoder | `denoising_autoencoder.py` | Cleans noisy EEG signals (PyTorch autoencoder) |
| Online Learner | `online_learner.py` | Adapts models to individual users over time |

## How Models Load

```
_find_model("emotion_classifier_model")
    │
    ├── 1. Look for .onnx file → fastest inference
    ├── 2. Look for .pkl file  → scikit-learn/LightGBM
    ├── 3. Look for .pt file   → PyTorch
    └── 4. Fall back to feature-based heuristics (no saved model needed)
```

Models are initialized once at import time in `api/routes.py` (lines 50-80).

## API Structure

76 endpoints in `api/routes.py`, grouped by category:

| Category | Endpoints | Line Range |
|----------|-----------|------------|
| EEG Analysis | analyze-eeg, simulate-eeg, analyze-wavelet, clean-signal, analyze-eeg-accurate | 233-1537 |
| Model Status | models/status, models/benchmarks | 403-487 |
| Neurofeedback | protocols, start, evaluate, stop | 559-617 |
| Sessions | start, stop, list, trends, weekly-report, compare, get, delete, export | 619-715 |
| Data Collection | collect-training-data, collected-data/stats | 716-775 |
| Calibration | start, submit, steps, start/{id}, add-epoch, complete, status/{id} | 777-1369 |
| Feedback | feedback, correction, self-report | 802-1439 |
| Connectivity | analyze-connectivity, anomaly/set-baseline | 826-890 |
| Devices | list, connect, disconnect, status, start-stream, stop-stream | 882-979 |
| Datasets | download-deap, download-dens, list | 981-2017 |
| Health | ingest, brain-session, daily-summary, insights, trends, export-to-healthkit, supported-metrics | 1032-1222 |
| Signal Quality | signal-quality, confidence/reliability, confidence/calibrate | 1223-1269 |
| State Engine | summary, coherence | 1371-1391 |
| Personalization | status/{id} | 1425-1439 |
| Spiritual | chakras/info, chakras, meditation-depth, aura, kundalini, prana-balance, consciousness, third-eye, full-analysis | 1538-1725 |
| Emotion Shift | detect, summary, awareness-score, reset | 1726-1793 |
| Cognitive Models | drowsiness, cognitive-load, attention, stress, lucid-dream, meditation, session-stats | 1795-1907 |
| Denoising | denoise, classify-artifacts, denoise/status | 1908-1987 |

## How to Add a New Model

1. Create `models/my_model.py` with class: `__init__(model_path)` + `predict(features) → dict`
2. Train and save weights to `models/saved/my_model.pkl` (or `.onnx`)
3. Add import + initialization in `api/routes.py` using `_find_model("my_model")`
4. Add endpoint: `@router.post("/predict-my-thing")` in the relevant category section
5. Add client call in `client/src/lib/ml-api.ts`

## How Training Works

```
Raw Datasets (DEAP, SEED, GAMEEMO, Muse, etc.)
    │
    └─▶ training/data_loaders.py (8 dataset loaders)
            │
            └─▶ training/train_*.py (per-model training script)
                    │
                    ├─▶ Feature extraction (17 features)
                    ├─▶ Train/val/test split
                    ├─▶ Model training (LightGBM, XGBoost, MLP, etc.)
                    ├─▶ Evaluation → benchmarks/*.json
                    └─▶ Save → models/saved/*.pkl or *.onnx
```

Key training scripts: `mega_trainer.py` (comprehensive, all algorithms), `train_emotion.py`, `train_sleep.py`, `train_dream.py`.

## Post-Task Checklist (MANDATORY after every completed task)

After finishing any task, always do these in order — no exceptions:

1. **Push to GitHub**: `git push`
2. **Deploy to Vercel**: Vercel auto-deploys on push to `main`. If not auto-connected,
   run `vercel --prod` from the project root.
3. **Update `STATUS.md`**: Mark completed items [x], update model accuracies, add new
   endpoints or pages built.
4. **Update `PRODUCT.md`**: Update the "Honest Assessment" percentages and "What Is Broken"
   section if anything was fixed.
5. **Update benchmark dashboard** (`client/src/pages/formal-benchmarks-dashboard.tsx`):
   Update model accuracy numbers, dataset statuses, and publishing-plan checkboxes to
   reflect the latest results.

## Git Commit Rules

- **NEVER add `Co-Authored-By: Claude` (or any Claude/AI co-author line) to commit messages.** Claude must not appear as a contributor in the git history.
