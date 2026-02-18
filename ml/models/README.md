# Models

Machine learning model classes for EEG signal classification.

## All 16 Models

| Class | File | Task | Accuracy |
|-------|------|------|----------|
| `EmotionClassifier` | `emotion_classifier.py` | 6 emotions + valence/arousal | 97.79% (LightGBM) |
| `SleepStagingModel` | `sleep_staging.py` | Wake/N1/N2/N3/REM classification | — |
| `DreamDetector` | `dream_detector.py` | Dreaming vs non-dreaming | — |
| `FlowStateDetector` | `flow_state_detector.py` | Flow state score (0-1) | — |
| `CreativityDetector` | `creativity_detector.py` | Creative thinking patterns | — |
| `MemoryEncodingPredictor` | `creativity_detector.py` | Memory encoding strength | — |
| `DrowsinessDetector` | `drowsiness_detector.py` | Sleepiness detection | — |
| `CognitiveLoadEstimator` | `cognitive_load_estimator.py` | Mental workload level | — |
| `AttentionClassifier` | `attention_classifier.py` | Attention level | — |
| `StressDetector` | `stress_detector.py` | Stress detection | — |
| `LucidDreamDetector` | `lucid_dream_detector.py` | Lucid dreaming detection | — |
| `MeditationClassifier` | `meditation_classifier.py` | Meditation depth | — |
| `AnomalyDetector` | `anomaly_detector.py` | Unusual pattern flagging | — |
| `ArtifactClassifier` | `artifact_classifier.py` | Artifact type identification | — |
| `DenoisingAutoencoder` | `denoising_autoencoder.py` | Signal noise removal | — |
| `OnlineLearner` | `online_learner.py` | Per-user model adaptation | — |

## Saved Weights (`saved/`)

Pre-trained model files live in `saved/`. All are gitignored.

| File | Format | Size |
|------|--------|------|
| `emotion_classifier_model.onnx` | ONNX | Small |
| `emotion_classifier_model.pkl` | Pickle | Medium |
| `dream_detector_model.pkl` | Pickle | Medium |
| `sleep_staging_model.pkl` | Pickle | Medium |
| `flow_state_model.pkl` | Pickle | Medium |
| `creativity_model.pkl` | Pickle | Medium |
| `memory_encoding_model.pkl` | Pickle | Medium |
| `emotion_deap_muse.pkl` | Pickle | Medium |

Additional variants at the `models/` level: `emotion_classifier_lgbm.joblib` (114MB), `emotion_classifier_rf.joblib` (3.1GB, gitignored separately), `emotion_classifier_xgb.joblib`, `emotion_classifier_mlp.pt`.

## Model Loading Chain

All models use auto-discovery via `_find_model()` in `api/routes.py`:

```
ONNX (.onnx) → Pickle (.pkl) → Feature-based fallback (no file needed)
```

Every model works without saved weights — it falls back to heuristic rules using band powers and ratios. This ensures the API always returns results, even in demo mode.
