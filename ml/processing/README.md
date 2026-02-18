# Processing — Signal Processing Pipeline

EEG signal preprocessing, feature extraction, and quality assessment modules.

## All Modules

| Module | Purpose |
|--------|---------|
| `eeg_processor.py` | Core signal preprocessing — bandpass filter (0.5-45 Hz), notch filter (50/60 Hz), feature extraction (17-feature vector) |
| `artifact_detector.py` | Detects eye blinks, muscle artifacts, and electrode pops in raw EEG |
| `signal_quality.py` | 5-point quality gate: amplitude, line noise, EMG contamination, stationarity, eye blinks |
| `calibration.py` | User calibration workflow — baseline recording, personal thresholds |
| `confidence_calibration.py` | Uncertainty estimation — adds reliability scores + confidence intervals to predictions |
| `connectivity.py` | Brain connectivity analysis — Granger causality, directed transfer function (DTF), graph metrics |
| `emotion_shift_detector.py` | Detects emotion transitions over time, tracks emotional awareness |
| `noise_augmentation.py` | Data augmentation for training — adds realistic noise to clean signals |
| `spiritual_energy.py` | Chakra analysis, consciousness level estimation, aura mapping from EEG patterns |
| `state_transitions.py` | Tracks brain state transitions (focus→relaxed→drowsy) using Markov model |
| `user_feedback.py` | Collects user feedback on predictions, personalizes future outputs |

## Pipeline Order

```
Raw EEG (256 Hz, n_channels x n_samples)
    │
    ├─▶ artifact_detector.py — flag/reject bad segments
    │
    ├─▶ eeg_processor.py
    │       ├─ Bandpass filter (0.5-45 Hz)
    │       ├─ Notch filter (50/60 Hz line noise)
    │       └─ Extract 17 features:
    │           5 band powers (delta, theta, alpha, beta, gamma)
    │           3 Hjorth parameters (activity, mobility, complexity)
    │           Spectral entropy
    │           4 band ratios (alpha/beta, theta/beta, etc.)
    │           2 asymmetry metrics
    │           Peak frequency
    │           Line noise ratio
    │
    ├─▶ signal_quality.py — quality score (0-5 gates)
    │
    └─▶ 17-feature vector → model.predict()
```
