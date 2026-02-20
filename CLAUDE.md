# Neural Dream Workshop

A brain-computer interface (BCI) web application that reads EEG signals from a Muse 2 headband and uses 16 machine-learning models to classify emotions, detect dreams, stage sleep, measure focus, and more — all visualized in a real-time React dashboard.

Built as a full-stack system: React frontend, Express.js middleware, FastAPI ML backend, PostgreSQL database.

## Quick Start

```bash
# Frontend + Express middleware (port 5000)
npm install && npm run dev

# ML backend (port 8000)
cd ml && pip install -r requirements.txt && uvicorn main:app --reload --port 8000
```

## Architecture

```
Browser (React)
    │
    ├── REST ──▶ Express.js (:5000) ──▶ PostgreSQL (Neon)
    │               │
    │               └── /api/dream-analysis, /api/ai-chat (GPT-5)
    │
    └── REST + WS ──▶ FastAPI (:8000)
                        │
                        ├── 16 ML models (LightGBM, ONNX, PyTorch)
                        ├── EEG signal processing pipeline
                        └── BrainFlow (Muse 2 hardware)
```

## Directory Map

| Folder | What It Is |
|--------|-----------|
| `client/` | React 18 + TypeScript frontend (17 pages, shadcn/ui, Tailwind) |
| `server/` | Express.js middleware — auth, DB, AI chat, data export |
| `shared/` | Drizzle ORM schema shared between client and server |
| `ml/` | Python ML backend — 16 models, 76 API endpoints, signal processing |
| `ml/models/` | Model classes + saved weights (ONNX, pkl, joblib) |
| `ml/processing/` | EEG signal processing pipeline (11 modules) |
| `ml/training/` | Training scripts + data loaders for 8 EEG datasets |
| `ml/health/` | Apple Health + Google Fit integration |
| `ml/hardware/` | BrainFlow EEG device manager (Muse 2) |
| `ml/api/` | FastAPI routes (2K-line routes.py) + WebSocket |
| `api/` | Vercel serverless function stubs |
| `docs/` | Scientific guide (40KB reference on EEG + ML) |
| `attached_assets/` | Screenshots and generated dream images |

## Key Conventions

- **Routing**: [wouter](https://github.com/molefrog/wouter) (not react-router)
- **UI Components**: [shadcn/ui](https://ui.shadcn.com/) in `client/src/components/ui/`
- **Server State**: TanStack Query (no Redux)
- **Styling**: Tailwind CSS, dark theme by default
- **Charts**: Recharts (main) + Chart.js (some pages)
- **ML Model Loading**: Auto-discovery with fallback chain: ONNX → pkl → feature-based
- **Import Aliases**: `@/` maps to `client/src/`
- **EEG Standard**: 256 Hz sampling rate, 17-feature vectors, numpy arrays

## Environment Variables

| Variable | Used By | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | Express | Neon PostgreSQL connection string |
| `OPENAI_API_KEY` | Express | GPT-5 for dream analysis + AI chat |
| `SESSION_SECRET` | Express | Express session encryption |
| `PORT` | Express | Server port (default 5000) |

## Deployment

- **Frontend + Express**: Vercel (see `vercel.json`)
- **ML Backend**: Docker or standalone (`uvicorn main:app`)
- **Database**: Neon PostgreSQL (`drizzle-kit push` for migrations)

## Key Files

| File | Why It Matters |
|------|---------------|
| `client/src/App.tsx` | All 17 routes defined here |
| `server/routes.ts` | Express API (10 endpoints: health, dreams, chat, settings, export) |
| `ml/api/routes.py` | FastAPI ML API (76 endpoints, 2K lines — read the README) |
| `shared/schema.ts` | Database schema (7 tables: users, health, dreams, emotions, chats, settings, push) |
| `ml/main.py` | FastAPI app entry point |
| `docs/COMPLETE_SCIENTIFIC_GUIDE.md` | 40KB scientific reference on EEG signal processing + ML |

---

## EEG Science Reference (Read This Every Session)

This section documents the neuroscience, formulas, and architecture decisions for the ML pipeline. **Read this before touching any ML code.**

### Muse 2 Electrode Positions

```
Muse 2 has 4 EEG channels + 2 reference (ear clips):
  BrainFlow delivery order (board_id 22/38):
  ch0 = TP9  — Left temporal
  ch1 = AF7  — Left frontal  (key for FAA — LEFT channel)
  ch2 = AF8  — Right frontal (key for FAA — RIGHT channel)
  ch3 = TP10 — Right temporal

  ⚠ The old comment "AF7=ch0, AF8=ch1" in eeg_processor.py was WRONG.
  BrainFlow's eeg_names for Muse 2 = ["TP9", "AF7", "AF8", "TP10"].

Sampling rate: 256 Hz
Data type: float32, microvolts (µV)
```

**Critical**: AF7/AF8 sit directly over the frontalis muscle. Any jaw clenching, forehead tensing, or eyebrow raising injects EMG artifact at 20-100 Hz into these channels. Gamma power (30-100 Hz) from AF7/AF8 is predominantly **muscle noise, not neural signal**.

### Brain Wave Bands (as defined in `eeg_processor.py`)

| Band | Frequency | Mental State |
|------|-----------|-------------|
| **Delta** | 0.5 – 4 Hz | Deep sleep, unconscious processing |
| **Theta** | 4 – 8 Hz | Drowsiness, meditation, creativity, memory encoding |
| **Alpha** | 8 – 12 Hz | Relaxation, eyes-closed, calm focus, mind wandering |
| **Beta** | 12 – 30 Hz | Active thinking, concentration, anxiety, stress |
| Low-beta | 12 – 20 Hz | Task focus (non-anxious) |
| High-beta | 20 – 30 Hz | Anxiety, stress, fight-or-flight |
| **Gamma** | 30 – 100 Hz | **⚠ Mostly EMG at AF7/AF8 on Muse 2** (muscle artifact) |

### Key EEG Ratios

```python
alpha_beta_ratio   = alpha / beta       # Relaxation vs alertness
theta_beta_ratio   = theta / beta       # Creativity/drowsiness vs focus
alpha_theta_ratio  = alpha / theta      # Calm wakefulness vs drowsiness
theta_alpha_ratio  = theta / alpha      # Meditation depth
high_beta_frac     = high_beta / beta   # Anxiety fraction of total beta
delta_theta_ratio  = delta / theta      # Deep sleep marker
```

### Frontal Alpha Asymmetry (FAA) — Most Important Feature

**Davidson (1992)** established: greater relative left-frontal alpha = approach motivation & positive affect. Greater relative right-frontal alpha = withdrawal motivation & negative affect/depression risk.

**Formula:**
```python
# In eeg_processor.py: compute_frontal_asymmetry()
frontal_asymmetry = log(AF8_alpha) - log(AF7_alpha)
# = log(right_alpha) - log(left_alpha)

asymmetry_valence = clip(tanh(frontal_asymmetry * 2.0), -1, 1)
# FAA > 0 → positive valence (happy, excited)
# FAA < 0 → negative valence (sad, depressed, fearful)
```

**Why this matters**: FAA is the single most validated EEG marker for emotional valence. It has 30 years of replication. The old code was discarding all multichannel data and only using ch0 (AF7), losing all asymmetry information. This was the **#1 bug** causing wrong emotion data.

### Valence vs Arousal (Russell's Circumplex Model)

All emotions live in a 2D space:
```
              HIGH AROUSAL (+1)
              excited, angry, fearful
                      |
NEGATIVE ─────────────┼───────────── POSITIVE
VALENCE (-1)          |           VALENCE (+1)
              sad, bored, depressed
              LOW AROUSAL (-1)
```

- **Valence** (positive/negative feeling): best predicted by FAA + alpha/beta ratio
- **Arousal** (energetic/calm): best predicted by beta/(alpha+beta) ratio + theta
- EEG predicts arousal better (70-85% accuracy) than valence (60-75%)

### Signal Processing Pipeline

```
Raw Muse 2 EEG (256 Hz, 4 channels, µV)
    │
    ▼ preprocess() — ml/processing/eeg_processor.py
    ├── Butterworth bandpass: 1-50 Hz, order=5 (filtfilt, zero-phase)
    ├── Notch filter: 50 Hz (European mains noise)
    ├── Notch filter: 60 Hz (US mains noise)
    │
    ▼ extract_band_powers() — Welch PSD
    ├── delta, theta, alpha, beta, low_beta, high_beta, gamma
    ├── Log-domain powers (better normality for ML)
    ├── Hjorth parameters (activity, mobility, complexity)
    ├── Spectral entropy
    ├── Differential entropy per band
    └── Key ratios (alpha/beta, theta/beta, etc.)
    │
    ▼ compute_frontal_asymmetry() — FAA
    ├── Per-channel alpha power (AF7 and AF8)
    ├── FAA = log(right_alpha) - log(left_alpha)
    └── asymmetry_valence = tanh(FAA * 2.0)
```

### Emotion Classifier Inference Chain

`ml/models/emotion_classifier.py` — **the most important ML file**

```
predict(eeg, fs=256)  ← receives (4, n_samples) array from Muse 2
    │
    ├── If DEAP ONNX model loaded AND accuracy ≥ 60%:
    │       → _predict_onnx()
    │
    ├── If DEAP sklearn model loaded AND accuracy ≥ 60%:
    │       → _predict_sklearn()
    │
    ├── If Muse-trained model available:
    │       → _predict_multichannel()
    │
    └── Else (DEFAULT for live Muse 2 — DEAP model is 51.3%, below threshold):
            → _predict_features()   ← THIS IS THE LIVE PATH
```

**DEAP model accuracy**: 51.3% — below the 60% threshold, so it falls back to `_predict_features()` (feature-based heuristics) for all live Muse 2 sessions. **The feature-based path is the production path.**

### Emotion Output Structure

```python
{
  "emotion": "happy"|"sad"|"angry"|"fear"|"surprise"|"neutral",
  "probabilities": {
      "happy": 0.0-1.0,
      "sad": 0.0-1.0,
      "angry": 0.0-1.0,
      "fear": 0.0-1.0,
      "surprise": 0.0-1.0,
      "neutral": 0.0-1.0
  },
  "valence": -1.0 to 1.0,      # negative ↔ positive feeling
  "arousal": 0.0 to 1.0,       # calm ↔ energetic
  "stress_index": 0.0-1.0,
  "focus_index": 0.0-1.0,
  "relaxation_index": 0.0-1.0,
  "anger_index": 0.0-1.0,
  "frontal_asymmetry": float,   # FAA raw value
  "temporal_asymmetry": float,
  "model_type": "feature-based"|"onnx"|"sklearn"|"multichannel"
}
```

---

## ML Pipeline Bug Fixes (Feb 2026)

These bugs were discovered via deep research and fixed in commit `270d27e`. **Do not revert these changes.**

### Bug 1: Multichannel Data Stripped Before FAA Computation (CRITICAL)

**File**: `ml/models/emotion_classifier.py`, `predict()`, line ~133

**Old code** (broken):
```python
return self._predict_features(eeg if eeg.ndim == 1 else eeg[0], fs)
# ↑ Always passed only AF7 (ch0), lost AF8/TP9/TP10 forever
```

**Fixed code**:
```python
return self._predict_features(eeg, fs)  # full multichannel array
```

**Inside `_predict_features()`**, added at the start:
```python
channels = eeg if eeg.ndim == 2 else None
signal = eeg[0] if eeg.ndim == 2 else eeg   # AF7 for band powers
processed = preprocess(signal, fs)
```

**Impact**: Without this fix, FAA was always zero. Every valence reading was based solely on band power ratios from AF7, making the system behave as if it had only one electrode.

### Bug 2: Gamma Used as Neural Signal (EMG Contamination)

**Old formulas** (wrong):
```python
arousal_raw = 0.35*β/(β+α) + 0.30*(1-α/(α+β+θ)) + 0.35*γ
stress_index = 0.40*high_β/(β+α) + 0.35*(θ/α)*0.3 + 0.25*γ
focus_index  = 0.35*β_ratio + 0.40*(1-α_ratio) + 0.25*γ
angry prob   = ... + 0.15*min(1, γ*2.5)
```

**Fixed formulas** (gamma removed or near-zero):
```python
# Arousal — no gamma, uses delta as inverse relaxation signal:
arousal_raw = 0.45*β/(β+α) + 0.30*(1-α/(α+β+θ)) + 0.25*(1-δ/(δ+β))

# Stress — no gamma:
stress_index = 0.45*high_β/(β+α) + 0.30*(θ/α)*0.3 + 0.25*high_β_frac

# Focus — no gamma:
focus_index = 0.45*β_ratio + 0.40*(1-α_ratio) + 0.15*(1-θ_β_ratio*0.2)

# Angry — gamma reduced from 15% to 5%:
probs[2] = 0.40*(neg_val) + 0.30*(arousal-0.45) + 0.25*beta_alpha + 0.05*min(1,γ*2.5)
```

**Impact**: Jaw clenching was falsely registering as high arousal, stress, anger, and focus.

### Bug 3: Sad Threshold Too High — Almost Never Triggered

**Old**:
```python
probs[1] = 0.50 * max(0, -valence - 0.25) + ...
# Required valence < -0.25 before any sad probability
```

**Fixed**:
```python
probs[1] = 0.50 * max(0, -valence - 0.10) + ...
# Triggers as soon as valence goes slightly negative
```

**Impact**: At resting state with neutral FAA (valence ≈ 0), sad was essentially impossible to detect.

### Bug 4: Duplicate Arousal in `_predict_onnx` (Silent Override)

In `_predict_onnx()`, lines 509 and 512 both computed `arousal` identically — the second silently overwrote the first with the same (wrong gamma-based) formula. Fixed to single corrected computation.

### Bug 5: Single-Term Valence in ONNX/Sklearn Paths

**Old** (one term):
```python
valence = tanh((alpha/beta - 0.7) * 2.0)
```

**Fixed** (two terms — alpha/beta ratio + absolute alpha level):
```python
valence = 0.65 * tanh((alpha/beta - 0.7) * 2.0) + 0.35 * tanh((alpha - 0.15) * 4)
```

**Impact**: Without the absolute alpha level term, the formula could give positive valence even with very low overall alpha (i.e., stressed state with slightly more alpha than beta).

### FAA Integration (New Feature)

**Added to `_predict_features()`**:
```python
faa_valence = 0.0
if channels is not None and channels.shape[0] >= 2:
    # BrainFlow Muse 2: ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10
    asym = compute_frontal_asymmetry(channels, fs, left_ch=1, right_ch=2)
    faa_valence = asym.get("asymmetry_valence", 0.0)

# Valence: 50% alpha/beta ratio + 50% FAA (when multichannel)
if channels is not None and channels.shape[0] >= 2:
    valence = clip(0.50 * valence_abr + 0.50 * faa_valence, -1, 1)
else:
    valence = clip(valence_abr, -1, 1)
```

---

## Model Accuracy Reality Check

| Model | File | CV Accuracy | Notes |
|-------|------|------------|-------|
| Emotion LGBM (mega-trained) | `models/emotion_classifier_lgbm.joblib` (109 MB) | **97.79%** | 3-class, PCA-88-dim features — NOT loaded by current code |
| Emotion MLP (PyTorch) | `models/emotion_classifier_mlp.pt` (2.5 MB) | **93.11%** | 3-class, PCA-88-dim features — NOT loaded by current code |
| Emotion XGBoost | `models/emotion_classifier_xgb.joblib` (54 MB) | ~88% | 3-class, PCA-88-dim features — NOT loaded |
| Emotion (DEAP pkl) | `models/saved/emotion_classifier_model.pkl` | 51.3% | Below 60% threshold → disabled |
| Emotion (ONNX) | `models/saved/emotion_classifier_model.onnx` | ~51% | Below 60% threshold → disabled |
| Sleep Staging | `models/saved/sleep_staging_model.pkl` | 92.98% | Active, reliable |
| Dream Detector | `models/saved/dream_detector_model.pkl` | 97.20% | Active, reliable |
| Flow State | `models/saved/flow_state_model.pkl` | 62.86% | Active, marginal |
| Creativity | `models/saved/creativity_model.pkl` | 99.18% | Likely overfit (850 samples) |

### CRITICAL: Why the 97.79% LGBM Model Is NOT Used

The high-accuracy models (`emotion_classifier_lgbm.joblib`, `emotion_classifier_mlp.pt`) are in `ml/models/` but **are not loaded by `emotion_classifier.py`** because:

1. **Wrong output class count**: These models output 3 classes (positive/neutral/negative), but `emotion_classifier.py` expects 6 classes (happy/sad/angry/fearful/relaxed/focused)
2. **Wrong feature format**: These models need **88-dim PCA-aligned features** (80 PCA components + 8 one-hot dataset indicators), not the 17 raw band-power features the system currently extracts
3. **Wrong filename**: `_find_model("emotion_classifier_model")` looks for files named `emotion_classifier_model.*`, not `emotion_classifier_lgbm.*`

### How to Enable the 97.79% Model (Future Work)

Would require:
1. Add a feature transformation step: raw 17 features → load saved PCA transform → 80-dim vector + dataset one-hot
2. Add a 3→6 class expansion or change system to 3-class output
3. Add the model path to `_find_model()` discovery chain
4. Add a benchmark JSON for it at `benchmarks/` so accuracy threshold check passes

**Datasets used by 97.79% LGBM**: GAMEEMO, EEG-ER, SEED, Brainwave, Muse-Mental, Muse-Subconscious, SEED-IV, STEW (123,234 samples total). Note: SEED achieves 100% accuracy on its own (likely pre-extracted features + within-dataset perfect separability) — this inflates the combined accuracy.

**Why emotion accuracy is low on Muse 2**:
- **Domain gap**: DEAP dataset uses 32-channel gel electrodes. Muse 2 has 4-channel dry electrodes. ~30 point accuracy penalty.
- **Cross-subject gap**: Within-subject 90.97% vs cross-subject 64.82% (-26 points).
- **Solution**: Feature-based heuristics calibrated to Muse 2 anatomy (what we use).

**Expected real-world accuracy with heuristics**:
- Arousal (calm vs energetic): 70-80%
- Valence (positive vs negative): 60-70%
- Specific emotions (happy/sad/angry): 55-65%

---

## Known Issues & Future Work

### Issues Fixed
- [x] Multichannel data stripped before FAA (commit `270d27e`)
- [x] Gamma EMG noise inflating arousal/stress/focus/anger
- [x] Sad never triggered (threshold -0.25 → -0.10)
- [x] Duplicate arousal in `_predict_onnx`
- [x] Single-term valence in ONNX/sklearn paths
- [x] Wrong FAA channel indices (commit `e7820e6`): fixed to `left_ch=1, right_ch=2`; fixed temporal asymmetry to ch0=TP9
- [x] DASM/RASM features implemented (`compute_dasm_rasm()`) and wired into valence blend (commit `0066254`)
- [x] Artifact rejection in `_predict_features()` — 75 µV threshold, frozen EMA on bad epoch (commit `0066254`)
- [x] `rereference_to_mastoid()` utility — corrects Fpz contamination at AF7/AF8 (commit `0066254`)
- [x] `compute_frontal_midline_theta()` — FMT power/DE/amplitude, more reference-robust than FAA (commit `0066254`)
- [x] `BaselineCalibrator` class — +15-29% accuracy, z-score normalization against resting state (commit `0066254`)
- [x] **Mastoid re-reference wired into live BrainFlow stream** — `brainflow_manager.get_current_data()` now applies `rereference_to_mastoid()` for all Muse devices before data reaches the classifier (commit `c675edd`)
- [x] **4-second sliding epoch buffer in `/analyze-eeg`** — `_EpochBuffer` accumulates frames into 4-sec windows (50% overlap, 2-sec hop); response includes `epoch_ready` flag (commit `19e74d7`)
- [x] **BaselineCalibrator exposed via API** — three endpoints: `POST /calibration/baseline/add-frame`, `GET /calibration/baseline/status`, `POST /calibration/baseline/reset` (commit `87e0c56`)
- [x] **FMT added to emotion output** — `compute_frontal_midline_theta()` called on AF7 channel and returned as `frontal_midline_theta` key in all `_predict_features()` return paths (commit `378af43`)

### Known Remaining Issues
- [ ] **97.79% LGBM model not integrated**: `ml/models/emotion_classifier_lgbm.joblib` exists but requires 3→6 class mapping and PCA feature transform to use. See "CRITICAL" section above.
- [x] ~~**No baseline calibration**~~: ✅ Fixed — `BaselineCalibrator` API endpoints added (commit `87e0c56`). Call `POST /calibration/baseline/add-frame` once per second during 2-min resting state; `is_ready=true` after 30 frames.
- [x] ~~**1-second epoch too short**~~: ✅ Fixed — 4-second sliding window buffer added to `/analyze-eeg` (commit `19e74d7`). `epoch_ready=false` in response until 4 seconds buffered.
- [ ] **No personalization**: The feature-based heuristics use population-average thresholds. Individual users have very different baselines (within-subject vs cross-subject gap: 26 points).
- [ ] **EMG at TP9/TP10**: Temporal channels are also near muscles. Artifact rejection/ICA would help.
- [ ] **Creativity/Memory models likely overfit**: 850 samples × 4 classes → ~212 per class is too few for reliable generalization.
- [ ] **Large models in wrong directory**: `emotion_classifier_rf.joblib` (3.1 GB) and others are in `ml/models/`, not `ml/models/saved/`. The `_find_model()` function only searches `ml/models/saved/`. No memory risk since they aren't loaded.

### Proposed Future Improvements (Priority Order)
1. ~~**Baseline protocol**~~: ✅ Done — `BaselineCalibrator` class in `eeg_processor.py` (commit `0066254`) + API endpoints (commit `87e0c56`). Call `add-frame` during 2-min resting state; normalize features before classification. +15-29% accuracy.
2. ~~**Longer epochs**~~: ✅ Done — 4-second sliding window buffer in `/analyze-eeg` (commit `19e74d7`). Slides every 2 sec (50% overlap). Response includes `epoch_ready` flag.
3. ~~**Mastoid re-reference in live stream**~~: ✅ Done — `brainflow_manager.get_current_data()` now applies mastoid re-referencing for all Muse devices (commit `c675edd`).
4. ~~**FMT in emotion output**~~: ✅ Done — `frontal_midline_theta` key added to all `_predict_features()` return paths (commit `378af43`).
5. **Personal calibration**: After 5 sessions, compute per-user band-power priors and adjust thresholds.
6. **Online learning**: `online_learner.py` exists but needs integration into the live inference path.
7. ~~**DASM/RASM features**~~: ✅ Done — `compute_dasm_rasm()` in `eeg_processor.py` + wired into `_predict_features()` valence (commit `0066254`).
8. **DREAMER dataset + retrain**: Download from Zenodo (free, no registration). Feature pipeline is now ready (DASM/RASM + FMT + mastoid reref all in place). Expected to close the domain gap vs DEAP.

---

## EEG Science: The 16 Models Explained

### What Each Model Actually Measures

**Emotion Classifier** (`emotion_classifier.py`)
- 6 discrete emotions + valence + arousal + stress/focus/relaxation indices
- Primary signals: FAA (valence), beta/alpha ratio (arousal/stress), theta/beta ratio (calm)
- Live path: feature-based heuristics (DEAP model below accuracy threshold)

**Sleep Staging** (`sleep_staging.py`)
- 5 stages: Wake, N1 (light sleep), N2 (sleep spindles/K-complexes), N3 (deep/slow-wave), REM
- Primary signals: delta dominates N3, theta in REM, alpha spindles in N1/N2
- Accuracy: 92.98% (ISRUC dataset)

**Dream Detector** (`dream_detector.py`)
- Binary: dreaming / not-dreaming
- Primary signals: REM sleep + theta oscillations + REMs detected via EOG-like artifact
- Accuracy: 97.20%

**Flow State Detector** (`flow_state_detector.py`)
- Measures "in the zone" state (0-1 score)
- Primary signals: alpha/theta coherence, moderate beta (not too high = not anxious)
- Research basis: Csikszentmihalyi's flow theory applied to EEG

**Creativity Detector** (`creativity_detector.py`)
- Alpha/theta power increases → divergent thinking / creativity
- Research: theta during incubation, alpha during insight (Kounios & Beeman, 2014)

**Drowsiness Detector** (`drowsiness_detector.py`)
- Theta power increase + alpha slowing + slow eye movements
- Research: PERCLOS metric + EEG theta (Lal & Craig, 2002)

**Cognitive Load Estimator** (`cognitive_load_estimator.py`)
- 3 levels: low/medium/high mental workload
- Primary signal: frontal theta increase (working memory load)
- Research: theta-gamma coupling (Lisman & Jensen, 2013)

**Attention Classifier** (`attention_classifier.py`)
- Beta/theta ratio → focused attention
- High beta + low theta = high attention; alpha suppression = task engagement

**Stress Detector** (`stress_detector.py`)
- 4 levels: relaxed / mild / moderate / high
- Primary signals: high-beta (20-30 Hz), right > left frontal alpha (Davidson asymmetry)
- Research: Al-Shargie et al. (2016), Giannakakis et al. (2019)

**Lucid Dream Detector** (`lucid_dream_detector.py`)
- Gamma bursts (40 Hz) during REM = lucid awareness
- Research: Voss et al. (2009) — first controlled lucid dream EEG study

**Meditation Classifier** (`meditation_classifier.py`)
- 5 depths: surface / light / moderate / deep / transcendent
- Deep: theta dominance, minimal beta
- Transcendent: gamma bursts + theta-gamma coupling
- Research: Lutz et al. (2004), advanced meditators show 25× more gamma

**Anomaly Detector** (`anomaly_detector.py`)
- Isolation Forest — flags statistically unusual EEG patterns
- Used for: detecting artifacts, unusual brain states, hardware disconnection

**Artifact Classifier** (`artifact_classifier.py`)
- Classifies: eye blink (large delta spike at Fp), muscle (broadband), electrode pop
- Important for data quality scoring

**Denoising Autoencoder** (`denoising_autoencoder.py`)
- PyTorch autoencoder — reconstructs clean signal from noisy input
- Trained on paired clean/noisy EEG

**Online Learner** (`online_learner.py`)
- Adapts model weights to individual users over time
- Partially integrated — needs wiring into live inference path

---

## Processing Utilities Reference

All in `ml/processing/eeg_processor.py`:

```python
preprocess(signal, fs)                       # Bandpass 1-50 Hz + notch 50/60 Hz
rereference_to_mastoid(signals)              # Re-reference to linked TP9+TP10 (corrects Fpz contamination)
extract_band_powers(signal, fs)              # delta/theta/alpha/beta/gamma powers
extract_features(signal, fs)                 # 17 features: powers + ratios + Hjorth + entropy
extract_features_multichannel(signals, fs)   # Average features across all 4 channels + DASM/RASM
compute_frontal_asymmetry(signals, fs)       # FAA (AF7 vs AF8)
compute_dasm_rasm(signals, fs)               # DASM + RASM: 10 asymmetry features across all bands (AF7 vs AF8)
compute_frontal_midline_theta(signal, fs)    # FMT power/DE/amplitude — reference-robust valence complement
BaselineCalibrator                           # Class: collect resting baseline, normalize live features (+15-29% accuracy)
compute_coherence(signals, fs, band)         # Mean inter-channel coherence in a band
compute_phase_locking_value(signals, fs)     # PLV across channel pairs
compute_cwt_spectrogram(signal, fs)          # Morlet wavelet time-frequency map
compute_dwt_features(signal, fs)             # DWT band energy decomposition
detect_sleep_spindles(signal, fs)            # 11-16 Hz bursts → N2 sleep
detect_k_complexes(signal, fs)              # Slow high-amplitude waves → N2 sleep
spectral_entropy(signal, fs)                 # Normalized spectral entropy
differential_entropy(signal, fs)             # Per-band differential entropy
```

---

## Muse 2 Hardware & BrainFlow Reference

From deep hardware + BrainFlow research. Read before touching `ml/hardware/brainflow_manager.py`.

### Hardware Specs

| Spec | Value | Notes |
|------|-------|-------|
| Sampling rate | 256 Hz | All 4 channels simultaneous |
| ADC resolution | **12-bit** | vs. 24-bit ADS1299 in research-grade EEG |
| LSB resolution | ~0.41 µV/bit | Estimated from 0–1682 µV raw range |
| Chip | PIC24 + OPA4374 opamp | No ADS1299 — explains 24-bit vs 12-bit gap |
| Reference electrode | **Fpz (forehead center)** | Critically close to AF7/AF8 — see below |
| AF7/AF8 material | Silver-coated flat ink | Higher impedance variability than gel |
| TP9/TP10 material | Conductive silicone rubber | More stable contact, but near temporalis muscle |
| Effective bandwidth | 0.5–50 Hz (hardware limited) | No hardware notch filter |
| Bluetooth lag | ~40 ms (±20 ms) | Consistent across validation studies |
| Additional sensors (Muse 2 only) | PPG, accelerometer (3-axis), gyroscope (3-axis) | Accessible via BrainFlow ANCILLARY preset |

### Critical Hardware Limitations

**TP9/TP10 temporal channels fail frequently**: A 2024 validation study (PMC11679099, Badolato et al.) found temporal site signals were entirely non-physiological in 17/N recordings — even when all quality indicators showed green. **Do not trust TP9/TP10 data for emotion-relevant signals without verifying signal quality independently.**

**Muse vs. research-grade alpha-band correlation**: Only r=0.46–0.47 — and these are the EMG-contaminated high-beta/gamma bands. True alpha-band correlation is not reported as significant, meaning Muse's frontal EEG does not reliably track the same alpha as research-grade systems at AF7/AF8.

**Broadband power elevation**: Muse records significantly higher broadband power than research-grade EEG across ALL frequency bands. Absolute power values from Muse cannot be compared to DEAP/SEED values — **baseline normalization is mandatory** before any cross-session or cross-device ML.

**No impedance measurement API**: Unlike research-grade systems, there is no impedance meter. The HSI (Horseshoe Indicator: 1=Good, 2=Medium, 4=Bad) is unreliable — studies found non-physiological signals with all-green indicators.

### BrainFlow Board IDs

```python
MUSE_2_BOARD = 38       # Native Bluetooth (SimpleBLE) — standard
MUSE_2_BLED_BOARD = ?   # Requires Silicon Labs BLED112 USB dongle (more stable)
```

**BLED112 USB dongle** significantly reduces packet dropout vs native OS Bluetooth. In noisy RF environments, native BT dropout rate can be 1–5%.

### BrainFlow Version Requirements

| Issue | Affected Versions | Fix |
|-------|------------------|-----|
| macOS BLE scanning failure | macOS 12.0–12.2 | Upgrade to macOS 12.3+ |
| Windows 11 + Python 3.9 deadlock | Python 3.9.6 | Use Python 3.10.5+ |
| Linux/macOS connection failures | BrainFlow < 4.9.3 | Upgrade to 4.9.3+ |
| Firmware mismatch (write_request vs write_command) | After Muse FW updates | Check FW version before session |

### Recommended BrainFlow Preprocessing Chain

```python
from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes, DetrendOperations

# ORDER MATTERS: detrend → notch → bandpass → (optional wavelet denoising)

# Step 1: Remove DC offset and linear drift
DataFilter.detrend(eeg_channel, DetrendOperations.CONSTANT.value)

# Step 2: Notch filter FIRST (before bandpass)
DataFilter.remove_environmental_noise(eeg_channel, 256, NoiseTypes.SIXTY.value)  # NA
# DataFilter.remove_environmental_noise(eeg_channel, 256, NoiseTypes.FIFTY.value)  # EU

# Step 3: Bandpass
DataFilter.perform_bandpass(
    eeg_channel, 256,
    start_freq=1.0,    # high-pass: preserves delta/theta (critical for emotion)
    stop_freq=45.0,    # low-pass: cuts EMG, safely below Nyquist (128 Hz)
    order=4,
    filter_type=FilterTypes.BUTTERWORTH_ZERO_PHASE.value,  # zero-phase = no phase distortion
    ripple=0
)
```

**Current `eeg_processor.py` uses SciPy (filtfilt + butter)** — equivalent to BrainFlow's BUTTERWORTH_ZERO_PHASE. Both are zero-phase and correct. BrainFlow's API is preferred if processing in `brainflow_manager.py` before data enters Python.

**Key filter rules**:
- Never set high-pass above 2 Hz for emotion work — delta (1-4 Hz) and theta (4-8 Hz) are attenuated
- Low-pass at 45 Hz (not 50 Hz) — safety margin below Nyquist and avoids 50 Hz line noise fundamental
- Order 4 Butterworth is community standard; higher order increases ringing at cutoffs
- Always zero-phase (filtfilt / BUTTERWORTH_ZERO_PHASE) — causal filters distort phase relationships

### The Fpz Reference Problem in Detail

Fpz sits between AF7 and AF8. Referencing to it causes near-cancellation of the frontal EEG signal:

```
AF7_measured = AF7_true - Fpz_ref  ≈ AF7_true - (AF7_true + AF8_true)/2
             = (AF7_true - AF8_true) / 2   ← attenuated, asymmetry-amplified
```

This means: (1) absolute power is suppressed, (2) FAA computation may look artificially symmetric or anti-phasic.

**Fix**: Re-reference offline — average TP9 and TP10 as linked mastoid, subtract from AF7/AF8:
```python
mastoid_ref = (TP9 + TP10) / 2
AF7_reref = AF7 - mastoid_ref
AF8_reref = AF8 - mastoid_ref
FAA = np.log(alpha_power(AF8_reref)) - np.log(alpha_power(AF7_reref))
```

**Check**: Look in `brainflow_manager.py` to confirm what reference BrainFlow applies to Muse 2 data before it reaches Python. If BrainFlow delivers Fpz-referenced data, re-referencing must be added to the preprocessing pipeline.

### Enabling 5th EEG Channel and PPG

```python
board.config_board("p50")  # Enables 5th EEG channel + PPG ANCILLARY preset simultaneously
```

Note: Cannot enable 5th channel without also enabling PPG — increases Bluetooth traffic, potentially increasing dropout.

### Artifact Rejection Thresholds (Paper-Validated)

```python
# From Krigolson 2021 (most used Muse preprocessing paper):
AMPLITUDE_THRESHOLD = 75  # µV — reject epoch if any channel exceeds this
GRADIENT_THRESHOLD = 10   # µV/ms — reject if consecutive samples change > 10 µV/ms

# Conservative (less rejection, more noise):
AMPLITUDE_THRESHOLD = 100  # µV — used in some other studies

# Discard first 60-120 seconds of every session (electrode settling / DC offset stabilization)
SETTLING_TIME_SECONDS = 120
```

### BrainFlow Channel Index Reference

```python
# For MUSE_2_BOARD (board_id=38):
eeg_channels = BoardShim.get_eeg_channels(BoardIds.MUSE_2_BOARD.value)
# Returns indices for: [TP9, AF7, AF8, TP10]  → [ch0, ch1, ch2, ch3]

# Confirmed correct channel order (from brainflow_manager.py):
# ch0 = TP9   (left temporal)
# ch1 = AF7   (left frontal)   ← FAA left channel
# ch2 = AF8   (right frontal)  ← FAA right channel
# ch3 = TP10  (right temporal)
```

---

## 2024-2025 Research Findings (For Future Improvements)

From research agent survey of 2024-2025 literature.

### Why DEAP Models Fail in Real-World Deployment (10 Reasons)

1. Only 32 subjects — deep models massively overfit
2. Class imbalance (high arousal 4:1 over low arousal)
3. Lab-only controlled stimuli (music videos)
4. 32-channel gel EEG — channels like Pz/Oz/T7/T8 don't exist in Muse
5. Gel electrode SNR ≠ dry electrode SNR
6. EOG artifacts pre-removed in DEAP; raw Muse 2 has them
7. Music-only stimuli don't generalize to other emotion induction contexts
8. Single session per subject (no day-to-day variability)
9. Self-report labels reflect recalled affect, not instantaneous emotion
10. Western/young/student population (Queen Mary University London)

**Result**: 98% DEAP subject-dependent → 60-70% on unseen subjects in real world.

### Better Datasets to Train On (Priority Order for Muse 2)

| Dataset | Subjects | Channels | Device | Download | Why Use |
|---------|---------|---------|--------|---------|---------|
| DREAMER | 23 | 14 | Emotiv EPOC (consumer!) | Free, Zenodo | Most similar to Muse 2 hardware |
| FACED (2023) | 123 | 32 | Research grade | Free, Synapse account | Largest, 9-class fine-grained labels |
| SEED-V | 20 | 62 | Research grade | Free, sign license | 5 emotions, multimodal |
| EAV (2024) | 42 | 30 | Research grade | Free, GitHub | Conversational emotion, 5 classes |

**DREAMER download**: https://zenodo.org/records/546113 (no registration needed)

### Better Feature Set for 4-Channel Muse 2

Current system uses: raw band powers (17 features/channel) — reasonable but not optimal.

**Recommended 31-feature compact vector** (from 2024 literature):
```
5 DE (Differential Entropy) × 4 channels  = 20 features
5 DASM (DE_AF8 - DE_AF7 per band)          =  5 features
5 RASM (DE_AF8 / DE_AF7 per band)          =  5 features
1 FAA  (ln(AF8_alpha) - ln(AF7_alpha))     =  1 feature
Total: 31 features
```

**DASM and RASM** extend FAA to ALL frequency bands, not just alpha. ✅ **Implemented in commit `9c4e8fc`** via `compute_dasm_rasm()` in `eeg_processor.py`. Automatically included in `extract_features_multichannel()` output (total features: 31 → 41).

**PubMed 2024 finding on Muse**: TP10 electrode alone achieved 91.42% accuracy. AF8+TP9+TP10 combination reached 88.05% average across participants.

### Best Architectures for 4-Channel Consumer EEG

| Model | Why Good for Muse | Expected Acc |
|-------|-----------------|-------------|
| **TSception** | Asymmetry-aware spatial convolutions, works with few channels | 85-92% |
| EEGNet | Compact (~few thousand params), edge-deployable | 79-85% |
| 1D-CNN + LSTM | Good temporal modeling, simple to implement | 80-87% |
| LightGBM on DE features | Interpretable, proven, fast inference | 80-88% |

**TSception** is specifically designed for left/right hemisphere asymmetry (like FAA), making it ideal for Muse 2's AF7/AF8 setup.

### Realistic Accuracy Targets for Muse 2

- Binary valence/arousal, cross-subject: **75-85%**
- After 5-10 min per-user fine-tuning: **85-92%**
- Do NOT compare against DEAP subject-dependent numbers (98%+) — meaningless for deployment

### Recommended Training Strategy

```
1. Pre-train on FACED (123 subjects, diverse) OR SEED (large, pre-computed DE)
2. Domain-adapt on DREAMER (14-ch Emotiv → closest to 4-ch consumer)
3. Fine-tune on 5-10 min of labeled user data (few-shot)
```

### FAA Deep Dive — What the Research Actually Shows (2021-2025)

From deep research across 30 papers. Read before modifying FAA-related code.

#### Three FAA Formulas (All Used in Literature)

```python
# 1. Canonical (most common — recommended)
FAA_ln = ln(F4_alpha) - ln(F3_alpha)        # log-transform FIRST, then subtract

# 2. Ratio-normalized
FAA_ratio = (F4 - F3) / (F3 + F4)

# 3. Raw difference (not recommended — no log normalization)
FAA_diff = F4 - F3
```

Positive FAA = more right-frontal alpha = more left-frontal activation = approach motivation / positive affect. Current system uses formula #1 (correct).

**For Muse 2**: Replace F3/F4 with AF7/AF8. The system correctly uses `ln(AF8_alpha) - ln(AF7_alpha)`.

#### FAA Measures Approach Motivation, NOT Pure Valence

**Critical theoretical revision (Harmon-Jones et al.)**: FAA indexes approach/withdrawal motivation direction, not valence. Anger — negative valence but approach-motivated — produces LEFT frontal activation (same as happiness). This breaks the simple "FAA = positive/negative" mapping. The system using FAA as a valence proxy is a reasonable approximation but not theoretically clean.

Approach emotions (left FAA): happiness, excitement, anger
Withdrawal emotions (right FAA): fear, sadness, disgust, depression

#### The FAA Reliability Problem — Numbers

| Reliability Measure | Value |
|--------------------|-|
| Test-retest (3 weeks) | r = 0.53-0.66 |
| Test-retest (3 months) | r = 0.61 |
| Frontolateral sites (F7/F8) | r = 0.63-0.73 |
| Frontomedial sites (F3/F4) | r = 0.30-0.45 |
| Variance that is stable trait | 60% |
| Variance that is state/noise | 40% |

**Even in ideal conditions, 40% of what you measure on any day is noise.** This is why session-to-session FAA readings jump around even without emotional state change.

#### Minimum Epochs for Reliable FAA

**100 artifact-free epochs (1-3 minutes minimum)** — from updated Coan & Allen primer (2019).

This means: **a 1-2 second window FAA is essentially noise.** Epoch-to-epoch FAA values show near-zero reliability. To get a stable FAA estimate, you must average across 100 artifact-free epochs. The current system computes FAA per-frame — treat these values as directional indicators only, never absolute truth.

#### Resting-State vs Task-Evoked FAA

| Type | Variance from individual differences | Use Case |
|------|-------------------------------------|---------|
| Resting-state FAA | **16%** | Depression biomarker research |
| Task-evoked (anger induction) | **72%** | State detection |
| Task-evoked (fear) | **88%** | State detection |
| Task-evoked (sadness) | **91%** | State detection |
| Task-evoked (happiness) | **41%** | State detection |

Task-evoked FAA during emotional stimuli is far more sensitive than resting-state. The "Capability Model" predicts EEG asymmetry during emotional challenge is a more powerful indicator than resting activity. For real-time monitoring without known stimulus timing, task-evoked advantage cannot be used.

#### The Muse Fpz Reference Problem (Critical)

**Muse 2 default reference is near Fpz** (forehead midline) — **not mastoid**. Fpz is ~2-3 cm from AF7/AF8. Referencing to a nearby electrode subtracts spatially correlated neural signal → **artificially low amplitude at AF7/AF8 under default settings**.

**Fix**: Re-reference AF7/AF8 to linked TP9/TP10 (mastoid-equivalent). This is the "mastoid-ref montage" from Cannard et al. (2021) and is required for meaningful FAA values. The BrainFlow API likely delivers data already referenced — check `brainflow_manager.py` to confirm which reference is applied before signal reaches Python.

**Additional problem**: CSD transformation (best way to remove volume conduction) requires 16-32 channels minimum — impossible with 4-channel Muse.

#### DEAP Valence Accuracy Reality Check (FAA-Based)

| Method | Valence Accuracy | Notes |
|--------|-----------------|-------|
| SVM + standard features (incl. FAA) | **55-57%** | Cross-subject, binary |
| Best traditional ML | **63-67%** | Cross-subject, binary |
| Random Forest | ~67% | Cross-subject |
| DEAP deep learning (published) | 85-98% | WITHIN-subject only — not generalizable |
| Cross-subject realistic | **57-72%** | What you get in real deployment |

Chance level for binary classification = 50%. The best cross-subject FAA-based valence classifiers achieve only 13-17 points above chance.

**Why arousal is always ~10 points better than valence**: Arousal has large, bilateral, consistent neural correlates (alpha ERD, beta increase). Valence requires detecting small hemispheric asymmetry differences — an order of magnitude harder.

#### Alpha Subband: 8-13 Hz vs High Alpha (11-13 Hz)

Standard: 8-13 Hz (use this for FAA — confirmed by Cannard 2021, Zhang 2023 finding IAF adds no benefit).

High alpha (11-13 Hz) shows greater specificity for emotional states than low alpha (8-10 Hz) per 2025 Scientific Reports paper. Could be worth extracting separately as an additional feature, but adds complexity.

#### FAA Caveats Summary

1. **AF7/AF8 ≠ F3/F4**: Most FAA research uses dlPFC sites (F3/F4). Muse uses prefrontal/orbital sites (AF7/AF8). Different anatomy, non-directly-comparable to canonical FAA literature.

2. **Multiverse analysis eLife 2021**: 270 analytical pipelines across 5 datasets → only 4.8% significant results (chance level). 8/13 significant results were in the *wrong direction*. FAA-depression relationship is "negligible."

3. **Muse-specific validation study** (220 adults): No association between well-being and FAA at AF7/AF8.

4. **TP9/TP10 more reliable than AF7/AF8** across sessions: forehead dry electrode contact degrades more across days.

5. **Approach motivation ≠ positive valence**: Anger also produces left FAA.

**Conclusion**: FAA is still the best available frontal valence signal, but treat as directional indicator only. The system's 50% FAA + 50% alpha/beta blending is the right approach. Do not weight FAA more than 50% without personalized calibration.

#### Alternative Valence Features Beyond FAA

**Frontal Midline Theta (FMT)** — more robust than FAA for real-time systems:
- Source: Anterior cingulate cortex (ACC) + medial prefrontal cortex (mPFC)
- Pleasant stimuli → higher theta in left hemisphere
- FP1/FP2 theta asymmetry discriminates anger vs. fear
- Less sensitive to reference electrode choice than FAA
- **Not yet implemented** in `eeg_processor.py`

**Prefrontal theta asymmetry**: Significant theta asymmetry in prefrontal/frontal/temporal regions correlates with emotion regulation difficulties (2024 study). Complements FAA.

### 4-Channel Specific Limitations

- **ICA doesn't work with 4 channels** — mathematically requires n_channels > n_artifacts. The artifact removal code that calls ICA will degrade or fail silently on Muse 2.
- **Feature selection mandatory** — with 4 channels and typical session sizes (<500 samples), extracting 17+ features causes overfitting. Target 10-20 features max.
- **30-second epochs outperform shorter windows** for Muse — the 2024 paper found 30-second segments most effective. Current 1-second live windows are far too short.
- **No public Muse-specific emotion dataset exists** — all public datasets use either research-grade (32+ ch) or 14-ch Emotiv (DREAMER, AMIGOS). Muse has fundamentally different electrode placement.

### Foundation Model Option (NeurIPS 2024)

**EEGPT** — 10M parameter transformer pre-trained on large mixed EEG corpus.
- Fine-tuned with linear probing on downstream tasks
- State-of-the-art on many benchmarks
- Code: https://github.com/BINE022/EEGPT
- This is the "GPT moment" for EEG — expected to become dominant paradigm

---

## Real-World Accuracy & Failure Modes (Research Agent Survey, Feb 2026)

From 25 peer-reviewed papers on real-world consumer EEG deployment. Critical for understanding why live emotion readings may appear wrong.

### Honest Accuracy Expectations for Muse 2

| Condition | Binary valence/arousal |
|-----------|----------------------|
| No calibration, naive deployment | **~50% (chance level)** |
| With baseline correction + artifact rejection | **65-75%** |
| + personalization (5 min labeled data) | **75-85%** |
| Lab-controlled, within-subject (NOT real-world) | 90-93% |

**Published Muse-specific numbers**: Frontal alpha asymmetry from 2 frontal channels → **75-76% valence accuracy** (IEEE, controlled lab). Multi-electrode combination (AF8+TP9+TP10) → **88.05% average** but within-subject only. Cross-day repeatability: **low reliability** (Muse validation study, ResearchGate).

**DEAP→SEED cross-dataset transfer**: 65.84% (trained on DEAP, tested on SEED). Reverse: 58.99%. Real-world Muse gap is considerably worse than this.

### Accuracy Impact Per Problem (Fixable vs. Hardware-Limited)

| Problem | Accuracy Impact | Fixable? |
|---------|----------------|---------|
| No baseline normalization | **-15 to -29 pts** | Yes — record 2-3 min resting state |
| Cross-subject model on new person | **-26 pts** | Partially — 5 min fine-tuning |
| Wrong epoch length (1 sec) | **-10 to -20 pts** | Yes — use 4-8 sec windows |
| Jaw clench / blink artifacts | Corrupts affected epochs entirely | Yes — artifact rejection |
| Missing notch filter | Degrades beta features | Yes — add to preprocessing |
| Wrong reference electrode | Distorts spatial patterns | Partially |
| Non-stationarity / session drift | Degrades over time | Partially — online normalization |
| Domain gap (DEAP/SEED → real world) | **-25 to -35 pts** | Partially — personalization |
| 4 electrodes vs 32 | **-5 to -15 pts** | No (hardware limit) |
| Self-report label noise in training | Sets ceiling at ~75-85% | No |

### Epoch Length — Critical Numbers (2024 Paper)

Valence recognition accuracy by epoch length:
- **1 second**: Very noisy (too few cycles of theta/alpha for stable Welch PSD)
- **5 seconds**: Best for within-subject analysis
- **8 seconds**: Best for valence (73.34% accuracy peak)
- **10 seconds**: Best for between-subject analysis

**Physics reason**: Theta (4-8 Hz) needs 2-3 cycles minimum → minimum 375-750ms just for frequency resolution. At 1 second, variance is extremely high. **Recommendation**: 4-second sliding window with 50% overlap (2-second hop). Apply exponential moving average (EMA) with 3-5 second decay constant to the output emotion labels to reduce noise.

### Artifact Rejection Thresholds (Reject the Entire Epoch)

```python
# Flag and reject any epoch where:
amplitude_exceeded = np.any(np.abs(epoch) > 100)           # µV — blink/movement
muscle_artifact = (high_freq_power > rolling_mean + 2*std)  # 25-40 Hz z-score
impulsive_artifact = (kurtosis(epoch) > 10)                 # impulsive spikes

if amplitude_exceeded or muscle_artifact or impulsive_artifact:
    skip_epoch()   # do NOT classify, update rolling stats only
```

**Jaw clenching EMG extends into alpha/beta bands**: Not just gamma. Jaw artifact has significant power down to 8 Hz (alpha band), directly corrupting your primary emotion signal. Even subtle facial expressions activate frontalis under AF7/AF8.

### Reference Electrode Caveat

**Average reference is INVALID for Muse 2's 4 channels.** Average reference requires 64-128 evenly distributed electrodes to be mathematically valid. With 4 frontal/temporal electrodes, it distorts all spatial information.

**Better option for Muse**: Treat TP9/TP10 as linked mastoid reference. Note: TP9/TP10 are themselves near the temporalis muscle and may carry emotion-related signals, which contaminates the reference. There is no perfect reference with 4-channel Muse.

### Baseline Normalization Formula (Must Implement)

```python
# Record 2 min eyes-closed + 1 min eyes-open resting state.
# Compute baseline_mean and baseline_std per feature.
# During live classification:
corrected_feature = (task_power - baseline_mean) / baseline_std

# Why: skull thickness, hair, headset fit cause 30-50% amplitude variation
# across individuals. Without this, cross-person classification is chance level.
```

**Re-record baseline at every session** — it drifts significantly across days.

### Non-Stationarity: Why Output Drifts Over Time

Even without emotional state changes, EEG features drift because:
- **Alpha peak frequency** drops continuously with time-on-task (NeuroImage, 2019)
- **Cortical adaptation**: EEG response to repeated stimuli weakens over time (habituation)
- **Cross-day variability**: Cross-day accuracy degrades significantly for the same person

**Fix**: Online running normalization — update feature mean/std continuously using last 5-10 minutes of clean data. Partially corrects within-session drift without full model retrain.

### Self-Report Label Noise — Accuracy Ceiling

DEAP/SEED use post-hoc self-assessment (SAM scale). Sources of label error:
1. Temporal mismatch: rated after clip ends, not during
2. Alexithymia: ~10% of people cannot reliably report their emotions
3. Normative vs individual labels: a horror clip labeled "fearful" may feel "exciting" to some

**Result**: Training labels have ~15-25% base error rate → effective accuracy ceiling of ~75-85% even with perfect signal quality. This is irreducible without real-time physiological ground truth.

### Minimum Viable Session Protocol for Meaningful Output

```
1. [2 min]  Electrode settling — do not record. Wait for impedance to stabilize.
2. [3 min]  Eyes-closed resting baseline recording.
             Instruct: breathe normally, minimize jaw tension.
3. [2 min]  Eyes-open resting baseline (fixation cross).
4. [5 min]  (Recommended) Personalization calibration:
             Show 3-4 clearly valenced clips, record labeled EEG.
             Fine-tune normalization parameters for this user.
Total: ~12 minutes before usable emotion data
```

**Without this protocol**: output is noise with confidence intervals spanning the entire label space.

### Validated Preprocessing Pipeline (Minimum Viable)

```
1. High-pass filter: 1 Hz (zero-phase / filtfilt — remove DC drift)
2. Notch filter: 50 or 60 Hz (DFT spectrum interpolation, not IIR notch)
3. Low-pass filter: 45 Hz
4. Artifact detection: reject epochs > ±100 µV or kurtosis > 10
5. Band-power extraction: theta, alpha, beta, FAA (4-8 sec epochs)
6. Baseline normalization: (feature - baseline_mean) / baseline_std
7. Epoch length: 4-sec sliding window, 50% overlap (2-sec hop)
8. Output smoothing: EMA with 3-5 sec decay constant on emotion labels
```

Current system gaps vs this pipeline: epochs are 1 sec (too short), no baseline normalization, IIR notch filter (causes phase distortion), no epoch rejection, no output EMA smoothing.

---

## Session History & Major Changes

| Date | Commit | Change |
|------|--------|--------|
| Feb 2026 | `270d27e` | Fix EEG emotion pipeline: FAA integration, remove EMG gamma noise, fix sad threshold, fix duplicate arousal, fix single-term valence |
| Earlier | `709ccac` | Add Tesla Sales Dashboard |
| Earlier | various | AI Companion fixes, session data consolidation, light mode tooltips, happiness bug |

For the full scientific background, read `docs/COMPLETE_SCIENTIFIC_GUIDE.md` (40KB).
For the interactive EEG science explainer with diagrams, open `docs/eeg-science-explainer.html` in a browser.
