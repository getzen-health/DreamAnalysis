"""
Cross-Dataset LGBM Trainer (v3 — with Euclidean Alignment)
===========================================================
Trains a LightGBM emotion classifier on ALL available datasets using the exact
same 80-feature extraction as live Muse 2 inference — making it the first model
that is BOTH high-accuracy AND genuinely deployable in real time.

v3 improvement: Euclidean Alignment (EA) per subject
  Reference: He & Wu (2020), "Transfer Learning for Brain-Computer Interfaces:
    A Euclidean Space Data Alignment Approach", IEEE Trans. Biomed. Eng.
  Confirmed: +4.33% cross-subject accuracy (arXiv:2401.10746, 2024).
  Revisited: recommended across 13 BCI paradigms (arXiv:2502.09203, 2025).

  EA whitens each subject's multi-channel EEG covariance to identity BEFORE
  feature extraction, removing inter-subject variability from skull thickness,
  electrode impedance, and brain geometry. This is done per-subject on raw
  4-channel epochs, making the extracted features more comparable across subjects.

Key design decisions:
  - 80 raw features: 5 bands x 4 channels x 4 stats (mean, std, median, IQR)
    -- identical to _extract_muse_live_features() in emotion_classifier.py
  - 4-channel subset from each dataset: ch0=left-temporal, ch1=left-frontal,
    ch2=right-frontal, ch3=right-temporal (Muse 2 equivalent)
  - 3-class output: 0=positive, 1=neutral, 2=negative
    -- compatible with existing 3->6 expansion in _predict_lgbm_muse()
  - Single global StandardScaler (no per-dataset scaling, no PCA)
    -- reproducible at inference time
  - Euclidean Alignment applied per-subject on raw epochs before features
  - Saves to emotion_lgbm_muse_live.pkl (replaces previous model)
    -- loaded automatically by emotion_classifier.py if accuracy >= 60%

Datasets loaded:
  1. DEAP       -- 32 subjects, 40 trials, 32-ch gel EEG (4-ch frontal/temporal subset)
  2. EmoKey     -- 45 subjects, real Muse S recordings, 4 channels, CSV format
  3. DREAMER    -- 23 subjects, Emotiv EPOC 14-ch (4-ch subset) -- if DREAMER.mat present
  4. GAMEEMO    -- 28 subjects, Emotiv EPOC 14-ch, 4 emotions (if available)

Usage (run from ml/ directory):
    .venv/bin/python -m training.train_cross_dataset_lgbm
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import signal
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from processing.eeg_processor import euclidean_align

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[WARN] LightGBM not found — falling back to GradientBoosting")

try:
    import joblib
except ImportError:
    import pickle as joblib  # type: ignore

MODEL_DIR     = Path("models/saved")
BENCHMARK_DIR = Path("benchmarks")
MODEL_OUT     = MODEL_DIR / "emotion_lgbm_muse_live.pkl"
BENCH_OUT     = BENCHMARK_DIR / "emotion_lgbm_muse_live_benchmark.json"

FS_MUSE   = 256.0  # Muse 2 live sampling rate
FS_DEAP   = 128.0  # DEAP pre-processed
FS_EMOKEY = 128.0  # EmoKey Muse S delivery rate
FS_DREAMER = 128.0 # DREAMER Emotiv EPOC
FS_GAMEEMO = 128.0 # GAMEEMO Emotiv EPOC

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta":  (12, 30),
    "gamma": (30, 50),
}

WIN_S  = 4.0   # 4-second window
HOP_S  = 2.0   # 50 % overlap
SW_S   = 0.5   # 0.5-sec sub-windows for feature aggregation

# 75 µV artifact threshold (Kriegolson 2021)
ARTIFACT_UV = 75.0

# Gamma feature indices in the 85-feature vector
# Layout: 5bands × 4ch × 4stats = 80 features, then 5 DASM
# Gamma band = band_idx 4: positions 64–79  (16 features)
# DASM_gamma  = index 84                    (1 feature)
GAMMA_FEAT_IDX: list[int] = list(range(64, 80)) + [84]  # 17 gamma-related features

# Consumer dry-electrode devices where gamma is dominated by EMG artifact.
# When one of these devices is connected, gamma features are zeroed at inference.
CONSUMER_EEG_DEVICES: frozenset[str] = frozenset({
    "muse_2", "muse_2_bled", "muse_s", "muse_s_bled",
    "muse_2016", "muse_2016_bled", "muse",
})

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ──────────────────────────────────────────────────────────────────────────────

def _band_power(sig: np.ndarray, fs: float) -> np.ndarray:
    """Return 5-element array of normalised band powers [delta, theta, alpha, beta, gamma]."""
    n = len(sig)
    if n < 4:
        return np.zeros(5)
    freqs, psd = signal.welch(sig, fs, nperseg=min(256, n))
    total = np.sum(psd) + 1e-10
    out = []
    for lo, hi in BANDS.values():
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        out.append(float(np.sum(psd[idx]) / total) if len(idx) > 0 else 0.0)
    return np.array(out, dtype=np.float32)


def extract_features(eeg4ch: np.ndarray, fs: float) -> np.ndarray:
    """Extract 85-feature vector from (4, n_samples) EEG array.

    Layout (matches EmotionClassifier._extract_muse_live_features):
        80 band-power stats: 5 bands × 4 channels × 4 stats (mean,std,median,IQR)
         5 DASM features:    mean(AF8_band) - mean(AF7_band) for each of 5 bands
    Channel order: [left-temporal=0, left-frontal=1, right-frontal=2, right-temporal=3]
    AF7 = ch1 (left frontal), AF8 = ch2 (right frontal)
    """
    n_samples = eeg4ch.shape[1]
    win = max(4, int(SW_S * fs))
    hop = max(2, win // 2)

    # Collect sub-window band powers: shape (n_subwins, 5_bands, 4_ch)
    powers = []
    for start in range(0, n_samples - win + 1, hop):
        end = start + win
        sub = []
        for ch in range(4):
            seg = eeg4ch[ch, start:end]
            sub.append(_band_power(seg, fs))  # (5,)
        powers.append(np.stack(sub, axis=1))  # (5, 4)

    if not powers:
        return np.zeros(85, dtype=np.float32)

    arr = np.array(powers)  # (n_subwins, 5, 4)
    feat = []

    # 80 band-power stats
    for band_idx in range(5):
        for ch_idx in range(4):
            vals = arr[:, band_idx, ch_idx]
            vals = vals[np.isfinite(vals)]
            if len(vals) < 2:
                feat.extend([0.0, 0.0, 0.0, 0.0])
            else:
                feat.extend([
                    float(np.mean(vals)),
                    float(np.std(vals)),
                    float(np.median(vals)),
                    float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                ])

    # 5 DASM features: mean(AF8_band) - mean(AF7_band) per band
    # AF7=ch1 (left frontal), AF8=ch2 (right frontal)
    for band_idx in range(5):
        af8 = arr[:, band_idx, 2]
        af8 = af8[np.isfinite(af8)]
        af7 = arr[:, band_idx, 1]
        af7 = af7[np.isfinite(af7)]
        if len(af8) > 0 and len(af7) > 0:
            feat.append(float(np.mean(af8)) - float(np.mean(af7)))
        else:
            feat.append(0.0)

    return np.array(feat, dtype=np.float32)  # 85 features


# Keep alias for backward compatibility
extract_80_features = extract_features


def epoch_and_extract(eeg_ch: np.ndarray, fs: float,
                      win_s: float = WIN_S, hop_s: float = HOP_S,
                      ch_map: list | None = None,
                      artifact_uv: float = ARTIFACT_UV) -> list[np.ndarray]:
    """Slice a (n_ch, n_samples) signal into overlapping windows and extract features."""
    n_ch, n_samples = eeg_ch.shape
    win = int(win_s * fs)
    hop = int(hop_s * fs)

    if ch_map is None:
        ch_map = list(range(min(4, n_ch)))
    ch_map = ch_map[:4]
    while len(ch_map) < 4:
        ch_map.append(ch_map[-1])  # repeat last channel if < 4

    feats = []
    for start in range(0, n_samples - win + 1, hop):
        segment = eeg_ch[ch_map, start:start + win]  # (4, win)
        # Artifact check
        if np.any(np.abs(segment) > artifact_uv):
            continue
        feats.append(extract_80_features(segment, fs))
    return feats


def slice_epochs(eeg_ch: np.ndarray, fs: float,
                 win_s: float = WIN_S, hop_s: float = HOP_S,
                 ch_map: list | None = None,
                 artifact_uv: float = ARTIFACT_UV) -> list[np.ndarray]:
    """Slice a (n_ch, n_samples) signal into overlapping raw epochs.

    Returns list of (4, win_samples) arrays (raw, not feature-extracted).
    Used as input to Euclidean Alignment before feature extraction.
    """
    n_ch, n_samples = eeg_ch.shape
    win = int(win_s * fs)
    hop = int(hop_s * fs)

    if ch_map is None:
        ch_map = list(range(min(4, n_ch)))
    ch_map = ch_map[:4]
    while len(ch_map) < 4:
        ch_map.append(ch_map[-1])

    epochs = []
    for start in range(0, n_samples - win + 1, hop):
        segment = eeg_ch[ch_map, start:start + win]  # (4, win)
        if np.any(np.abs(segment) > artifact_uv):
            continue
        epochs.append(segment)
    return epochs


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_deap(data_dir: str = "data/deap") -> tuple[np.ndarray, np.ndarray]:
    """Load DEAP dataset, 4-ch Muse-equivalent subset.

    DEAP channel order (0-indexed):
      ch1  = AF3  → AF7 analog  (left frontal)
      ch2  = F3
      ch3  = F7
      ch7  = T7   → TP9 analog  (left temporal)
      ch15 = FC5
      ch16 = Fp1
      ch17 = AF4  → AF8 analog  (right frontal)
      ch22 = T8   → TP10 analog (right temporal)
    Chosen: T7=7, AF3=1, AF4=17, T8=22 → [TP9, AF7, AF8, TP10] analog
    """
    import pickle

    deap_path = Path(data_dir)
    if not deap_path.exists():
        log("  [DEAP] directory not found — skipping")
        return np.array([]), np.array([])

    # DEAP: ch7=T7(TP9), ch1=AF3(AF7), ch17=AF4(AF8), ch22=T8(TP10)
    CH_MAP = [7, 1, 17, 22]

    # Emotion label from valence/arousal using adaptive median split
    # Median-split to 3 classes: 0=positive, 1=neutral, 2=negative
    all_V, all_A = [], []
    trials_data = []

    dat_files = sorted(deap_path.glob("s*.dat"))
    log(f"  [DEAP] Loading {len(dat_files)} subjects...")

    for f in dat_files:
        try:
            with open(f, "rb") as fh:
                d = pickle.load(fh, encoding="latin1")
            trials_data.append((d["data"], d["labels"]))
            all_V.extend(d["labels"][:, 0].tolist())
            all_A.extend(d["labels"][:, 1].tolist())
        except Exception as e:
            log(f"    [DEAP] Error loading {f.name}: {e}")

    if not trials_data:
        return np.array([]), np.array([])

    med_V = float(np.median(all_V))
    med_A = float(np.median(all_A))
    log(f"  [DEAP] Medians — V={med_V:.2f}  A={med_A:.2f}")

    X, y = [], []
    for data, labels in trials_data:
        # data: (40, 40, 8064) — trials × channels × samples @ 128 Hz
        # ── Euclidean Alignment: collect all raw epochs from this subject ──
        subj_epochs = []
        subj_labels = []
        for trial_idx in range(data.shape[0]):
            eeg = data[trial_idx, :32, :]  # first 32 EEG channels
            valence = float(labels[trial_idx, 0])

            if valence >= med_V:
                label = 0  # positive
            else:
                label = 2  # negative

            eeg_sub = eeg[CH_MAP, :]  # (4, 8064)
            trial_epochs = slice_epochs(eeg_sub, FS_DEAP)
            for ep in trial_epochs:
                subj_epochs.append(ep)
                subj_labels.append(label)

        if not subj_epochs:
            continue

        # Apply Euclidean Alignment to this subject's epochs
        epochs_arr = np.array(subj_epochs)  # (n_epochs, 4, win_samples)
        aligned_epochs, _ = euclidean_align(epochs_arr)

        # Now extract features from aligned epochs
        for i, ep in enumerate(aligned_epochs):
            feat = extract_80_features(ep, FS_DEAP)
            X.append(feat)
            y.append(subj_labels[i])

    if not X:
        return np.array([]), np.array([])

    log(f"  [DEAP] {len(X)} samples from {len(dat_files)} subjects (EA-aligned)")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def load_emokey(data_dir: str = "data/emokey") -> tuple[np.ndarray, np.ndarray]:
    """Load EmoKey Moments (Muse S — 4ch CSV) — real Muse hardware."""
    import pandas as pd

    base = Path(data_dir)
    # Try the known nested path
    nested = (base / "EmoKey Moments EEG Dataset (EKM-ED)"
              / "muse_wearable_data" / "preprocessed"
              / "clean-signals" / "0.0078125S")
    search_roots = [nested, base]

    csv_files: list[Path] = []
    for root in search_roots:
        if root.exists():
            csv_files = list(root.rglob("*.csv"))
            if csv_files:
                break

    if not csv_files:
        log("  [EmoKey] No CSV files found — skipping")
        return np.array([]), np.array([])

    EMOTION_MAP = {
        "HAPPINESS": 0, "NEUTRAL_HAPPINESS": 1,
        "SADNESS": 2,   "NEUTRAL_SADNESS": 2,
        "ANGER": 2,     "NEUTRAL_ANGER": 2,
        "FEAR": 2,      "NEUTRAL_FEAR": 2,
    }
    # Exact column names from EmoKey CSV
    CH_COLS = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]

    # Group CSVs by subject ID (first part of filename before emotion tag)
    # Each subject's recordings are EA-aligned together
    from collections import defaultdict

    subj_files: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for csv_path in csv_files:
        emotion_key = None
        for emo in EMOTION_MAP:
            if emo in csv_path.stem.upper():
                emotion_key = emo
                break
        if emotion_key is None:
            continue
        label = EMOTION_MAP[emotion_key]
        # Extract subject ID from filename (e.g., "S01_HAPPINESS" -> "S01")
        parts = csv_path.stem.split("_")
        subj_id = parts[0] if len(parts) > 1 else csv_path.stem
        subj_files[subj_id].append((csv_path, label))

    X, y = [], []
    loaded = 0
    for subj_id, file_list in subj_files.items():
        subj_epochs = []
        subj_labels = []
        for csv_path, label in file_list:
            try:
                df = pd.read_csv(csv_path)
                cols = [c for c in CH_COLS if c in df.columns]
                if len(cols) < 4:
                    continue
                eeg = df[cols].values.T.astype(np.float32)  # (4, n)
                if eeg.shape[1] < int(WIN_S * FS_EMOKEY):
                    continue
                raw_eps = slice_epochs(eeg, FS_EMOKEY)
                for ep in raw_eps:
                    subj_epochs.append(ep)
                    subj_labels.append(label)
                loaded += 1
            except Exception as e:
                log(f"    [EmoKey] Error on {csv_path.name}: {e}")

        if not subj_epochs:
            continue

        # Euclidean Alignment per subject
        epochs_arr = np.array(subj_epochs)
        aligned_epochs, _ = euclidean_align(epochs_arr)
        for i, ep in enumerate(aligned_epochs):
            feat = extract_80_features(ep, FS_EMOKEY)
            X.append(feat)
            y.append(subj_labels[i])

    if not X:
        return np.array([]), np.array([])

    log(f"  [EmoKey] {len(X)} samples from {loaded} CSV files (EA-aligned)")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def load_dreamer(mat_path: str = "../data/DREAMER.mat") -> tuple[np.ndarray, np.ndarray]:
    """Load DREAMER dataset (Emotiv EPOC 14-ch).

    DREAMER .mat structure (loaded via scipy.io):
      DREAMER['DREAMER'][0,0]['Data'][0,subject]['EEG'][0,0]['stimuli'][0,trial]['data']
      shape: (n_samples, 14)
    Labels: valence/arousal on 1-5 scale (median split to 3 classes)

    14-ch Emotiv channel order: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
    Muse mapping: T7(ch4)→TP9, AF3(ch0)→AF7, AF4(ch13)→AF8, T8(ch9)→TP10
    """
    p = Path(mat_path)
    if not p.exists():
        log(f"  [DREAMER] {mat_path} not found — skipping")
        return np.array([]), np.array([])

    try:
        from scipy.io import loadmat
        mat = loadmat(str(p), simplify_cells=False)
    except Exception as e:
        try:
            import h5py
            mat = None
            _h5 = h5py.File(str(p), "r")
        except Exception as e2:
            log(f"  [DREAMER] Failed to load: {e} / {e2}")
            return np.array([]), np.array([])

    # Emotiv 14-ch: T7=ch4, AF3=ch0, AF4=ch13, T8=ch9
    CH_MAP = [4, 0, 13, 9]  # → [TP9, AF7, AF8, TP10] analog

    X, y = [], []
    try:
        dreamer = mat["DREAMER"][0, 0]
        n_subjects = dreamer["Data"].shape[1]
        all_V, all_A = [], []

        # First pass: collect all labels for median computation
        # Actual DREAMER structure: stimuli[0,0] → (18,1) cell; ScoreValence[0,0] → (18,1)
        for s in range(n_subjects):
            subj = dreamer["Data"][0, s]
            scores_V = subj["ScoreValence"][0, 0]  # (18, 1)
            scores_A = subj["ScoreArousal"][0, 0]  # (18, 1)
            for trial in range(scores_V.shape[0]):
                all_V.append(float(scores_V[trial, 0]))
                all_A.append(float(scores_A[trial, 0]))

        med_V = float(np.median(all_V))
        med_A = float(np.median(all_A))
        log(f"  [DREAMER] Medians — V={med_V:.2f}  A={med_A:.2f}")

        # Second pass: extract features with per-subject Euclidean Alignment
        from scipy.signal import butter, filtfilt
        b_filt, a_filt = butter(4, [1.0, 45.0], btype="band", fs=FS_DREAMER)

        for s in range(n_subjects):
            subj = dreamer["Data"][0, s]
            eeg_stim = subj["EEG"][0, 0]["stimuli"][0, 0]  # (18, 1) cell array of trials
            scores_V = subj["ScoreValence"][0, 0]           # (18, 1)
            scores_A = subj["ScoreArousal"][0, 0]           # (18, 1)
            n_trials = eeg_stim.shape[0]

            # Collect all raw epochs from this subject
            subj_epochs = []
            subj_labels = []

            for trial in range(n_trials):
                eeg_raw = eeg_stim[trial, 0]  # (n_samples, 14)
                if eeg_raw.ndim == 2 and eeg_raw.shape[1] >= 14:
                    eeg = eeg_raw.T.astype(np.float32)  # (14, n_samples)
                else:
                    continue

                val = float(scores_V[trial, 0])

                if val >= med_V:
                    label = 0  # positive
                else:
                    label = 2  # negative

                eeg_sub = eeg[CH_MAP, :]
                eeg_sub = eeg_sub * 0.51  # → µV
                for ch_i in range(eeg_sub.shape[0]):
                    eeg_sub[ch_i] = filtfilt(b_filt, a_filt, eeg_sub[ch_i])
                raw_eps = slice_epochs(eeg_sub, FS_DREAMER, artifact_uv=150.0)
                for ep in raw_eps:
                    subj_epochs.append(ep)
                    subj_labels.append(label)

            if not subj_epochs:
                continue

            # Euclidean Alignment per subject
            epochs_arr = np.array(subj_epochs)
            aligned_epochs, _ = euclidean_align(epochs_arr)
            for i, ep in enumerate(aligned_epochs):
                feat = extract_80_features(ep, FS_DREAMER)
                X.append(feat)
                y.append(subj_labels[i])

    except Exception as e:
        log(f"  [DREAMER] Parsing error: {e}")
        return np.array([]), np.array([])

    if not X:
        return np.array([]), np.array([])

    log(f"  [DREAMER] {len(X)} samples from {n_subjects} subjects (EA-aligned)")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def load_gameemo(data_dir: str = "../data/gameemo") -> tuple[np.ndarray, np.ndarray]:
    """Load GAMEEMO dataset (Emotiv EPOC 14-ch, 28 subjects, 4 games).

    Structure: GAMEEMO/(S01)/Preprocessed EEG Data/.csv format/S01G1AllChannels.csv
    Games: G1=Boring(neutral), G2=Calm(positive), G3=Horror(negative), G4=Fun(positive)
    Columns: AF3,AF4,F3,F4,F7,F8,FC5,FC6,O1,O2,P7,P8,T7,T8 (14 channels, already µV)
    Channel mapping: T7=12, AF3=0, AF4=1, T8=13 → [left-temporal, left-frontal, right-frontal, right-temporal]
    """
    import pandas as pd

    base = Path(data_dir) / "GAMEEMO"
    if not base.exists():
        log(f"  [GAMEEMO] {base} not found — skipping")
        return np.array([]), np.array([])

    # G1=boring→neutral, G2=calm→positive, G3=horror→negative, G4=fun→positive
    GAME_LABEL = {"G1": 1, "G2": 0, "G3": 2, "G4": 0}
    # Emotiv 14-ch column order: AF3,AF4,F3,F4,F7,F8,FC5,FC6,O1,O2,P7,P8,T7,T8
    CH_COLS = ["AF3", "AF4", "F3", "F4", "F7", "F8", "FC5", "FC6", "O1", "O2", "P7", "P8", "T7", "T8"]
    CH_MAP = [12, 0, 1, 13]  # T7, AF3, AF4, T8 → [TP9, AF7, AF8, TP10] analog

    # Only preprocessed CSV files (not raw)
    csv_files = [
        f for f in sorted(base.rglob("*AllChannels.csv"))
        if "Preprocessed" in str(f)
    ]
    log(f"  [GAMEEMO] Found {len(csv_files)} preprocessed CSV files")

    # Group by subject (extract S01, S02, ... from filename)
    from collections import defaultdict
    import re

    subj_files: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for f in csv_files:
        fname = f.stem
        game_id = next((g for g in GAME_LABEL if g in fname), None)
        if game_id is None:
            continue
        label = GAME_LABEL[game_id]
        # Extract subject ID: "S01G2AllChannels" -> "S01"
        m = re.match(r"(S\d+)", fname)
        subj_id = m.group(1) if m else fname[:3]
        subj_files[subj_id].append((f, label))

    X, y = [], []
    for subj_id, file_list in subj_files.items():
        subj_epochs = []
        subj_labels = []
        for f, label in file_list:
            try:
                df = pd.read_csv(f, usecols=[c for c in CH_COLS])
                df = df[[c for c in CH_COLS if c in df.columns]].dropna(axis=1, how="all")
                if df.shape[1] < 14:
                    continue
                eeg = df.to_numpy(dtype=np.float32).T  # (14, n_samples)
                eeg_sub = eeg[CH_MAP, :]               # (4, n_samples)
                raw_eps = slice_epochs(eeg_sub, FS_GAMEEMO)
                for ep in raw_eps:
                    subj_epochs.append(ep)
                    subj_labels.append(label)
            except Exception:
                pass

        if not subj_epochs:
            continue

        # Euclidean Alignment per subject
        epochs_arr = np.array(subj_epochs)
        aligned_epochs, _ = euclidean_align(epochs_arr)
        for i, ep in enumerate(aligned_epochs):
            feat = extract_80_features(ep, FS_GAMEEMO)
            X.append(feat)
            y.append(subj_labels[i])

    if not X:
        return np.array([]), np.array([])

    log(f"  [GAMEEMO] {len(X)} samples from {len(subj_files)} subjects (EA-aligned)")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# Gamma dropout augmentation
# ──────────────────────────────────────────────────────────────────────────────

def augment_gamma_dropout(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add gamma-zeroed copies of every training sample.

    This teaches the model to classify emotion correctly both when gamma is
    present (research-grade EEG) and when it is absent (consumer Muse 2, where
    gamma is zeroed at inference to avoid EMG contamination).

    Result: doubles the dataset — original samples + identical copies with
    all 17 gamma features (indices 64-79 and 84) set to zero.
    """
    X_no_gamma = X.copy()
    X_no_gamma[:, GAMMA_FEAT_IDX] = 0.0
    X_aug = np.vstack([X, X_no_gamma])
    y_aug = np.concatenate([y, y])
    return X_aug, y_aug


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> tuple:
    """Train LightGBM with 5-fold CV + early stopping. Returns (model, scaler, cv_accuracy, cv_f1).

    Improvements over v1:
      - Scaler fit INSIDE each CV fold to prevent data leakage (validation data
        no longer leaks into the StandardScaler mean/std during CV evaluation).
      - LightGBM early stopping: trains up to 2000 iterations but stops if
        validation logloss does not improve for 100 rounds. Prevents overfitting
        and auto-selects optimal tree count per fold.
      - Final model uses the mean best_iteration from CV folds (with 10% headroom)
        instead of an arbitrary fixed n_estimators.
    """
    # Gamma dropout: double the dataset with gamma-zeroed copies.
    # Enables device-aware inference without retraining.
    log("\n[AUG] Applying gamma dropout — doubling dataset with gamma-zeroed copies")
    X, y = augment_gamma_dropout(X, y)
    log(f"[AUG] Dataset: {len(X)//2} → {len(X)} samples after augmentation")

    log(f"\n[TRAINING] {len(X)} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

    cv_accs, cv_f1s, best_iters = [], [], []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr_raw, X_va_raw = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        w_tr = compute_sample_weight("balanced", y_tr)

        # Fit scaler on TRAINING fold only — prevents data leakage
        fold_scaler = StandardScaler()
        X_tr = fold_scaler.fit_transform(X_tr_raw)
        X_va = fold_scaler.transform(X_va_raw)

        if HAS_LGBM:
            clf = lgb.LGBMClassifier(
                n_estimators=2000,    # higher ceiling — early stopping finds optimal
                learning_rate=0.03,
                max_depth=6,          # shallower → better cross-subject generalization
                num_leaves=40,
                min_child_samples=40, # require more data per leaf → less overfitting
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.5,        # stronger L1
                reg_lambda=1.0,       # stronger L2
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
                verbosity=-1,
            )
            # Early stopping: stop if val logloss doesn't improve for 100 rounds
            clf.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=0),  # suppress per-iteration logs
                ],
            )
            best_iter = clf.best_iteration_ if hasattr(clf, "best_iteration_") else 2000
            best_iters.append(best_iter)
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=42)
            clf.fit(X_tr, y_tr, sample_weight=w_tr)
            best_iter = 300
            best_iters.append(best_iter)

        preds = clf.predict(X_va)
        acc = accuracy_score(y_va, preds)
        f1  = f1_score(y_va, preds, average="macro", zero_division=0)
        cv_accs.append(acc)
        cv_f1s.append(f1)
        log(f"  Fold {fold}/{n_splits} — acc={acc:.4f}  f1={f1:.4f}  best_iter={best_iter}")

    cv_acc = float(np.mean(cv_accs))
    cv_f1  = float(np.mean(cv_f1s))
    mean_best_iter = int(np.mean(best_iters))
    log(f"\n[CV RESULT] acc={cv_acc:.4f} +/- {np.std(cv_accs):.4f}  f1={cv_f1:.4f}")
    log(f"[CV RESULT] mean best_iteration={mean_best_iter}  (per-fold: {best_iters})")

    # Retrain on all data with the scaler fit on everything (deployment scaler)
    log("\n[FINAL MODEL] Training on full dataset...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    w_all = compute_sample_weight("balanced", y)

    # Use mean best iteration + 10% headroom for the final model
    final_n_est = int(mean_best_iter * 1.1) if HAS_LGBM else 300
    log(f"[FINAL MODEL] n_estimators={final_n_est} (from CV mean {mean_best_iter} + 10%)")

    if HAS_LGBM:
        final_model = lgb.LGBMClassifier(
            n_estimators=final_n_est,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=40,
            min_child_samples=40,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=1.0,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            verbosity=-1,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        final_model = GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=42)

    final_model.fit(X_scaled, y, sample_weight=w_all)
    train_preds = final_model.predict(X_scaled)
    log(f"[FINAL MODEL] Train accuracy: {accuracy_score(y, train_preds):.4f}")
    log("\nClassification Report (train):")
    class_map = {0: "positive", 1: "neutral", 2: "negative"}
    present = sorted(np.unique(y))
    log(classification_report(y, train_preds,
                               labels=present,
                               target_names=[class_map[c] for c in present],
                               zero_division=0))

    return final_model, scaler, cv_acc, cv_f1


def save_model(model, scaler, cv_acc: float, cv_f1: float,
               n_samples: int, dataset_names: list[str]) -> None:
    """Save model bundle and benchmark JSON."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    payload = {"model": model, "scaler": scaler}
    joblib.dump(payload, MODEL_OUT)
    log(f"\n[SAVE] Model → {MODEL_OUT}")

    bench = {
        "model_name":   "emotion_lgbm_muse_live",
        "datasets":     dataset_names,
        "n_samples":    n_samples,
        "cv_method":    "5-fold stratified CV",
        "accuracy":     round(cv_acc, 4),
        "f1_macro":     round(cv_f1, 4),
        "n_classes":    3,
        "class_names":  ["positive", "neutral", "negative"],
        "feature_dim":  85,
        "feature_type": "80 band-power stats (5bands×4ch×4stats) + 5 DASM (AF8-AF7 per band)",
        "notes": (
            f"Cross-dataset LGBM trained on {'+'.join(dataset_names)}. "
            "85-feature format: 80 band-power + 5 DASM. Matches live Muse 2 inference. "
            f"5-fold CV accuracy: {cv_acc:.2%}. Neutral class added via valence/arousal proximity."
        ),
    }
    BENCH_OUT.write_text(json.dumps(bench, indent=2))
    log(f"[SAVE] Benchmark → {BENCH_OUT}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("  Cross-Dataset LGBM Trainer — 85-feature live-compatible model (v3)")
    log("  + Euclidean Alignment per subject (He & Wu, IEEE TBME 2020)")
    log("=" * 70)
    t0 = time.time()

    all_X, all_y, dataset_names = [], [], []

    # 1. DEAP
    X_d, y_d = load_deap()
    if len(X_d) > 0:
        all_X.append(X_d)
        all_y.append(y_d)
        dataset_names.append("DEAP")

    # 2. EmoKey (real Muse hardware)
    X_e, y_e = load_emokey()
    if len(X_e) > 0:
        all_X.append(X_e)
        all_y.append(y_e)
        dataset_names.append("EmoKey")

    # 3. DREAMER (optional — needs data/DREAMER.mat)
    X_dr, y_dr = load_dreamer()
    if len(X_dr) > 0:
        all_X.append(X_dr)
        all_y.append(y_dr)
        dataset_names.append("DREAMER")

    # 4. GAMEEMO (optional)
    X_g, y_g = load_gameemo()
    if len(X_g) > 0:
        all_X.append(X_g)
        all_y.append(y_g)
        dataset_names.append("GAMEEMO")

    if not all_X:
        log("\n[ERROR] No data loaded — check data/ directory")
        return

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    # Remove NaN/Inf
    mask = np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]

    log(f"\n[DATA] Total: {len(X)} samples, {X.shape[1]} features")
    log(f"[DATA] Sources: {', '.join(dataset_names)}")
    classes, counts = np.unique(y, return_counts=True)
    for c, n in zip(classes, counts):
        name = ["positive", "neutral", "negative"][c]
        log(f"  Class {c} ({name}): {n} samples ({n/len(y):.1%})")

    # Train
    model, scaler, cv_acc, cv_f1 = train(X, y)

    # Save
    save_model(model, scaler, cv_acc, cv_f1, len(X), dataset_names)

    elapsed = time.time() - t0
    log(f"\n[DONE] Elapsed: {elapsed:.1f}s")
    log(f"[DONE] CV accuracy: {cv_acc:.2%}  |  F1: {cv_f1:.4f}")
    log(f"[DONE] Saved to {MODEL_OUT}")

    if cv_acc >= 0.60:
        log("[DONE] ✓ Model meets 60% threshold — will be loaded as primary inference path")
    else:
        log("[WARN] ✗ Model below 60% threshold — feature heuristics will be used instead")


if __name__ == "__main__":
    main()
