"""
Mega LGBM Unified Cross-Dataset Trainer
=========================================
Trains a LightGBM emotion classifier on all available raw-EEG datasets
using a SINGLE global PCA (80 components) + global StandardScaler.
Trains a single deployable model — all transforms
are saved and reproducible at live inference time.

Pipeline:
  raw EEG (4ch, n_samples) → extract_features() → 85-dim
  → GlobalScaler (StandardScaler, fit on all training data)
  → GlobalPCA   (PCA 85→80, fit on all training data)
  → LightGBM    → 3-class (positive=0, neutral=1, negative=2)

Datasets included:
  - DEAP     (32 subjects, 128 Hz, 32-ch gel → 4-ch subset)
  - DREAMER  (23 subjects, Emotiv EPOC 14-ch)
  - GAMEEMO  (28 subjects, Emotiv EPOC 14-ch, 4 emotions)
  - DENS     (27/40 subjects, EGI 128-ch → 4-ch Muse equivalent)
  - FACED    (123 subjects, 32-ch, pre-extracted DE)
  - SEED-IV  (15 subjects × 3 sessions, 62-ch, pre-extracted DE, 4 emotions)
  - EEG-ER   (100 recordings, Emotiv EPOC 14-ch, 128 Hz, 4 emotions)
  - STEW     (45 segments, Emotiv EPOC 14-ch, 128 Hz, cognitive workload 3-class)
  - Muse-Sub (40 recordings, Muse 4-ch band powers, subconscious decisions)

Saves:
  models/saved/emotion_mega_lgbm.pkl          — {model, scaler, pca, metadata}
  benchmarks/emotion_mega_lgbm_benchmark.json — accuracy + CV

Usage (run from ml/ directory):
    python3 training/train_mega_lgbm_unified.py
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).resolve().parent
_ML_ROOT = _HERE.parent

for _p in [str(_ML_ROOT), str(_ML_ROOT / "processing"), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from scipy import signal as _sig
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    _LGB_OK = True
except ImportError:
    _LGB_OK = False
    print("ERROR: lightgbm not installed.", file=sys.stderr)
    sys.exit(1)

try:
    import scipy.io as sio
    _SCI_OK = True
except ImportError:
    _SCI_OK = False

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_OK = True
except ImportError:
    _SMOTE_OK = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR     = _ML_ROOT / "models" / "saved"
BENCHMARK_DIR = _ML_ROOT / "benchmarks"
MODEL_OUT     = MODEL_DIR  / "emotion_mega_lgbm.pkl"
BENCH_OUT     = BENCHMARK_DIR / "emotion_mega_lgbm_benchmark.json"

# Project-root /data/ directory (datasets live here, not in ml/data/)
_DATA_ROOT = _ML_ROOT.parent / "data"

N_PCA_COMPONENTS = 80
EMOTIONS_3 = ["positive", "neutral", "negative"]

# Welch PSD settings
BANDS = {
    "delta": (0.5, 4),
    "theta": (4,   8),
    "alpha": (8,  12),
    "beta":  (12, 30),
    "gamma": (30, 50),
}

# ---------------------------------------------------------------------------
# Feature extraction (85-dim — identical to _extract_muse_live_features)
# ---------------------------------------------------------------------------

def _band_power(sig: np.ndarray, fs: float) -> np.ndarray:
    """Return 5-element normalised band-power array [delta,theta,alpha,beta,gamma]."""
    n = len(sig)
    if n < 4:
        return np.zeros(5, dtype=np.float32)
    freqs, psd = _sig.welch(sig, fs, nperseg=min(256, n))
    total = np.sum(psd) + 1e-10
    out = []
    for lo, hi in BANDS.values():
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        out.append(float(np.sum(psd[idx]) / total) if len(idx) > 0 else 0.0)
    return np.array(out, dtype=np.float32)


def extract_features(eeg4ch: np.ndarray, fs: float,
                     win_s: float = 0.5, hop_s: float = 0.25) -> np.ndarray:
    """Extract 85-dim feature vector from (4, n_samples) EEG.

    Layout (matches _extract_muse_live_features in emotion_classifier.py):
      [0:80]  5 bands × 4 channels × 4 stats (mean, std, median, IQR) — band-major
      [80:85] 5 DASM features: mean(AF8_band) − mean(AF7_band) per band
    """
    WIN = int(win_s * fs)
    HOP = int(hop_s * fs)
    n   = eeg4ch.shape[1]

    powers: List[np.ndarray] = []
    for start in range(0, n - WIN + 1, HOP):
        sub = eeg4ch[:, start: start + WIN]   # (4, WIN)
        bp  = np.stack([_band_power(sub[ch], fs) for ch in range(4)])  # (4, 5)
        powers.append(bp.T)                    # (5, 4)

    if not powers:
        return np.zeros(85, dtype=np.float32)

    arr = np.array(powers, dtype=np.float32)  # (n_wins, 5, 4)
    feat: List[float] = []
    for b in range(5):
        for ch in range(4):
            vals = arr[:, b, ch]
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
    # DASM: AF8(ch2) − AF7(ch1) per band
    for b in range(5):
        af8 = arr[:, b, 2]; af8 = af8[np.isfinite(af8)]
        af7 = arr[:, b, 1]; af7 = af7[np.isfinite(af7)]
        feat.append(float(np.mean(af8) - np.mean(af7))
                    if len(af8) > 0 and len(af7) > 0 else 0.0)

    return np.array(feat, dtype=np.float32)  # 85 features


def _sliding_windows(eeg4ch: np.ndarray, fs: float,
                     win_s: float = 4.0, hop_s: float = 2.0,
                     artifact_uv: float = 75.0) -> np.ndarray:
    """Split (4, n_samples) EEG into 4-sec windows, return (n_windows, 85)."""
    WIN = int(win_s * fs)
    HOP = int(hop_s * fs)
    n   = eeg4ch.shape[1]
    rows = []
    for start in range(0, n - WIN + 1, HOP):
        epoch = eeg4ch[:, start: start + WIN]
        if np.any(np.abs(epoch) > artifact_uv):
            continue
        rows.append(extract_features(epoch, fs))
    return np.array(rows, dtype=np.float32) if rows else np.empty((0, 85), np.float32)


# ---------------------------------------------------------------------------
# DEAP loader
# ---------------------------------------------------------------------------

_DEAP_DIR = _DATA_ROOT / "deap" if (_DATA_ROOT / "deap").exists() else _ML_ROOT / "data" / "deap"
_DEAP_FS  = 128.0
# Muse-equivalent channels in DEAP 32-ch montage (0-indexed):
#   FP1=0≈AF7, FP2=16≈AF8, T7=12≈TP9, T8=28≈TP10
_DEAP_CH = [0, 16, 12, 28]    # [left-frontal, right-frontal, left-temporal, right-temporal]
# Reorder to TP9,AF7,AF8,TP10 (Muse 2 order)
_DEAP_CH_ORD = [2, 0, 1, 3]   # indices into _DEAP_CH → [T7→TP9, FP1→AF7, FP2→AF8, T8→TP10]

_DEAP_9TO3 = {0: 2, 1: 2, 2: 0, 3: 0}   # DEAP 0=LVLA=neg,1=LVHA=neg,2=HVLA=pos,3=HVHA=pos


def _load_deap_subject(pkl_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one DEAP subject .dat file (40 trials × 40 channels × 8064 samples)."""
    try:
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh, encoding="latin1")
        eeg  = data["data"][:, :32, :]    # (40, 32, 8064)
        labs = data["labels"]             # (40, 4) valence/arousal/dominance/liking
    except Exception as exc:
        log.debug("DEAP load error %s: %s", pkl_path.name, exc)
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)

    X_list, y_list = [], []
    for trial in range(eeg.shape[0]):
        valence = labs[trial, 0]   # 1-9 scale
        arousal = labs[trial, 1]
        # 3-class from valence + arousal quadrant
        if valence >= 5 and arousal >= 5:
            label = 0   # positive (HVHA)
        elif valence >= 5 and arousal < 5:
            label = 0   # positive (HVLA)
        elif valence < 5 and arousal < 5:
            label = 2   # negative (LVLA)
        else:
            label = 2   # negative (LVHA)

        # Select 4 channels → reorder to Muse layout
        ch_idx = [_DEAP_CH[i] for i in _DEAP_CH_ORD]
        seg    = eeg[trial][ch_idx].astype(np.float32)   # (4, 8064)

        feats = _sliding_windows(seg, _DEAP_FS)
        for f in feats:
            X_list.append(f)
            y_list.append(label)

    if not X_list:
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)
    return np.array(X_list, np.float32), np.array(y_list, np.int32)


def load_deap() -> Tuple[np.ndarray, np.ndarray]:
    dat_files = sorted(_DEAP_DIR.glob("s*.dat"))
    if not dat_files:
        log.warning("DEAP: no .dat files found in %s", _DEAP_DIR)
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)

    Xs, ys = [], []
    for f in dat_files:
        X, y = _load_deap_subject(f)
        if X.size > 0:
            Xs.append(X); ys.append(y)

    if not Xs:
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)

    log.info("DEAP: %d subjects → %d samples", len(Xs), sum(len(y) for y in ys))
    return np.vstack(Xs), np.concatenate(ys)


# ---------------------------------------------------------------------------
# DREAMER loader
# ---------------------------------------------------------------------------

_DREAMER_PATH = _DATA_ROOT / "dreamer" / "DREAMER.mat"
_DREAMER_FS   = 128.0
# Emotiv EPOC 14 channels: AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4
# Muse equivalents: AF7≈F7(ch1), AF8≈F8(ch12), TP9≈T7(ch4), TP10≈T8(ch9)
_DREAMER_CH = [1, 12, 4, 9]  # [AF7≈F7, AF8≈F8, TP9≈T7, TP10≈T8]


def load_dreamer() -> Tuple[np.ndarray, np.ndarray]:
    if not _DREAMER_PATH.exists() or not _SCI_OK:
        log.warning("DREAMER: %s not found — skipping", _DREAMER_PATH)
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)

    try:
        mat   = sio.loadmat(str(_DREAMER_PATH), struct_as_record=False, squeeze_me=True)
        dreamer = mat["DREAMER"]
        data_s  = dreamer.Data
    except Exception as exc:
        log.warning("DREAMER load error: %s", exc)
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)

    Xs, ys = [], []
    for subj_idx in range(len(data_s)):
        subj = data_s[subj_idx]
        eeg  = subj.EEG
        stimuli = eeg.stimuli
        n_trials = len(stimuli) if hasattr(stimuli, "__len__") else 0

        val_labels = subj.ScoreValence
        if not hasattr(val_labels, "__len__"):
            continue

        for t in range(n_trials):
            try:
                seg_raw = np.array(stimuli[t], dtype=np.float32)  # (n_samples, 14)
                if seg_raw.ndim != 2 or seg_raw.shape[1] < 14:
                    continue
                # Convert Emotiv EPOC raw ADC → µV: (raw - 4096) × 0.51 µV/bit
                seg_uv = (seg_raw - 4096.0) * 0.51
                seg = seg_uv[:, _DREAMER_CH].T  # (4, n_samples)
                val = float(val_labels[t]) if hasattr(val_labels, "__len__") else float(val_labels)
                label = 0 if val >= 3.5 else 2   # binary: pos/neg; neutral rare in DREAMER
                # Use relaxed threshold (500 µV) — DREAMER temporal channels have
                # large muscle artifacts; band-power features are ratio-normalised
                feats = _sliding_windows(seg, _DREAMER_FS, artifact_uv=500.0)
                for f in feats:
                    Xs.append(f); ys.append(label)
            except Exception:
                continue

    if not Xs:
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)
    log.info("DREAMER: %d samples", len(ys))
    return np.array(Xs, np.float32), np.array(ys, np.int32)


# ---------------------------------------------------------------------------
# GAMEEMO loader
# ---------------------------------------------------------------------------

_GAMEEMO_DIR = _DATA_ROOT / "gameemo"
_GAMEEMO_FS  = 128.0
# Emotiv EPOC 14-ch layout (0-indexed): AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4
# Muse equivalents: AF7≈F7(1), AF8≈F8(12), TP9≈T7(4), TP10≈T8(9)
_GAMEEMO_CH_ORDER = ["AF3", "F7", "F3", "FC5", "T7", "P7",
                     "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
_GAMEEMO_MUSE_KEYS = ["F7", "F8", "T7", "T8"]   # AF7, AF8, TP9, TP10 equivalents
# G1=boring→neutral, G2=calm→positive, G3=horror→negative, G4=fun→positive
_GAMEEMO_GAME_LABEL = {"G1": 1, "G2": 0, "G3": 2, "G4": 0}


def _gameemo_label_from_filename(stem: str) -> Optional[int]:
    """Extract game label from filename like 'S01G3AllChannels'."""
    import re
    m = re.search(r"G(\d)", stem)
    if m:
        key = f"G{m.group(1)}"
        return _GAMEEMO_GAME_LABEL.get(key)
    return None


def load_gameemo() -> Tuple[np.ndarray, np.ndarray]:
    gameemo_base = _GAMEEMO_DIR / "GAMEEMO"
    if not gameemo_base.exists():
        log.warning("GAMEEMO: data not found in %s — skipping", gameemo_base)
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)

    Xs, ys = [], []
    # Walk subject dirs like (S01), (S02), ...
    for subj_dir in sorted(gameemo_base.iterdir()):
        if not subj_dir.is_dir():
            continue
        mat_dir = subj_dir / "Preprocessed EEG Data" / ".mat format"
        if not mat_dir.exists():
            continue
        for mat_file in sorted(mat_dir.glob("*.mat")):
            label = _gameemo_label_from_filename(mat_file.stem)
            if label is None:
                continue
            try:
                mat = sio.loadmat(str(mat_file), struct_as_record=False,
                                  squeeze_me=True)
                # Each channel is a separate key: AF3, F7, T7, etc.
                channels = {k: np.array(mat[k], dtype=np.float32)
                            for k in _GAMEEMO_MUSE_KEYS if k in mat}
                if len(channels) < 4:
                    continue
                # Build (4, n_samples): order → TP9≈T7, AF7≈F7, AF8≈F8, TP10≈T8
                seg = np.stack([channels["T7"], channels["F7"],
                                channels["F8"], channels["T8"]])  # (4, n)
                feats = _sliding_windows(seg, _GAMEEMO_FS)
                for f in feats:
                    Xs.append(f); ys.append(label)
            except Exception:
                continue

    if not Xs:
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)
    log.info("GAMEEMO: %d samples", len(ys))
    return np.array(Xs, np.float32), np.array(ys, np.int32)


# ---------------------------------------------------------------------------
# EAV loader (raw EEG → 85-dim)
# ---------------------------------------------------------------------------
# EAV = EEG-Audio-Video Emotion Dataset (Lee et al., 2024, Scientific Data)
# 42 subjects, 30-ch EEG at 500 Hz, 5 emotions, 200 trials/subject.
#
# ⚠  RESTRICTED ACCESS — fill out DUA form at:
#    https://zenodo.org/records/10205702
# Once approved, download EEG files and place them at:
#    ml/data/eav/EAV/subject01/EEG/subject01_eeg.mat
#    ml/data/eav/EAV/subject01/EEG/subject01_eeg_label.mat
#    ... (repeat for subject02 … subject42)
#
# Mat file internals:
#   subject##_eeg.mat       → key 'seg1' (fallback 'seg'),  shape (10000, 30, 200)
#   subject##_eeg_label.mat → key 'label', one-hot (200, 5)
#   Emotion columns: 0=anger, 1=disgust, 2=fear, 3=joy, 4=neutral
#   3-class: anger/disgust/fear→negative, joy→positive, neutral→neutral

_EAV_DIR = _ML_ROOT / "data" / "eav" / "EAV"
_EAV_FS  = 500.0
# EAV 30-ch layout (standard 10-20, BrainAmp): channel order from the paper.
# Closest Muse equivalents (0-indexed in the 30-ch set):
#   F7(4)→AF7,  F8(14)→AF8,  T7(20)→TP9,  T8(21)→TP10
# Indices verified from standard 10-20 position lists for 30-ch montages.
_EAV_CH = [4, 14, 20, 21]   # [AF7≈F7, AF8≈F8, TP9≈T7, TP10≈T8]
# Reorder to Muse layout [TP9, AF7, AF8, TP10]
_EAV_CH_MUSE = [20, 4, 14, 21]

# 5-class → 3-class (column index in one-hot label matrix)
_EAV_5TO3 = {3: 0, 4: 1, 0: 2, 1: 2, 2: 2}   # joy→pos, neutral→neu, rest→neg


def load_eav() -> Tuple[np.ndarray, np.ndarray]:
    """Load EAV raw EEG and convert to 85-dim feature vectors.

    Skips silently if data directory is missing (dataset is restricted-access).
    """
    if not _EAV_DIR.exists():
        log.warning(
            "EAV: %s not found — skipping.\n"
            "  Get access at https://zenodo.org/records/10205702 then place\n"
            "  subject folders under ml/data/eav/EAV/",
            _EAV_DIR,
        )
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)

    subj_dirs = sorted(
        d for d in _EAV_DIR.iterdir()
        if d.is_dir() and d.name.lower().startswith("subject")
    )
    if not subj_dirs:
        log.warning("EAV: no subject* dirs in %s — skipping", _EAV_DIR)
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)

    Xs, ys = [], []
    n_loaded = 0

    for subj_dir in subj_dirs:
        eeg_dir   = subj_dir / "EEG"
        sid       = subj_dir.name          # e.g. "subject01"
        eeg_file  = eeg_dir / f"{sid}_eeg.mat"
        lab_file  = eeg_dir / f"{sid}_eeg_label.mat"
        if not (eeg_file.exists() and lab_file.exists()):
            continue

        try:
            eeg_mat = sio.loadmat(str(eeg_file),  struct_as_record=False, squeeze_me=True)
            lab_mat = sio.loadmat(str(lab_file), struct_as_record=False, squeeze_me=True)

            # EEG: key 'seg1' (shape 10000×30×200) or fallback 'seg'
            raw = None
            for key in ("seg1", "seg", "eeg", "data"):
                if key in eeg_mat:
                    raw = np.array(eeg_mat[key], dtype=np.float32)
                    break
            if raw is None:
                continue

            # Normalise to (n_trials, 30, 10000)
            if raw.ndim == 3:
                if raw.shape == (10000, 30, 200):
                    raw = raw.transpose(2, 1, 0)     # (200, 30, 10000)
                elif raw.shape == (200, 30, 10000):
                    pass
                elif raw.shape[1] == 30:
                    raw = raw.transpose(0, 1, 2)     # already (trials, 30, time)
                else:
                    continue
            else:
                continue

            # Labels: one-hot (200, 5) → argmax → 3-class
            lbl = np.array(lab_mat["label"], dtype=np.float32)
            if lbl.ndim == 2 and lbl.shape[1] == 5:
                cls5 = np.argmax(lbl, axis=1)        # (200,) values 0-4
            elif lbl.ndim == 1 and len(lbl) == 200:
                cls5 = lbl.astype(np.int32)
            else:
                continue
            labels3 = np.array([_EAV_5TO3[int(c)] for c in cls5], dtype=np.int32)

            # Channel selection → (n_trials, 4, 10000)
            if max(_EAV_CH_MUSE) >= raw.shape[1]:
                continue
            eeg4 = raw[:, _EAV_CH_MUSE, :]          # (200, 4, 10000)

            # Extract 85-dim features per trial via sliding windows
            for trial_idx in range(eeg4.shape[0]):
                seg = eeg4[trial_idx]                # (4, 10000)
                feats = _sliding_windows(seg, _EAV_FS, artifact_uv=200.0)
                for f in feats:
                    Xs.append(f)
                    ys.append(int(labels3[trial_idx]))

            n_loaded += 1
        except Exception as exc:
            log.debug("EAV %s error: %s", subj_dir.name, exc)
            continue

    if not Xs:
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)
    log.info("EAV: %d subjects → %d samples", n_loaded, len(ys))
    return np.array(Xs, np.float32), np.array(ys, np.int32)


# ---------------------------------------------------------------------------
# FACED loader (pre-extracted DE features → 85-dim)
# ---------------------------------------------------------------------------

_FACED_DE_DIR = _ML_ROOT / "data" / "faced" / "EEG_Features" / "DE"
# FACED 32-ch BrainProducts layout; Muse equivalents:
#   T7(23)→TP9, FP1(0)→AF7, FP2(2)→AF8, T8(24)→TP10
_FACED_MUSE_CH = [23, 0, 2, 24]   # [TP9, AF7, AF8, TP10]

# FACED 28-video emotion label sequence (9-class → integer)
# Negative first, neutral middle, positive last.
_FACED_VIDEO_LABELS = [
    1, 1, 1,      # disgust   (0–2)
    2, 2, 2,      # fear      (3–5)
    3, 3, 3,      # sadness   (6–8)
    4, 4, 4, 4,   # neutral   (9–12) — 4 clips
    5, 5, 5,      # amusement (13–15)
    7, 7, 7,      # joy       (16–18)
    6, 6, 6,      # inspiration (19–21)
    8, 8, 8,      # tenderness  (22–24)
    0, 0, 0,      # anger     (25–27)
]
# 9-class → 3-class: positive(0), neutral(1), negative(2)
_FACED_9TO3 = {5: 0, 6: 0, 7: 0, 8: 0,  4: 1,  0: 2, 1: 2, 2: 2, 3: 2}


def _faced_de_to_85dim(de_4ch_chunk: np.ndarray) -> np.ndarray:
    """Convert (4, n_windows, 5) DE chunk to 85-dim feature vector.

    Layout matches extract_features():
      [0:80]  5 bands × 4 channels × 4 stats (mean, std, median, IQR)
      [80:85] 5 DASM: mean(AF8_b) − mean(AF7_b) per band
              Channel order: [0]=TP9, [1]=AF7, [2]=AF8, [3]=TP10
    """
    feat: List[float] = []
    for b in range(5):
        for ch in range(4):
            vals = de_4ch_chunk[ch, :, b].astype(np.float64)
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
    # DASM: AF8(ch2) − AF7(ch1) per band
    for b in range(5):
        af8 = de_4ch_chunk[2, :, b]; af8 = af8[np.isfinite(af8)]
        af7 = de_4ch_chunk[1, :, b]; af7 = af7[np.isfinite(af7)]
        feat.append(float(np.mean(af8) - np.mean(af7))
                    if len(af8) > 0 and len(af7) > 0 else 0.0)
    return np.array(feat, dtype=np.float32)


def load_faced() -> Tuple[np.ndarray, np.ndarray]:
    """Load FACED DE features and convert to 85-dim vectors.

    Each video gives 14 overlapping 4-second chunks (hop=2) from 30 1-sec windows.
    """
    import pickle as _pkl
    pkl_files = sorted(_FACED_DE_DIR.glob("sub*.pkl.pkl"))
    if not pkl_files:
        log.warning("FACED: no sub*.pkl.pkl files in %s — skipping", _FACED_DE_DIR)
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)

    WIN, HOP = 4, 2   # 4-second chunk, 2-second hop (over 1-sec DE windows)
    Xs, ys = [], []

    for pkl_path in pkl_files:
        try:
            with open(pkl_path, "rb") as fh:
                data = _pkl.load(fh)            # (28, 32, 30, 5)
            if data.ndim != 4 or data.shape[1] < 25:
                continue
            # Select 4 Muse-equivalent channels → (28, 4, 30, 5)
            de = data[:, _FACED_MUSE_CH, :, :]  # (28, 4, 30, 5)

            for vid in range(de.shape[0]):
                if vid >= len(_FACED_VIDEO_LABELS):
                    break
                label_3 = _FACED_9TO3[_FACED_VIDEO_LABELS[vid]]
                n_t = de.shape[2]               # 30 time windows
                for t in range(0, n_t - WIN + 1, HOP):
                    chunk = de[vid, :, t:t+WIN, :]  # (4, WIN, 5)
                    feat  = _faced_de_to_85dim(chunk)
                    Xs.append(feat)
                    ys.append(label_3)
        except Exception:
            continue

    if not Xs:
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)
    log.info("FACED: %d subjects → %d samples", len(pkl_files), len(ys))
    return np.array(Xs, np.float32), np.array(ys, np.int32)


# ---------------------------------------------------------------------------
# SEED-IV loader (pre-extracted DE features → 85-dim vectors)
# ---------------------------------------------------------------------------

_SEED_IV_DIR = _ML_ROOT / "data" / "seed_iv"

# Session clip labels from ReadMe.txt
# 0=neutral, 1=sad, 2=fear, 3=happy
_SEED_IV_SESSION_LABELS: dict = {
    1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}

# 4-class → 3-class: neutral(0)→1, sad(1)→2, fear(2)→2, happy(3)→0
_SEED_IV_4TO3: dict = {0: 1, 1: 2, 2: 2, 3: 0}

# Select 4 channels closest to Muse 2 from 62-ch 10-20 layout:
#   T7(23)→TP9,  FP1(0)→AF7,  FP2(2)→AF8,  T8(31)→TP10
_SEED_IV_MUSE_CH = [23, 0, 2, 31]


def load_seed_iv() -> Tuple[np.ndarray, np.ndarray]:
    """Load SEED-IV pre-extracted DE features and convert to 85-dim vectors.

    Structure: eeg_feature_smooth/{session}/{subj}_{date}.mat
    Each .mat has keys de_LDS{1..24}: shape (62, n_windows, 5).
    Windows are already 4-sec; we aggregate WIN=4 consecutive windows
    (16 sec) with HOP=2 to compute statistics, same approach as FACED.
    """
    feat_dir = _SEED_IV_DIR / "eeg_feature_smooth"
    if not feat_dir.exists() or not _SCI_OK:
        log.warning("SEED-IV: %s not found — skipping", feat_dir)
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)

    WIN, HOP = 4, 2  # aggregate 4 × 4-sec windows → 16 sec, hop 2
    Xs, ys = [], []
    n_files = 0

    for session_id in sorted(_SEED_IV_SESSION_LABELS.keys()):
        session_dir = feat_dir / str(session_id)
        if not session_dir.exists():
            continue
        clip_labels = _SEED_IV_SESSION_LABELS[session_id]

        for mat_path in sorted(session_dir.glob("*.mat")):
            try:
                mat = sio.loadmat(str(mat_path), squeeze_me=True)
            except Exception:
                continue

            n_files += 1
            for trial_idx, raw_label in enumerate(clip_labels):
                trial_num = trial_idx + 1
                key = f"de_LDS{trial_num}"
                if key not in mat:
                    continue
                arr = mat[key]              # (62, n_windows, 5)
                if arr.ndim != 3 or arr.shape[0] < 32:
                    continue
                # Select 4 Muse-equivalent channels → (4, n_windows, 5)
                de4 = arr[_SEED_IV_MUSE_CH, :, :]
                label_3 = _SEED_IV_4TO3[int(raw_label)]
                n_t = de4.shape[1]
                for t in range(0, n_t - WIN + 1, HOP):
                    chunk = de4[:, t:t+WIN, :]   # (4, WIN, 5)
                    feat = _faced_de_to_85dim(chunk)
                    Xs.append(feat)
                    ys.append(label_3)

    if not Xs:
        log.warning("SEED-IV: 0 samples extracted (check data at %s)", feat_dir)
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)

    log.info("SEED-IV: %d files → %d samples", n_files, len(ys))
    return np.array(Xs, np.float32), np.array(ys, np.int32)


# ---------------------------------------------------------------------------
# EEG-ER loader (Emotiv EPOC 14-ch, raw µV, 128 Hz → 85-dim)
# ---------------------------------------------------------------------------

_EEG_ER_BASE = Path('/Users/sravyalu/.cache/kagglehub/datasets/'
                    'khan1803115/eeg-dataset-for-emotion-recognition/versions/1')
_EEG_ER_FS = 128.0
# Emotiv EPOC 14-ch order: AF3(0) F7(1) F3(2) FC5(3) T7(4) P7(5) O1(6)
#                           O2(7)  P8(8) T8(9) FC6(10) F4(11) F8(12) AF4(13)
# Muse-equivalent: [T7(4)→TP9, F7(1)→AF7, F8(12)→AF8, T8(9)→TP10]
_EEG_ER_CH = [4, 1, 12, 9]
# G1/G2 → negative(2), G3 → positive(0), G4 → neutral(1)
_EEG_ER_GAME_LABEL = {"G1": 2, "G2": 2, "G3": 0, "G4": 1}


def load_eeg_er() -> Tuple[np.ndarray, np.ndarray]:
    """Load EEG-ER dataset (100 CSVs, Emotiv EPOC 14-ch, 128 Hz)."""
    if not _EEG_ER_BASE.exists() or not _SCI_OK:
        log.warning("EEG-ER: %s not found — skipping", _EEG_ER_BASE)
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)

    import re
    Xs, ys = [], []
    csv_files = sorted(_EEG_ER_BASE.rglob("*AllChannels.csv"))
    if not csv_files:
        log.warning("EEG-ER: no *AllChannels.csv found in %s", _EEG_ER_BASE)
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)

    for csv_path in csv_files:
        m = re.search(r"G(\d)", csv_path.name)
        if not m:
            continue
        game_key = f"G{m.group(1)}"
        label = _EEG_ER_GAME_LABEL.get(game_key)
        if label is None:
            continue
        try:
            import pandas as _pd
            df = _pd.read_csv(csv_path, skiprows=1, header=None)
            df = df.apply(_pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
            if df.shape[1] < 14:
                continue
            data = df.iloc[:, :14].values.astype(np.float32).T  # (14, n_samples)
            if max(_EEG_ER_CH) >= data.shape[0]:
                continue
            seg = data[_EEG_ER_CH, :]  # (4, n_samples)
            feats = _sliding_windows(seg, _EEG_ER_FS)
            for f in feats:
                Xs.append(f); ys.append(label)
        except Exception:
            continue

    if not Xs:
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)
    log.info("EEG-ER: %d files → %d samples", len(csv_files), len(ys))
    return np.array(Xs, np.float32), np.array(ys, np.int32)


# ---------------------------------------------------------------------------
# STEW loader (Emotiv EPOC 14-ch, .mat, 128 Hz → 85-dim)
# ---------------------------------------------------------------------------

_STEW_BASE = Path('/Users/sravyalu/.cache/kagglehub/datasets/'
                  'mitulahirwal/mental-cognitive-workload-eeg-data-stew-dataset/versions/1')
_STEW_FS = 128.0
# Same Emotiv EPOC layout as EEG-ER
_STEW_CH = [4, 1, 12, 9]  # [T7→TP9, F7→AF7, F8→AF8, T8→TP10]
# Workload: 0=low→positive(0), 1=medium→neutral(1), 2=high→negative(2)
_STEW_3CLASS = {0: 0, 1: 1, 2: 2}


def load_stew() -> Tuple[np.ndarray, np.ndarray]:
    """Load STEW workload dataset (14-ch Emotiv EPOC, 45 segments, 128 Hz).

    dataset.mat: (14, 19200, 45) — channels × samples × segments
    class_012.mat: (45,) with labels 0/1/2

    Data is in raw Emotiv ADC counts (baseline ~4200 counts).
    Linear-detrend per channel before sliding-window extraction.
    """
    dataset_path = _STEW_BASE / "dataset.mat"
    labels_path  = _STEW_BASE / "class_012.mat"

    if not dataset_path.exists() or not _SCI_OK:
        log.warning("STEW: %s not found — skipping", _STEW_BASE)
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)

    try:
        dm = sio.loadmat(str(dataset_path))
        lm = sio.loadmat(str(labels_path))
        eeg_key = next(k for k in dm if not k.startswith("_"))
        lbl_key = next(k for k in lm if not k.startswith("_"))
        eeg_data = dm[eeg_key]          # (14, 19200, 45)
        raw_labels = lm[lbl_key].flatten()  # (45,)
    except Exception as e:
        log.warning("STEW: failed to load .mat files: %s", e)
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)

    from scipy.signal import detrend as _detrend
    Xs, ys = [], []
    n_channels, n_samples, n_segs = eeg_data.shape
    for seg_idx in range(n_segs):
        label = _STEW_3CLASS[int(raw_labels[seg_idx])]
        seg = eeg_data[_STEW_CH, :, seg_idx].astype(np.float32)  # (4, n_samples)
        # STEW raw ADC counts — linear-detrend each channel to remove DC + drift
        seg = _detrend(seg, axis=1).astype(np.float32)
        # Use relaxed artifact gate (2000 counts ≈ ~1 mV) to skip only gross outliers
        feats = _sliding_windows(seg, _STEW_FS, artifact_uv=2000.0)
        for f in feats:
            Xs.append(f); ys.append(label)

    if not Xs:
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)
    log.info("STEW: %d segments → %d samples", n_segs, len(ys))
    return np.array(Xs, np.float32), np.array(ys, np.int32)


# ---------------------------------------------------------------------------
# Muse-Subconscious loader (Muse band powers → 85-dim via 4-row windows)
# ---------------------------------------------------------------------------

_MUSE_SUB_ZIP = _ML_ROOT / "data" / "Muse_Subconscious.zip"
# Band-power column names in order: 5 bands × 4 channels
_MUSE_SUB_BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
_MUSE_SUB_CHS   = ["TP9", "AF7", "AF8", "TP10"]
_MUSE_SUB_COLS  = [f"{b}_{c}" for b in _MUSE_SUB_BANDS for c in _MUSE_SUB_CHS]
# Mellow/Concentration → 3-class labels
# High mellow (>0.6) → positive(0); High concentration (>0.6) → neutral(1); else → negative(2)


def load_muse_subconscious() -> Tuple[np.ndarray, np.ndarray]:
    """Load Muse Subconscious Decisions dataset (20 CSVs, 4-ch Muse band powers).

    CSVs are at ~256 Hz (one row per EEG sample with pre-computed band powers).
    Resample to 1 Hz by averaging per-second, then window at WIN=4 s, HOP=2 s.
    Each 4-sec window → (4_ch, 4_time, 5_bands) → _faced_de_to_85dim() → 85-dim.
    """
    if not _MUSE_SUB_ZIP.exists():
        log.warning("Muse-Subconscious: %s not found — skipping", _MUSE_SUB_ZIP)
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)

    import zipfile
    WIN, HOP = 4, 2   # seconds after resampling to 1 Hz
    Xs, ys = [], []
    n_files = 0

    with zipfile.ZipFile(_MUSE_SUB_ZIP, "r") as zf:
        # Skip macOS metadata entries (.__xxx files)
        csv_names = [f for f in zf.namelist()
                     if f.endswith(".csv") and "/Muse/" in f and "/._" not in f]
        for csv_name in csv_names:
            try:
                import pandas as _pd
                with zf.open(csv_name) as fh:
                    df = _pd.read_csv(fh)

                feat_cols = [c for c in _MUSE_SUB_COLS if c in df.columns]
                if len(feat_cols) < 20:
                    continue
                if "Mellow" not in df.columns or "Concentration" not in df.columns:
                    continue

                # Parse timestamps and resample to 1 Hz (average per second)
                df["_ts"] = _pd.to_datetime(df["TimeStamp"], errors="coerce")
                df = df.dropna(subset=["_ts"])
                if len(df) < WIN * 2:
                    continue
                df["_sec"] = df["_ts"].dt.floor("1s")

                # Keep only good-signal rows (HeadBandOn=True or column absent)
                if "HeadBandOn" in df.columns:
                    df = df[df["HeadBandOn"].astype(str).str.lower()
                            .isin(["true", "1", "yes", "1.0"])]
                if len(df) < WIN * 2:
                    continue

                agg_cols = feat_cols + ["Mellow", "Concentration"]
                resampled = (
                    df.groupby("_sec")[agg_cols]
                    .mean()
                    .sort_index()
                    .reset_index(drop=True)
                )
                # Apply numeric coercion and drop NaN rows
                for col in agg_cols:
                    resampled[col] = _pd.to_numeric(resampled[col], errors="coerce")
                resampled = resampled.dropna(subset=feat_cols).reset_index(drop=True)

                if len(resampled) < WIN:
                    continue

                data   = resampled[feat_cols].values.astype(np.float32)  # (n_sec, 20)
                mellow = resampled["Mellow"].values
                conc   = resampled["Concentration"].values

                n_files += 1
                for start in range(0, len(data) - WIN + 1, HOP):
                    window = data[start:start+WIN]                  # (4, 20)
                    w_mel = np.nanmean(mellow[start:start+WIN])
                    w_con = np.nanmean(conc[start:start+WIN])
                    if np.isnan(w_mel) and np.isnan(w_con):
                        continue
                    w_mel = 0.0 if np.isnan(w_mel) else w_mel
                    w_con = 0.0 if np.isnan(w_con) else w_con

                    if w_mel > 0.6:
                        label = 0   # positive (mellow/relaxed)
                    elif w_con > 0.6:
                        label = 1   # neutral (focused/concentrated)
                    else:
                        label = 2   # negative (neither mellow nor focused)

                    # Reshape (4_rows, 20_cols) → (4_ch, 4_time, 5_bands)
                    arr = window.reshape(WIN, 5, 4).transpose(2, 0, 1)  # (4, 4, 5)
                    if not np.all(np.isfinite(arr)):
                        continue
                    feat = _faced_de_to_85dim(arr)
                    Xs.append(feat); ys.append(label)
            except Exception:
                continue

    if not Xs:
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)
    log.info("Muse-Subconscious: %d files → %d samples", n_files, len(ys))
    return np.array(Xs, np.float32), np.array(ys, np.int32)


# ---------------------------------------------------------------------------
# DENS loader (raw EEG → 85-dim features)
# ---------------------------------------------------------------------------

_DENS_DIR  = _ML_ROOT / "data" / "dens"
_DENS_FS   = 250.0
_DENS_NCH  = 132
# Muse-equivalent channels (EGI 128-ch, verified by 3D coord matching):
#   TP9=E48(ch47), AF7=E32(ch31), AF8=E1(ch0), TP10=E119(ch118)
_DENS_CH = [47, 31, 0, 118]  # [TP9, AF7, AF8, TP10]

import csv as _csv


def _load_dens_fdt(fdt_path: Path, n_ch: int, n_samp: int) -> Optional[np.ndarray]:
    """Load full or partial DENS .fdt binary (float32, channel-major)."""
    try:
        raw = np.fromfile(fdt_path, dtype=np.float32)
        if raw.size == n_ch * n_samp:
            return raw.reshape((n_ch, n_samp))
        n_actual = raw.size // n_ch
        if n_actual < 1:
            return None
        return raw[: n_actual * n_ch].reshape((n_ch, n_actual))
    except Exception:
        return None


def _dens_stm_onsets(eeg_struct) -> List[int]:
    events = eeg_struct.event if hasattr(eeg_struct.event, "__len__") else []
    return sorted(int(float(ev.latency)) for ev in events
                  if str(getattr(ev, "type", "")) == "stm")


def _dens_beh_valences(beh_path: Path) -> List[float]:
    try:
        with open(beh_path, newline="") as fh:
            rows = list(_csv.DictReader(fh, delimiter="\t"))
        return [float(r["valence"]) for r in rows if "valence" in r]
    except Exception:
        return []


def load_dens() -> Tuple[np.ndarray, np.ndarray]:
    sub_dirs = sorted(d for d in _DENS_DIR.iterdir()
                      if d.is_dir() and d.name.startswith("sub-"))
    if not sub_dirs:
        log.warning("DENS: no sub-* dirs in %s", _DENS_DIR)
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)

    Xs, ys = [], []
    n_loaded = 0

    for sub_dir in sub_dirs:
        eeg_dir = sub_dir / "eeg"
        beh_dir = sub_dir / "beh"
        set_files = list(eeg_dir.glob("*_eeg.set"))
        beh_files = list(beh_dir.glob("*_beh.tsv"))
        fdt_files = list(eeg_dir.glob("*.fdt"))
        if not (set_files and beh_files and fdt_files):
            continue

        try:
            mat = sio.loadmat(str(set_files[0]), struct_as_record=False, squeeze_me=True)
            eeg_s = mat["EEG"]
            n_ch  = int(eeg_s.nbchan)
            n_samp = int(eeg_s.pnts)
        except Exception:
            continue

        if max(_DENS_CH) >= n_ch:
            continue

        data = _load_dens_fdt(fdt_files[0], n_ch, n_samp)
        if data is None:
            continue

        n_avail   = data.shape[1]
        stm_onsets = _dens_stm_onsets(eeg_s)
        valences   = _dens_beh_valences(beh_files[0])

        WIN = int(4.0 * _DENS_FS)
        HOP = int(2.0 * _DENS_FS)
        n_trials_used = 0

        for i, onset in enumerate(stm_onsets):
            if onset >= n_avail:
                continue
            if i >= len(valences):
                break
            val = valences[i]
            label = 0 if val >= 6.0 else (2 if val <= 4.0 else 1)

            end   = min(onset + int(60.0 * _DENS_FS), n_avail)
            seg   = data[_DENS_CH, onset:end].astype(np.float32)  # (4, T)
            # Re-reference to linked mastoid (TP9=ch0, TP10=ch3)
            mastoid = (seg[0] + seg[3]) / 2
            seg = seg - mastoid

            # DENS data is in raw EGI units (not µV) — skip amplitude threshold.
            # Band-power features are ratio-normalised so absolute scale is irrelevant.
            n_s = seg.shape[1]
            start = 0
            while start + WIN <= n_s:
                epoch = seg[:, start: start + WIN]
                Xs.append(extract_features(epoch, _DENS_FS))
                ys.append(label)
                start += HOP
            n_trials_used += 1

        if n_trials_used > 0:
            n_loaded += 1

    if not Xs:
        return np.empty((0, 85), np.float32), np.empty(0, np.int32)
    log.info("DENS: %d subjects → %d samples", n_loaded, len(ys))
    return np.array(Xs, np.float32), np.array(ys, np.int32)


# ---------------------------------------------------------------------------
# SMOTE
# ---------------------------------------------------------------------------

def _smote(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not _SMOTE_OK:
        return X, y
    counts = {int(c): int((y == c).sum()) for c in np.unique(y)}
    if min(counts.values()) < 2:
        return X, y
    k = max(1, min(5, min(counts.values()) - 1))
    try:
        sm = SMOTE(k_neighbors=k, random_state=42)
        return sm.fit_resample(X, y)
    except Exception as exc:
        log.warning("SMOTE failed (%s) — using raw data", exc)
        return X, y


# ---------------------------------------------------------------------------
# Global PCA pipeline
# ---------------------------------------------------------------------------

def build_pca_pipeline(X: np.ndarray, n_comp: int = N_PCA_COMPONENTS):
    """Fit StandardScaler → PCA on X, return (scaler, pca, X_transformed)."""
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)
    n_actual = min(n_comp, X.shape[1], X.shape[0])
    pca     = PCA(n_components=n_actual, random_state=42)
    X_pca   = pca.fit_transform(X_sc)
    var_exp = pca.explained_variance_ratio_.sum() * 100
    log.info("PCA: %d → %d dims (%.1f%% variance explained)", X.shape[1], n_actual, var_exp)
    return scaler, pca, X_pca


def apply_pca_pipeline(X: np.ndarray, scaler, pca) -> np.ndarray:
    return pca.transform(scaler.transform(X))


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate(X: np.ndarray, y: np.ndarray,
                       scaler, pca) -> dict:
    """Cross-validate LGBM on PCA-transformed features, then retrain on full data."""
    X_pca = apply_pca_pipeline(X, scaler, pca)

    # ── 5-fold stratified CV ──
    cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_pca, y)):
        clf = lgb.LGBMClassifier(
            n_estimators=800, num_leaves=63, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            class_weight="balanced", random_state=42,
            n_jobs=-1, verbose=-1,
        )
        clf.fit(X_pca[tr_idx], y[tr_idx])
        score = accuracy_score(y[va_idx], clf.predict(X_pca[va_idx]))
        cv_scores.append(score)
        log.info("  Fold %d: %.2f%%", fold + 1, score * 100)

    cv_mean = float(np.mean(cv_scores))
    cv_std  = float(np.std(cv_scores))
    log.info("CV: %.2f%% ± %.2f%%", cv_mean * 100, cv_std * 100)

    # ── Retrain on full data ──
    final_clf = lgb.LGBMClassifier(
        n_estimators=1000, num_leaves=63, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        class_weight="balanced", random_state=42,
        n_jobs=-1, verbose=-1,
    )
    final_clf.fit(X_pca, y)
    train_acc = accuracy_score(y, final_clf.predict(X_pca))

    return {
        "model": final_clf,
        "cv_mean": cv_mean,
        "cv_std":  cv_std,
        "train_acc": train_acc,
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_artifacts(result: dict, scaler, pca,
                   n_samples: int, datasets_used: List[str]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "model":          result["model"],
        "scaler":         scaler,           # StandardScaler (85→85)
        "pca":            pca,              # PCA (85→80)
        "n_features_raw": 85,
        "n_pca_features": pca.n_components_,
        "n_classes":      3,
        "class_names":    EMOTIONS_3,
        "cv_accuracy":    result["cv_mean"],
        "datasets":       datasets_used,
        "label_map":      {0: "positive", 1: "neutral", 2: "negative"},
        "feature_layout": (
            "85-dim: 5bands × 4ch × 4stats (mean,std,med,IQR) = 80 "
            "+ 5 DASM (AF8−AF7 per band). "
            "ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10."
        ),
    }
    with open(MODEL_OUT, "wb") as fh:
        import pickle
        pickle.dump(payload, fh, protocol=4)
    log.info("Model saved → %s", MODEL_OUT)

    benchmark = {
        "dataset":          "+".join(datasets_used),
        "n_samples":        n_samples,
        "classifier":       "LightGBM (global PCA 85→80)",
        "accuracy":         round(result["cv_mean"], 4),   # CV accuracy
        "cv_accuracy_mean": round(result["cv_mean"], 4),
        "cv_accuracy_std":  round(result["cv_std"],  4),
        "train_accuracy":   round(result["train_acc"], 4),
        "n_features_raw":   85,
        "n_pca_features":   int(pca.n_components_),
        "n_classes":        3,
        "classes":          EMOTIONS_3,
        "notes": (
            "Single global PCA (85→80) fitted on ALL training data — "
            "scaler+PCA+LGBM all saved inside the .pkl for lossless inference. "
            "3→6 class expansion via ancillary EEG features at inference. "
            "Gamma features zeroed for consumer devices (Muse 2 / Muse S)."
        ),
    }
    with open(BENCH_OUT, "w") as fh:
        json.dump(benchmark, fh, indent=2)
    log.info("Benchmark saved → %s", BENCH_OUT)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    log.info("=== Mega LGBM Unified Cross-Dataset Trainer ===")
    log.info("Strategy: global PCA (85→80) + LightGBM, all artifacts saved")

    # ── 1. Load all datasets ──
    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    datasets_used: List[str] = []

    log.info("── Loading DEAP ──")
    X_d, y_d = load_deap()
    if X_d.size > 0:
        all_X.append(X_d); all_y.append(y_d)
        datasets_used.append("DEAP")
        log.info("DEAP loaded: %d samples", len(y_d))

    log.info("── Loading DREAMER ──")
    X_dr, y_dr = load_dreamer()
    if X_dr.size > 0:
        all_X.append(X_dr); all_y.append(y_dr)
        datasets_used.append("DREAMER")
        log.info("DREAMER loaded: %d samples", len(y_dr))

    log.info("── Loading GAMEEMO ──")
    X_g, y_g = load_gameemo()
    if X_g.size > 0:
        all_X.append(X_g); all_y.append(y_g)
        datasets_used.append("GAMEEMO")
        log.info("GAMEEMO loaded: %d samples", len(y_g))

    log.info("── Loading DENS ──")
    X_dn, y_dn = load_dens()
    if X_dn.size > 0:
        all_X.append(X_dn); all_y.append(y_dn)
        datasets_used.append("DENS")
        log.info("DENS loaded: %d samples", len(y_dn))

    log.info("── Loading FACED ──")
    X_fc, y_fc = load_faced()
    if X_fc.size > 0:
        all_X.append(X_fc); all_y.append(y_fc)
        datasets_used.append("FACED")
        log.info("FACED loaded: %d samples", len(y_fc))

    log.info("── Loading EAV ──")
    X_eav, y_eav = load_eav()
    if X_eav.size > 0:
        all_X.append(X_eav); all_y.append(y_eav)
        datasets_used.append("EAV")
        log.info("EAV loaded: %d samples", len(y_eav))

    log.info("── Loading SEED-IV ──")
    X_sv, y_sv = load_seed_iv()
    if X_sv.size > 0:
        all_X.append(X_sv); all_y.append(y_sv)
        datasets_used.append("SEED-IV")
        log.info("SEED-IV loaded: %d samples", len(y_sv))

    log.info("── Loading EEG-ER ──")
    X_er, y_er = load_eeg_er()
    if X_er.size > 0:
        all_X.append(X_er); all_y.append(y_er)
        datasets_used.append("EEG-ER")
        log.info("EEG-ER loaded: %d samples", len(y_er))

    log.info("── Loading STEW ──")
    X_st, y_st = load_stew()
    if X_st.size > 0:
        all_X.append(X_st); all_y.append(y_st)
        datasets_used.append("STEW")
        log.info("STEW loaded: %d samples", len(y_st))

    log.info("── Loading Muse-Subconscious ──")
    X_ms, y_ms = load_muse_subconscious()
    if X_ms.size > 0:
        all_X.append(X_ms); all_y.append(y_ms)
        datasets_used.append("Muse-Sub")
        log.info("Muse-Subconscious loaded: %d samples", len(y_ms))

    if not all_X:
        log.error("No data loaded — check data directories.")
        sys.exit(1)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    n_raw = len(y)
    log.info("Combined: %d samples, %d features | datasets: %s",
             n_raw, X.shape[1], datasets_used)

    class_counts = {EMOTIONS_3[i]: int((y == i).sum()) for i in range(3)}
    log.info("Class counts: %s", class_counts)

    # ── 2. SMOTE ──
    X_bal, y_bal = _smote(X, y)
    log.info("After SMOTE: %d samples", len(y_bal))

    # ── 3. Global PCA pipeline ──
    log.info("── Fitting global scaler + PCA ──")
    scaler, pca, _ = build_pca_pipeline(X_bal)

    # ── 4. Train + CV ──
    log.info("── Training LightGBM with 5-fold CV ──")
    result = train_and_evaluate(X_bal, y_bal, scaler, pca)
    log.info("Final: CV %.2f%% ± %.2f%% | Train %.2f%%",
             result["cv_mean"] * 100, result["cv_std"] * 100,
             result["train_acc"] * 100)

    # ── 5. Save ──
    save_artifacts(result, scaler, pca, n_raw, datasets_used)

    elapsed = time.time() - t0
    log.info("Done in %.1f seconds.", elapsed)
    log.info("Model path: %s", MODEL_OUT)
    log.info("CV accuracy: %.2f%% — will be loaded automatically by emotion_classifier.py", result["cv_mean"] * 100)


if __name__ == "__main__":
    main()
