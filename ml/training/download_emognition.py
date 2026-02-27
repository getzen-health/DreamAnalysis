"""Emognition Dataset: download instructions and loader.

The Emognition dataset (Muse 2, 43 participants, 9 emotion labels) was recorded
using a consumer-grade Muse 2 headband at 256 Hz.  It is one of the few public
EEG emotion datasets captured on Muse-class hardware, making it directly
applicable to this project.

Dataset Reference:
    Sawicki, D. et al. (2023). Emognition dataset: EEG data for emotion recognition.
    Harvard Dataverse. https://doi.org/10.7910/DVN/R9WAF4

Dataset URL:
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/R9WAF4

Since Harvard Dataverse requires a browser login for bulk download, no automated
download is available.  Run this file for step-by-step instructions:

    python -m training.download_emognition

After placing the files, call:

    X, y, feature_names = load_emognition("ml/data/emognition")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

DATASET_URL = "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/R9WAF4"
DOI         = "doi:10.7910/DVN/R9WAF4"
EMOGNITION_FS = 256          # Hz — Muse 2 sampling rate
N_CHANNELS    = 4            # AF7, AF8, TP9, TP10
N_EMOTIONS    = 9            # original label count
EPOCH_SEC     = 4.0          # seconds per feature-extraction window
EPOCH_SAMPLES = int(EMOGNITION_FS * EPOCH_SEC)  # 1024 samples per epoch

# Emognition 9-class → 3-class mapping
# Labels: ANGER, DISGUST, FEAR, SADNESS, SURPRISE, AWE, CONTENTMENT, AMUSEMENT, HAPPINESS
_LABEL_MAP_9_TO_3: dict[str, int] = {
    "anger":      0,  # negative
    "disgust":    0,  # negative
    "fear":       0,  # negative
    "sadness":    0,  # negative
    "surprise":   1,  # neutral (ambiguous valence)
    "awe":        1,  # neutral (ambiguous valence)
    "contentment": 2, # positive
    "amusement":  2,  # positive
    "happiness":  2,  # positive
}
_CLASS_NAMES = ["negative", "neutral", "positive"]

# Muse 2 BrainFlow channel order in the Emognition CSV files (BrainFlow default):
#   TP9, AF7, AF8, TP10
_CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]

# Expected CSV column prefixes in the Emognition files
_EEG_COLUMN_PREFIXES = ["AF7", "AF8", "TP9", "TP10", "eeg_af7", "eeg_af8", "eeg_tp9", "eeg_tp10"]


# ── Manual download instructions ──────────────────────────────────────────────

def print_download_instructions(data_dir: Optional[str] = None) -> None:
    """Print step-by-step instructions for downloading the Emognition dataset."""
    target = Path(data_dir) if data_dir else Path("ml/data/emognition")

    instructions = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    EMOGNITION DATASET — MANUAL DOWNLOAD                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Dataset: Emognition (Muse 2 EEG, 43 participants, 9 emotion labels, 256 Hz)
DOI:     {DOI}
URL:     {DATASET_URL}

Harvard Dataverse requires a browser session for bulk file access.
Follow these steps:

STEP 1 — Create a Harvard Dataverse account (free)
    Open: https://dataverse.harvard.edu
    Click "Sign Up" in the top-right corner.
    Verify your email address.

STEP 2 — Navigate to the dataset
    Open: {DATASET_URL}
    The dataset is titled "Emognition: EEG dataset for emotion recognition".

STEP 3 — Download the data files
    On the dataset page, click "Access Dataset" → "Original Format ZIP"
    OR download individual files from the "Files" tab.

    Required files (download all CSV files):
      ├── participant_*.csv       (one file per participant, contains raw EEG)
      ├── labels.csv              (emotion labels per trial)
      └── metadata.csv            (participant demographics)

    If the dataset has changed structure, look for any .csv or .zip files
    listed under the "Files" tab and download them all.

STEP 4 — Place files in the correct directory
    Create the directory and extract/move files:

      mkdir -p {target}
      # Move downloaded files into: {target}/
      # Expected structure:
      #   {target}/participant_01.csv
      #   {target}/participant_02.csv
      #   ... (43 participant files)
      #   {target}/labels.csv
      #   {target}/metadata.csv

STEP 5 — Verify
    Run this script to load and extract features:

      cd ml
      python -m training.download_emognition --verify --data-dir {target}

ALTERNATIVE: pyDataverse (automated, requires account credentials)
    pip install pyDataverse
    python -c "
    from pyDataverse.api import NativeApi, DataAccessApi
    # Requires an API token from your Harvard Dataverse account settings
    api = NativeApi('https://dataverse.harvard.edu', '<YOUR_API_TOKEN>')
    ds  = api.get_dataset('{DOI}')
    "

═══════════════════════════════════════════════════════════════════════════════
Once downloaded, run the loader:

    from training.download_emognition import load_emognition
    X, y, feature_names = load_emognition('{target}')
    print(f'Loaded {{X.shape[0]}} samples, {{X.shape[1]}} features')
═══════════════════════════════════════════════════════════════════════════════
"""
    print(instructions)


# ── CSV parsing helpers ────────────────────────────────────────────────────────

def _find_eeg_columns(columns: list[str]) -> list[str]:
    """Return EEG column names from a CSV header, tolerating naming variants."""
    import re
    eeg_cols: list[str] = []
    patterns = [
        re.compile(r"^(AF7|af7)$"),
        re.compile(r"^(AF8|af8)$"),
        re.compile(r"^(TP9|tp9)$"),
        re.compile(r"^(TP10|tp10)$"),
        re.compile(r"eeg[_\-]?(AF7|af7)", re.IGNORECASE),
        re.compile(r"eeg[_\-]?(AF8|af8)", re.IGNORECASE),
        re.compile(r"eeg[_\-]?(TP9|tp9)", re.IGNORECASE),
        re.compile(r"eeg[_\-]?(TP10|tp10)", re.IGNORECASE),
        re.compile(r"channel[_\-]?[0-3]$"),
        re.compile(r"ch[0-3]$"),
    ]
    for col in columns:
        if any(p.search(col) for p in patterns):
            eeg_cols.append(col)
    return eeg_cols[:4]  # take at most 4 channels


def _map_emotion_label(raw_label: str) -> Optional[int]:
    """Convert a raw Emognition label string to 3-class integer, or None if unknown."""
    normalised = raw_label.strip().lower().replace("-", "_").replace(" ", "_")
    return _LABEL_MAP_9_TO_3.get(normalised)


def _epoch_and_extract(
    eeg_data: np.ndarray,
    fs: float,
    epoch_samples: int,
    hop_samples: int,
) -> list[np.ndarray]:
    """Slice a (n_channels, n_samples) array into overlapping 4-sec epochs."""
    _, n_total = eeg_data.shape
    epochs: list[np.ndarray] = []
    start = 0
    while start + epoch_samples <= n_total:
        epochs.append(eeg_data[:, start : start + epoch_samples])
        start += hop_samples
    return epochs


# ── Main loader ───────────────────────────────────────────────────────────────

def load_emognition(
    data_dir: str | Path,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[list[str]]]:
    """Load Emognition CSV files and return feature matrix (X, y, feature_names).

    Reads per-participant CSV files from *data_dir*, slices 4-second epochs
    (50 % overlap), extracts 85-dimensional feature vectors via
    ``extract_features_multichannel()``, and maps the 9 original emotion
    labels to 3 classes (negative / neutral / positive).

    Args:
        data_dir: Path to the directory containing Emognition CSV files.
                  If the directory does not exist, download instructions are
                  printed and the function returns (None, None, None).

    Returns:
        Tuple ``(X, y, feature_names)`` where:
          - ``X`` is ``(n_samples, n_features)`` float32 ndarray
          - ``y`` is ``(n_samples,)`` int ndarray with values 0 / 1 / 2
          - ``feature_names`` is a list of feature name strings

        Returns ``(None, None, None)`` if the dataset is not found.
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        print(
            f"\n[Emognition] Data directory not found: {data_path}\n"
            "The dataset must be downloaded manually.\n"
        )
        print_download_instructions(str(data_path))
        return None, None, None

    csv_files = sorted(data_path.glob("*.csv"))
    participant_files = [f for f in csv_files if f.stem.lower().startswith("participant")]

    if not participant_files:
        # Try any CSV that is not labels/metadata
        participant_files = [
            f for f in csv_files
            if f.stem.lower() not in ("labels", "metadata", "readme")
        ]

    if not participant_files:
        print(
            f"\n[Emognition] No participant CSV files found in {data_path}.\n"
            "Expected files named participant_*.csv\n"
        )
        print_download_instructions(str(data_path))
        return None, None, None

    try:
        import pandas as pd
    except ImportError:
        log.error("pandas is required for load_emognition. Install with: pip install pandas")
        return None, None, None

    try:
        from processing.eeg_processor import extract_features_multichannel, preprocess
    except ImportError:
        log.error(
            "processing.eeg_processor not found. "
            "Run this script from the ml/ directory."
        )
        return None, None, None

    # Load optional label file
    label_df: Optional[pd.DataFrame] = None
    label_path = data_path / "labels.csv"
    if label_path.exists():
        try:
            label_df = pd.read_csv(label_path)
        except Exception as exc:
            log.warning("Could not read labels.csv: %s", exc)

    hop_samples   = EPOCH_SAMPLES // 2   # 50 % overlap
    all_X:  list[np.ndarray] = []
    all_y:  list[int]        = []
    n_skipped_label = 0
    n_skipped_length = 0

    for pfile in participant_files:
        try:
            df = pd.read_csv(pfile)
        except Exception as exc:
            log.warning("Skipping %s — could not read: %s", pfile.name, exc)
            continue

        # Find EEG columns
        eeg_cols = _find_eeg_columns(list(df.columns))
        if len(eeg_cols) < 4:
            # Fall back: take first 4 numeric columns after timestamp
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "timestamp" in [c.lower() for c in numeric_cols]:
                numeric_cols = [c for c in numeric_cols if "timestamp" not in c.lower()]
            eeg_cols = numeric_cols[:4]

        if len(eeg_cols) < 4:
            log.warning("Skipping %s — could not identify 4 EEG columns (found %d)", pfile.name, len(eeg_cols))
            continue

        # Determine emotion label for this participant file
        # Strategy: check for a "label", "emotion", or "stimulus" column
        label_col = next(
            (c for c in df.columns if c.lower() in ("label", "emotion", "stimulus", "condition")),
            None,
        )

        # Extract EEG array — shape (n_channels, n_samples)
        eeg_raw = df[eeg_cols].values.T.astype(np.float32)  # (4, n_samples)

        if label_col is not None:
            # Per-row labels: group by trial/label segment
            label_series = df[label_col].astype(str)
            unique_labels = label_series.unique()

            for emotion_str in unique_labels:
                mask = label_series == emotion_str
                segment = eeg_raw[:, mask.values]
                class_idx = _map_emotion_label(emotion_str)
                if class_idx is None:
                    n_skipped_label += 1
                    continue
                if segment.shape[1] < EPOCH_SAMPLES:
                    n_skipped_length += 1
                    continue

                epochs = _epoch_and_extract(segment, EMOGNITION_FS, EPOCH_SAMPLES, hop_samples)
                for epoch in epochs:
                    try:
                        # preprocess each channel independently
                        processed = np.stack([
                            preprocess(epoch[ch], EMOGNITION_FS)
                            for ch in range(epoch.shape[0])
                        ], axis=0)
                        feats = extract_features_multichannel(processed, EMOGNITION_FS)
                        fv    = np.array(list(feats.values()), dtype=np.float32)
                        all_X.append(fv)
                        all_y.append(class_idx)
                    except Exception as exc:
                        log.debug("Feature extraction failed for epoch: %s", exc)

        else:
            # No inline label column — try to derive from filename
            # e.g. participant_01_happiness.csv or row label_df lookup
            emotion_str = None
            for part in pfile.stem.split("_"):
                if _map_emotion_label(part) is not None:
                    emotion_str = part
                    break

            class_idx = _map_emotion_label(emotion_str) if emotion_str else None
            if class_idx is None:
                log.debug("Could not determine label for %s — skipping", pfile.name)
                n_skipped_label += 1
                continue

            if eeg_raw.shape[1] < EPOCH_SAMPLES:
                n_skipped_length += 1
                continue

            epochs = _epoch_and_extract(eeg_raw, EMOGNITION_FS, EPOCH_SAMPLES, hop_samples)
            for epoch in epochs:
                try:
                    processed = np.stack([
                        preprocess(epoch[ch], EMOGNITION_FS)
                        for ch in range(epoch.shape[0])
                    ], axis=0)
                    feats = extract_features_multichannel(processed, EMOGNITION_FS)
                    fv    = np.array(list(feats.values()), dtype=np.float32)
                    all_X.append(fv)
                    all_y.append(class_idx)
                except Exception as exc:
                    log.debug("Feature extraction failed for epoch: %s", exc)

    if not all_X:
        log.error(
            "No samples extracted. "
            "Skipped %d segments due to unknown labels, %d due to short length.",
            n_skipped_label, n_skipped_length,
        )
        return None, None, None

    X = np.stack(all_X, axis=0)   # (n_samples, n_features)
    y = np.array(all_y, dtype=np.int32)

    # Derive feature names from the last extracted dict
    try:
        from processing.eeg_processor import extract_features_multichannel as _efm
        _dummy_sig = np.zeros((4, EPOCH_SAMPLES), dtype=np.float32)
        _dummy_feats = _efm(_dummy_sig, EMOGNITION_FS)
        feature_names = list(_dummy_feats.keys())
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    print(
        f"[Emognition] Loaded {X.shape[0]} samples, {X.shape[1]} features "
        f"from {len(participant_files)} participant files."
    )
    print(f"  Classes:  {_CLASS_NAMES}")
    for cls_idx, cls_name in enumerate(_CLASS_NAMES):
        n = int((y == cls_idx).sum())
        print(f"    {cls_name:10s}: {n} samples")

    if n_skipped_label:
        print(f"  Skipped {n_skipped_label} segments with unrecognised emotion labels.")
    if n_skipped_length:
        print(f"  Skipped {n_skipped_length} segments shorter than {EPOCH_SEC}s.")

    return X, y, feature_names


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Emognition dataset download instructions and loader."
    )
    parser.add_argument(
        "--data-dir",
        default="data/emognition",
        help="Path to Emognition data directory (default: data/emognition)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Attempt to load and print dataset statistics.",
    )
    args = parser.parse_args()

    if args.verify:
        logging.basicConfig(level=logging.INFO)
        X, y, names = load_emognition(args.data_dir)
        if X is not None:
            print(f"\nVerification passed: X={X.shape}, y={y.shape}, features={len(names)}")
        else:
            print("\nVerification failed — dataset not loaded.")
            sys.exit(1)
    else:
        print_download_instructions(args.data_dir)
