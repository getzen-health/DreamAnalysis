"""YASA (Yet Another Spindle Algorithm) wrapper for advanced sleep analysis (#527).

Provides:
  - detect_spindles(): sleep spindle detection (11-16 Hz bursts in N2/N3)
  - detect_slow_oscillations(): slow oscillation detection (0.3-1.5 Hz in N3)
  - advanced_sleep_staging(): full automated sleep staging using YASA's built-in model

Designed as an enhanced pipeline that sits alongside the existing sleep_staging.py.
When YASA is available, it provides higher-quality spindle/SO detection; when not,
the existing eeg_processor.py spindle detector is used as a fallback.

YASA reference: Vallat & Walker (2021), JOSS, doi:10.21105/joss.03284
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

# Lazy-import YASA — it's optional and heavy
_YASA_AVAILABLE: Optional[bool] = None


def _check_yasa() -> bool:
    """Check if YASA is importable. Result is cached."""
    global _YASA_AVAILABLE
    if _YASA_AVAILABLE is None:
        try:
            import yasa  # noqa: F401
            _YASA_AVAILABLE = True
        except ImportError:
            _YASA_AVAILABLE = False
    return _YASA_AVAILABLE


def detect_spindles(
    eeg: np.ndarray,
    fs: float = 256.0,
    thresh: Dict[str, float] | None = None,
) -> Dict:
    """Detect sleep spindles using YASA (or fallback to basic detection).

    Args:
        eeg: 1D EEG signal array
        fs: Sampling frequency in Hz
        thresh: Optional thresholds for YASA spindle detection
                (keys: 'rel_pow', 'corr', 'rms')

    Returns:
        Dict with:
          - spindles_detected: bool
          - count: int (number of spindles found)
          - density: float (spindles per 30-second epoch)
          - mean_duration_s: float
          - mean_frequency_hz: float
          - events: list of dicts with start/end/duration/frequency per spindle
          - method: "yasa" or "fallback"
    """
    if not isinstance(eeg, np.ndarray) or eeg.ndim != 1 or len(eeg) < int(fs * 2):
        return {
            "spindles_detected": False,
            "count": 0,
            "density": 0.0,
            "mean_duration_s": 0.0,
            "mean_frequency_hz": 0.0,
            "events": [],
            "method": "none",
        }

    if _check_yasa():
        return _detect_spindles_yasa(eeg, fs, thresh)
    return _detect_spindles_fallback(eeg, fs)


def _detect_spindles_yasa(
    eeg: np.ndarray, fs: float, thresh: Dict[str, float] | None
) -> Dict:
    """Use YASA's spindle detector."""
    import yasa

    kwargs: Dict = {"data": eeg, "sf": fs}
    if thresh:
        kwargs["thresh"] = thresh

    sp = yasa.spindles_detect(**kwargs)

    if sp is None:
        return {
            "spindles_detected": False,
            "count": 0,
            "density": 0.0,
            "mean_duration_s": 0.0,
            "mean_frequency_hz": 0.0,
            "events": [],
            "method": "yasa",
        }

    summary = sp.summary()
    duration_s = len(eeg) / fs
    epochs_30s = max(duration_s / 30.0, 1.0)
    count = len(summary)

    events = []
    for _, row in summary.iterrows():
        events.append({
            "start_s": float(row.get("Start", 0)),
            "end_s": float(row.get("End", 0)),
            "duration_s": float(row.get("Duration", 0)),
            "frequency_hz": float(row.get("Frequency", 0)),
            "rms_uv": float(row.get("RMS", 0)),
        })

    mean_dur = float(summary["Duration"].mean()) if count > 0 else 0.0
    mean_freq = float(summary["Frequency"].mean()) if count > 0 else 0.0

    return {
        "spindles_detected": count > 0,
        "count": count,
        "density": round(count / epochs_30s, 2),
        "mean_duration_s": round(mean_dur, 3),
        "mean_frequency_hz": round(mean_freq, 2),
        "events": events,
        "method": "yasa",
    }


def _detect_spindles_fallback(eeg: np.ndarray, fs: float) -> Dict:
    """Basic spindle detection using bandpass + amplitude threshold."""
    from processing.eeg_processor import detect_sleep_spindles

    result = detect_sleep_spindles(eeg, fs)
    # detect_sleep_spindles returns a list of dicts (one per spindle)
    if isinstance(result, list):
        count = len(result)
        mean_dur = 0.0
        if count > 0:
            durations = [s.get("end", 0) - s.get("start", 0) for s in result]
            mean_dur = float(np.mean(durations)) if durations else 0.0
    else:
        # Handle unexpected dict return
        count = result.get("count", 0)
        mean_dur = result.get("mean_duration", 0.0)

    duration_s = len(eeg) / fs
    epochs_30s = max(duration_s / 30.0, 1.0)

    return {
        "spindles_detected": count > 0,
        "count": count,
        "density": round(count / epochs_30s, 2),
        "mean_duration_s": round(mean_dur, 3),
        "mean_frequency_hz": 13.0,  # approximate center of sigma band
        "events": [],
        "method": "fallback",
    }


def detect_slow_oscillations(
    eeg: np.ndarray,
    fs: float = 256.0,
) -> Dict:
    """Detect slow oscillations (0.3-1.5 Hz) using YASA or fallback.

    Args:
        eeg: 1D EEG signal array
        fs: Sampling frequency in Hz

    Returns:
        Dict with:
          - so_detected: bool
          - count: int
          - density: float (per 30s epoch)
          - mean_duration_s: float
          - mean_ptp_uv: float (peak-to-peak amplitude)
          - events: list of dicts
          - method: "yasa" or "fallback"
    """
    if not isinstance(eeg, np.ndarray) or eeg.ndim != 1 or len(eeg) < int(fs * 2):
        return {
            "so_detected": False,
            "count": 0,
            "density": 0.0,
            "mean_duration_s": 0.0,
            "mean_ptp_uv": 0.0,
            "events": [],
            "method": "none",
        }

    if _check_yasa():
        return _detect_so_yasa(eeg, fs)
    return _detect_so_fallback(eeg, fs)


def _detect_so_yasa(eeg: np.ndarray, fs: float) -> Dict:
    """Use YASA's slow oscillation detector."""
    import yasa

    sw = yasa.sw_detect(eeg, sf=fs)

    if sw is None:
        return {
            "so_detected": False,
            "count": 0,
            "density": 0.0,
            "mean_duration_s": 0.0,
            "mean_ptp_uv": 0.0,
            "events": [],
            "method": "yasa",
        }

    summary = sw.summary()
    duration_s = len(eeg) / fs
    epochs_30s = max(duration_s / 30.0, 1.0)
    count = len(summary)

    events = []
    for _, row in summary.iterrows():
        events.append({
            "start_s": float(row.get("Start", 0)),
            "end_s": float(row.get("End", 0)),
            "duration_s": float(row.get("Duration", 0)),
            "ptp_uv": float(row.get("PTP", 0)),
            "frequency_hz": float(row.get("Frequency", 0)),
        })

    mean_dur = float(summary["Duration"].mean()) if count > 0 else 0.0
    mean_ptp = float(summary["PTP"].mean()) if count > 0 else 0.0

    return {
        "so_detected": count > 0,
        "count": count,
        "density": round(count / epochs_30s, 2),
        "mean_duration_s": round(mean_dur, 3),
        "mean_ptp_uv": round(mean_ptp, 2),
        "events": events,
        "method": "yasa",
    }


def _detect_so_fallback(eeg: np.ndarray, fs: float) -> Dict:
    """Basic slow oscillation detection using delta band power threshold."""
    from processing.eeg_processor import extract_band_powers, preprocess

    processed = preprocess(eeg, fs)
    bands = extract_band_powers(processed, fs)
    delta = bands.get("delta", 0)

    # Rough heuristic: high delta = slow oscillations present
    so_detected = delta > 0.5
    return {
        "so_detected": so_detected,
        "count": 1 if so_detected else 0,
        "density": 1.0 if so_detected else 0.0,
        "mean_duration_s": 0.0,
        "mean_ptp_uv": 0.0,
        "events": [],
        "method": "fallback",
    }


def advanced_sleep_staging(
    eeg: np.ndarray,
    fs: float = 256.0,
    epoch_length_s: float = 30.0,
) -> Dict:
    """Full automated sleep staging using YASA's built-in classifier.

    Args:
        eeg: 1D EEG signal (should be at least several minutes)
        fs: Sampling frequency in Hz
        epoch_length_s: Epoch length for staging (default 30s per AASM standard)

    Returns:
        Dict with:
          - stages: list of stage labels per epoch
          - probabilities: list of dicts with per-stage probabilities
          - hypnogram: list of integer stage codes (W=0, N1=1, N2=2, N3=3, REM=4)
          - n_epochs: int
          - method: "yasa" or "fallback"
          - spindle_summary: spindle analysis for the full recording
          - so_summary: slow oscillation analysis for the full recording
    """
    if not isinstance(eeg, np.ndarray) or eeg.ndim != 1:
        return {
            "stages": [],
            "probabilities": [],
            "hypnogram": [],
            "n_epochs": 0,
            "method": "none",
            "spindle_summary": detect_spindles(np.zeros(1), fs),
            "so_summary": detect_slow_oscillations(np.zeros(1), fs),
        }

    min_samples = int(fs * epoch_length_s * 2)  # need at least 2 epochs
    if len(eeg) < min_samples:
        return {
            "stages": [],
            "probabilities": [],
            "hypnogram": [],
            "n_epochs": 0,
            "method": "none",
            "spindle_summary": detect_spindles(eeg, fs),
            "so_summary": detect_slow_oscillations(eeg, fs),
        }

    stage_map = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

    if _check_yasa():
        try:
            return _staging_yasa(eeg, fs, epoch_length_s, stage_map)
        except Exception:
            pass  # YASA API may have changed — fall back gracefully
    return _staging_fallback(eeg, fs, epoch_length_s, stage_map)


def _staging_yasa(
    eeg: np.ndarray,
    fs: float,
    epoch_length_s: float,
    stage_map: Dict[int, str],
) -> Dict:
    """Use YASA's SleepStaging classifier."""
    import yasa
    import mne

    # YASA >= 0.6 requires MNE Raw object, not raw numpy array
    info = mne.create_info(ch_names=["EEG"], sfreq=fs, ch_types=["eeg"])
    raw = mne.io.RawArray(eeg.reshape(1, -1) / 1e6, info, verbose=False)  # uV to V
    sls = yasa.SleepStaging(raw, eeg_name="EEG")
    hypno = sls.predict()
    proba = sls.predict_proba()

    # Convert YASA string labels to our format
    label_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
    hypno_int = [label_map.get(str(s), 0) for s in hypno]
    stages = [stage_map.get(s, "Wake") for s in hypno_int]

    probabilities = []
    for _, row in proba.iterrows():
        prob_dict = {}
        for col in proba.columns:
            stage_idx = label_map.get(col, 0)
            prob_dict[stage_map[stage_idx]] = round(float(row[col]), 4)
        probabilities.append(prob_dict)

    return {
        "stages": stages,
        "probabilities": probabilities,
        "hypnogram": hypno_int,
        "n_epochs": len(stages),
        "method": "yasa",
        "spindle_summary": detect_spindles(eeg, fs),
        "so_summary": detect_slow_oscillations(eeg, fs),
    }


def _staging_fallback(
    eeg: np.ndarray,
    fs: float,
    epoch_length_s: float,
    stage_map: Dict[int, str],
) -> Dict:
    """Fallback to existing SleepStagingModel epoch-by-epoch."""
    from models.sleep_staging import SleepStagingModel

    model = SleepStagingModel()
    epoch_samples = int(fs * epoch_length_s)
    n_epochs = len(eeg) // epoch_samples

    stages = []
    probabilities = []
    hypnogram = []

    for i in range(n_epochs):
        start = i * epoch_samples
        end = start + epoch_samples
        epoch = eeg[start:end]
        result = model.predict(epoch, fs)
        stages.append(result["stage"])
        probabilities.append(result.get("probabilities", {}))
        hypnogram.append(result.get("stage_index", 0))

    return {
        "stages": stages,
        "probabilities": probabilities,
        "hypnogram": hypnogram,
        "n_epochs": n_epochs,
        "method": "fallback",
        "spindle_summary": detect_spindles(eeg, fs),
        "so_summary": detect_slow_oscillations(eeg, fs),
    }
