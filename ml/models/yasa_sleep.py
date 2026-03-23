"""YASA-based sleep staging, spindle detection, and slow wave detection for Muse 2 EEG.

Wraps YASA (Yet Another Spindle Algorithm) for single-channel frontal EEG
sleep analysis. Validated on Muse-S with Cohen's Kappa 0.76, accuracy 88-96%
across stages (Dec 2025 study). Works with AF7/AF8 frontal channels at 256 Hz.

YASA's SleepStaging uses a pre-trained LightGBM classifier on 30-second epochs.
Minimum data requirement: ~5 minutes (10 epochs) for meaningful staging.

Dependencies: yasa>=0.6.0, mne, numpy, scipy (all in requirements.txt).

GitHub issue: #527
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# Minimum duration (seconds) for sleep staging. YASA needs at least ~5 min
# of data to extract meaningful features and run its LightGBM classifier.
_MIN_STAGING_DURATION_SEC = 300  # 5 minutes


def _check_yasa_available() -> bool:
    """Check if YASA is importable."""
    try:
        import yasa  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------

def stage_with_yasa(
    eeg_data: np.ndarray,
    fs: int = 256,
    channel_name: str = "EEG",
) -> Dict:
    """Stage sleep using YASA from a raw 1D EEG array.

    Creates an MNE RawArray from the input data, runs YASA's SleepStaging
    classifier, and computes standard sleep statistics.

    Args:
        eeg_data: 1D array of EEG samples in microvolts.
        fs: Sample rate in Hz (256 for Muse 2).
        channel_name: Channel name for MNE info (e.g. "EEG", "AF7", "AF8").

    Returns:
        Dict with keys:
            stages: list of stage labels per 30s epoch (WAKE, N1, N2, N3, REM)
            probabilities: list of dicts with per-epoch stage probabilities
            summary: dict of AASM sleep statistics from YASA
            model_type: "yasa"
        Or on error:
            error: str describing the issue
            stages: []
            summary: {}
    """
    if not _check_yasa_available():
        return {"error": "YASA not installed. Install with: pip install yasa",
                "stages": [], "summary": {}}

    import mne
    import yasa

    # Validate minimum duration
    duration_sec = len(eeg_data) / fs
    if duration_sec < _MIN_STAGING_DURATION_SEC:
        return {
            "error": f"Data too short for sleep staging: {duration_sec:.0f}s "
                     f"(minimum {_MIN_STAGING_DURATION_SEC}s / "
                     f"{_MIN_STAGING_DURATION_SEC // 60} minutes required)",
            "stages": [],
            "summary": {},
        }

    try:
        # Create MNE RawArray — MNE expects Volts, Muse provides microvolts
        info = mne.create_info([channel_name], sfreq=fs, ch_types=["eeg"])
        raw = mne.io.RawArray(
            eeg_data.reshape(1, -1) / 1e6,  # uV -> V for MNE
            info,
            verbose=False,
        )

        # Run YASA sleep staging
        sls = yasa.SleepStaging(raw, eeg_name=channel_name)
        hypnogram = sls.predict()  # Returns Hypnogram object (YASA >= 0.7)

        # Extract stage labels as list of strings
        stages = hypnogram.hypno.tolist()

        # Extract per-epoch probabilities (use .proba on Hypnogram, not
        # the deprecated sls.predict_proba())
        proba_df = hypnogram.proba
        if proba_df is not None:
            probabilities = proba_df.to_dict("records")
        else:
            # Fallback for older YASA versions
            proba_df = sls.predict_proba()
            probabilities = proba_df.to_dict("records")

        # Compute sleep statistics using integer hypnogram
        hypno_int = hypnogram.as_int().values
        stats = yasa.sleep_statistics(hypno_int, sf_hyp=1 / 30)

        return {
            "stages": stages,
            "probabilities": probabilities,
            "summary": stats,
            "model_type": "yasa",
        }

    except Exception as exc:
        log.exception("YASA sleep staging failed")
        return {
            "error": f"YASA sleep staging failed: {exc}",
            "stages": [],
            "summary": {},
        }


def detect_spindles_yasa(
    eeg_data: np.ndarray,
    fs: int = 256,
) -> Dict:
    """Detect sleep spindles using YASA.

    Spindles are 12-15 Hz oscillatory bursts lasting 0.5-2 seconds,
    characteristic of N2/N3 sleep. Spindle density predicts overnight
    memory consolidation (Mander et al. 2014).

    Args:
        eeg_data: 1D array of EEG samples in microvolts.
        fs: Sample rate in Hz (256 for Muse 2).

    Returns:
        Dict with keys: count, density (per minute), avg_duration_ms,
        avg_frequency_hz, spindles (list of first 20 spindle details).
    """
    if not _check_yasa_available():
        return {"error": "YASA not installed", "count": 0, "density": 0.0,
                "avg_duration_ms": 0.0, "avg_frequency_hz": 0.0, "spindles": []}

    import yasa

    try:
        sp = yasa.spindles_detect(eeg_data, sf=fs, verbose=False)

        if sp is None:
            return {
                "count": 0,
                "density": 0.0,
                "avg_duration_ms": 0.0,
                "avg_frequency_hz": 0.0,
                "spindles": [],
            }

        summary = sp.summary()
        duration_min = len(eeg_data) / fs / 60.0
        count = len(summary)

        return {
            "count": count,
            "density": count / duration_min if duration_min > 0 else 0.0,
            "avg_duration_ms": float(summary["Duration"].mean() * 1000) if count > 0 else 0.0,
            "avg_frequency_hz": float(summary["Frequency"].mean()) if count > 0 else 0.0,
            "spindles": summary.to_dict("records")[:20],  # Cap at 20
        }

    except Exception as exc:
        log.exception("YASA spindle detection failed")
        return {
            "error": f"Spindle detection failed: {exc}",
            "count": 0,
            "density": 0.0,
            "avg_duration_ms": 0.0,
            "avg_frequency_hz": 0.0,
            "spindles": [],
        }


def detect_slow_waves_yasa(
    eeg_data: np.ndarray,
    fs: int = 256,
) -> Dict:
    """Detect slow oscillations using YASA.

    Slow waves (0.3-1.5 Hz) are hallmarks of deep NREM sleep (N3).
    SO-spindle coupling drives hippocampal-neocortical memory transfer
    (Staresina et al. 2015, Helfrich et al. 2018).

    Args:
        eeg_data: 1D array of EEG samples in microvolts.
        fs: Sample rate in Hz (256 for Muse 2).

    Returns:
        Dict with keys: count, density (per minute), avg_amplitude_uv.
    """
    if not _check_yasa_available():
        return {"error": "YASA not installed", "count": 0, "density": 0.0,
                "avg_amplitude_uv": 0.0}

    import yasa

    try:
        sw = yasa.sw_detect(eeg_data, sf=fs, verbose=False)

        if sw is None:
            return {
                "count": 0,
                "density": 0.0,
                "avg_amplitude_uv": 0.0,
            }

        summary = sw.summary()
        duration_min = len(eeg_data) / fs / 60.0
        count = len(summary)

        return {
            "count": count,
            "density": count / duration_min if duration_min > 0 else 0.0,
            "avg_amplitude_uv": float(summary["PTP"].mean()) if count > 0 else 0.0,
        }

    except Exception as exc:
        log.exception("YASA slow wave detection failed")
        return {
            "error": f"Slow wave detection failed: {exc}",
            "count": 0,
            "density": 0.0,
            "avg_amplitude_uv": 0.0,
        }


# ---------------------------------------------------------------------------
# Class wrapper
# ---------------------------------------------------------------------------

class YASASleepStager:
    """YASA-based sleep staging for Muse 2 data.

    Uses single frontal channel (AF7 or AF8) at 256 Hz.
    Provides sleep staging, spindle detection, slow wave detection,
    and a combined full_analysis method.
    """

    def __init__(self):
        self._available = _check_yasa_available()

    def stage_sleep(
        self,
        eeg_data: np.ndarray,
        fs: int = 256,
        channel: str = "AF7",
    ) -> Dict:
        """Stage sleep from Muse 2 EEG data.

        Args:
            eeg_data: 1D array of EEG samples in microvolts (>= 5 min).
            fs: Sample rate (256 for Muse 2).
            channel: Channel name ("AF7" or "AF8").

        Returns:
            Dict with stages, confidence, summary, spindles, model_type.
        """
        if not self._available:
            return {"error": "YASA not installed", "stages": [], "summary": {}}

        result = stage_with_yasa(eeg_data, fs=fs, channel_name=channel)

        # Add model_type marker for the class wrapper
        if "error" not in result:
            result["model_type"] = "yasa_single_channel"
        else:
            result["model_type"] = "yasa_single_channel"

        return result

    def detect_spindles(self, eeg_data: np.ndarray, fs: int = 256) -> Dict:
        """Detect spindles. Delegates to detect_spindles_yasa."""
        return detect_spindles_yasa(eeg_data, fs=fs)

    def detect_slow_waves(self, eeg_data: np.ndarray, fs: int = 256) -> Dict:
        """Detect slow waves. Delegates to detect_slow_waves_yasa."""
        return detect_slow_waves_yasa(eeg_data, fs=fs)

    def full_analysis(
        self,
        eeg_data: np.ndarray,
        fs: int = 256,
        channel: str = "AF7",
    ) -> Dict:
        """Run full sleep analysis: staging + spindle + slow wave detection.

        Args:
            eeg_data: 1D array of EEG samples in microvolts (>= 5 min).
            fs: Sample rate (256 for Muse 2).
            channel: Channel name ("AF7" or "AF8").

        Returns:
            Dict with keys: staging, spindles, slow_waves.
        """
        return {
            "staging": self.stage_sleep(eeg_data, fs=fs, channel=channel),
            "spindles": self.detect_spindles(eeg_data, fs=fs),
            "slow_waves": self.detect_slow_waves(eeg_data, fs=fs),
        }
