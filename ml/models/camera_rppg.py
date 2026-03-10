"""Camera rPPG — contactless heart rate and HRV from face video.

Uses the CHROM algorithm (McDuff et al. 2014) to extract a pulse signal
from mean RGB values of a face ROI captured by a phone camera.

Reference:
    McDuff et al. (2014). Improvements in Remote Cardiopulmonary
    Measurement Using a Five Band Separation of Imaging
    Photoplethysmography. IEEE TNSRE.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

log = logging.getLogger(__name__)

# Minimum recording duration required for reliable HR/HRV estimation.
MIN_DURATION_SECONDS = 15.0


try:
    from scipy.signal import butter, filtfilt, find_peaks, welch

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False
    log.warning(
        "scipy is not installed. camera_rppg endpoints will return errors. "
        "Install with: pip install scipy"
    )


def _require_scipy() -> None:
    """Raise a clear error when scipy is not available."""
    if not _SCIPY_AVAILABLE:
        raise RuntimeError(
            "scipy is required for camera rPPG processing but is not installed. "
            "Run: pip install scipy"
        )


def _bandpass(signal: np.ndarray, fps: float, lo: float = 0.7, hi: float = 3.0) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter for cardiac frequencies (0.7–3 Hz).

    Args:
        signal: 1-D array of rPPG signal samples.
        fps:    Sampling rate (frames per second).
        lo:     Low cut-off frequency in Hz (default 0.7 ≈ 42 BPM).
        hi:     High cut-off frequency in Hz (default 3.0 ≈ 180 BPM).

    Returns:
        Bandpass-filtered signal, same length as input.
    """
    nyq = fps / 2.0
    lo_norm = lo / nyq
    hi_norm = hi / nyq
    # Clamp to valid range
    lo_norm = max(1e-4, min(lo_norm, 0.999))
    hi_norm = max(lo_norm + 1e-4, min(hi_norm, 0.999))
    b, a = butter(4, [lo_norm, hi_norm], btype="band")
    return filtfilt(b, a, signal)


def _chrom_signal(rgb_signal: np.ndarray) -> np.ndarray:
    """Apply CHROM colour-space projection to extract rPPG pulse channel.

    CHROM (McDuff 2014):
        X = 3R - 2G
        Y = 1.5R + G - 1.5B
        Xn = (X - mean(X)) / std(X)
        Yn = (Y - mean(Y)) / std(Y)
        S  = Xn - (std(Xn) / std(Yn)) * Yn

    Args:
        rgb_signal: (N, 3) float array — mean [R, G, B] per frame from face ROI.

    Returns:
        1-D rPPG signal of length N.
    """
    R = rgb_signal[:, 0].astype(np.float64)
    G = rgb_signal[:, 1].astype(np.float64)
    B = rgb_signal[:, 2].astype(np.float64)

    X = 3.0 * R - 2.0 * G
    Y = 1.5 * R + G - 1.5 * B

    Xn = (X - X.mean()) / (X.std() + 1e-8)
    Yn = (Y - Y.mean()) / (Y.std() + 1e-8)

    alpha = Xn.std() / (Yn.std() + 1e-8)
    S = Xn - alpha * Yn
    return S


def _ibi_to_hrv(ibi_ms: np.ndarray) -> Dict[str, float]:
    """Compute time-domain and frequency-domain HRV metrics from IBI array.

    Args:
        ibi_ms: 1-D array of successive inter-beat intervals in milliseconds.

    Returns:
        Dictionary with:
            hr_bpm       – mean heart rate (BPM)
            rmssd_ms     – root mean square of successive differences (ms)
            sdnn_ms      – standard deviation of NN intervals (ms)
            pnn50        – fraction of successive diffs > 50 ms (0–1)
            lf_hf_ratio  – LF/HF power ratio (autonomic balance)
            stress_index – 0 (no stress) to 1 (high stress), from HRV
    """
    if len(ibi_ms) < 2:
        return {
            "hr_bpm": float("nan"),
            "rmssd_ms": float("nan"),
            "sdnn_ms": float("nan"),
            "pnn50": float("nan"),
            "lf_hf_ratio": float("nan"),
            "stress_index": float("nan"),
        }

    mean_ibi = float(np.mean(ibi_ms))
    hr_bpm = 60000.0 / mean_ibi if mean_ibi > 0 else float("nan")
    diff_ibi = np.diff(ibi_ms)
    rmssd = float(np.sqrt(np.mean(diff_ibi**2)))
    sdnn = float(np.std(ibi_ms, ddof=1))
    pnn50 = float(np.mean(np.abs(diff_ibi) > 50.0))

    # Frequency-domain LF/HF from Welch PSD on evenly-sampled IBI.
    # Re-sample IBI at 4 Hz for spectral analysis.
    lf_hf_ratio = float("nan")
    if len(ibi_ms) >= 8:
        try:
            fs_hrv = 4.0  # Hz
            # Compute cumulative time axis from IBI
            cumtime = np.cumsum(ibi_ms) / 1000.0  # seconds
            t_uniform = np.arange(cumtime[0], cumtime[-1], 1.0 / fs_hrv)
            ibi_interp = np.interp(t_uniform, cumtime, ibi_ms)
            freqs, psd = welch(ibi_interp, fs=fs_hrv, nperseg=min(64, len(ibi_interp)))
            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
            hf_mask = (freqs >= 0.15) & (freqs < 0.40)
            lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0
            hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 1e-9
            lf_hf_ratio = lf_power / (hf_power + 1e-9)
        except Exception as exc:
            log.debug("LF/HF computation failed: %s", exc)

    # Stress index: high LF/HF + low RMSSD → high stress.
    # Normalised to [0, 1] using population reference ranges.
    # RMSSD: 20 ms = high stress, 80 ms = low stress.
    # LF/HF: 0.5 = low stress, 4.0 = high stress.
    rmssd_stress = float(np.clip(1.0 - (rmssd - 20.0) / (80.0 - 20.0), 0.0, 1.0))
    if np.isfinite(lf_hf_ratio):
        lfhf_stress = float(np.clip((lf_hf_ratio - 0.5) / (4.0 - 0.5), 0.0, 1.0))
        stress_index = 0.6 * rmssd_stress + 0.4 * lfhf_stress
    else:
        stress_index = rmssd_stress

    return {
        "hr_bpm": round(hr_bpm, 1),
        "rmssd_ms": round(rmssd, 2),
        "sdnn_ms": round(sdnn, 2),
        "pnn50": round(pnn50, 4),
        "lf_hf_ratio": round(lf_hf_ratio, 3) if np.isfinite(lf_hf_ratio) else None,
        "stress_index": round(float(np.clip(stress_index, 0.0, 1.0)), 3),
    }


class CameraRPPG:
    """Contactless heart rate and HRV extractor using the CHROM algorithm.

    Input: per-frame mean RGB values from a face ROI.
    Output: HR (BPM), RMSSD (ms), pNN50 (%), LF/HF ratio, stress_index (0-1).

    Minimum requirement: 15 seconds of video (450 frames at 30 fps).
    """

    MIN_FRAMES: int = int(MIN_DURATION_SECONDS * 30)  # conservative default

    def process_frames(
        self,
        rgb_frames: List[List[float]],
        fps: float = 30.0,
    ) -> Dict:
        """Extract HR/HRV from a sequence of per-frame mean RGB values.

        Args:
            rgb_frames: List of [R, G, B] mean values per frame (0–255 or 0–1).
                        Must contain at least ``MIN_FRAMES`` entries.
            fps:        Camera capture rate in frames per second.

        Returns:
            Dictionary with hr_bpm, rmssd_ms, sdnn_ms, pnn50, lf_hf_ratio,
            stress_index, n_frames, duration_s, n_peaks, algorithm.
            On error: {"error": "<message>"}.
        """
        _require_scipy()

        min_frames = max(self.MIN_FRAMES, int(MIN_DURATION_SECONDS * fps))
        if len(rgb_frames) < min_frames:
            return {
                "error": (
                    f"Insufficient frames: got {len(rgb_frames)}, "
                    f"need at least {min_frames} ({MIN_DURATION_SECONDS:.0f}s at {fps} fps)"
                )
            }

        try:
            rgb = np.array(rgb_frames, dtype=np.float64)
            if rgb.ndim != 2 or rgb.shape[1] != 3:
                return {"error": "rgb_frames must be a list of [R, G, B] triplets"}

            S = _chrom_signal(rgb)
            return self._extract_metrics(S, fps, len(rgb_frames))
        except Exception as exc:
            log.exception("process_frames failed")
            return {"error": str(exc)}

    def process_raw_signal(
        self,
        signal: np.ndarray,
        fps: float = 30.0,
    ) -> Dict:
        """Extract HR/HRV from a pre-extracted rPPG or green-channel signal.

        Args:
            signal: 1-D numpy array of rPPG signal samples.
            fps:    Sampling rate in Hz (frames per second).

        Returns:
            Same dictionary as :meth:`process_frames`.
        """
        _require_scipy()

        min_frames = max(self.MIN_FRAMES, int(MIN_DURATION_SECONDS * fps))
        if len(signal) < min_frames:
            return {
                "error": (
                    f"Insufficient signal length: got {len(signal)}, "
                    f"need at least {min_frames} ({MIN_DURATION_SECONDS:.0f}s at {fps} fps)"
                )
            }

        try:
            arr = np.asarray(signal, dtype=np.float64).ravel()
            return self._extract_metrics(arr, fps, len(arr))
        except Exception as exc:
            log.exception("process_raw_signal failed")
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_metrics(
        self,
        raw_signal: np.ndarray,
        fps: float,
        n_frames: int,
    ) -> Dict:
        """Filter, detect peaks, compute HR/HRV from a rPPG signal.

        Args:
            raw_signal: 1-D signal (already CHROM-projected or raw green channel).
            fps:        Sampling rate (frames per second).
            n_frames:   Original frame count (for reporting).

        Returns:
            HR/HRV metrics dictionary.
        """
        # Bandpass filter to cardiac band (0.7–3 Hz)
        filtered = _bandpass(raw_signal, fps, lo=0.7, hi=3.0)

        # Detect peaks with minimum distance of 0.4 s (150 BPM ceiling)
        min_distance = max(1, int(fps * 0.4))
        peaks, _ = find_peaks(filtered, distance=min_distance)

        n_peaks = len(peaks)
        if n_peaks < 2:
            return {
                "error": (
                    "Too few cardiac peaks detected — check face ROI quality "
                    "or increase recording duration."
                ),
                "n_frames": n_frames,
                "duration_s": round(n_frames / fps, 2),
                "n_peaks": n_peaks,
                "algorithm": "CHROM",
            }

        # Inter-beat intervals in milliseconds
        ibi_ms = np.diff(peaks) / fps * 1000.0

        # Physiological plausibility filter: keep IBIs in 333–2000 ms (30–180 BPM)
        valid = (ibi_ms >= 333.0) & (ibi_ms <= 2000.0)
        ibi_ms = ibi_ms[valid]

        if len(ibi_ms) < 2:
            return {
                "error": "Detected peaks outside physiological HR range (30–180 BPM).",
                "n_frames": n_frames,
                "duration_s": round(n_frames / fps, 2),
                "n_peaks": n_peaks,
                "algorithm": "CHROM",
            }

        hrv = _ibi_to_hrv(ibi_ms)
        hrv.update(
            {
                "n_frames": n_frames,
                "duration_s": round(n_frames / fps, 2),
                "n_peaks": n_peaks,
                "algorithm": "CHROM",
            }
        )
        return hrv
