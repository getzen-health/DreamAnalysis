"""PPG feature extraction for HRV analysis.

Muse 2 PPG sensor: accessible via board.config_board("p50") in BrainFlow.
Sampling rate: 64 Hz (PPG), 256 Hz (EEG).

Features extracted:
Time-domain HRV: RMSSD, SDNN, pNN50, mean_IBI, HR_bpm
Frequency-domain HRV: LF_power, HF_power, LF_HF_ratio (via Lomb-Scargle)

Scientific basis:
- Communications Biology (2025): Delta-HRV coupling during sleep staging
- PubMed (2025): XGBoost with PPG+EEG achieves 97.58% 4-class emotion
"""

import logging
import numpy as np
from scipy import signal as scipy_signal
from typing import Dict, Optional

log = logging.getLogger(__name__)


class PPGFeatureExtractor:
    """Extract HRV features from a raw PPG signal (Muse 2 ANCILLARY sensor)."""

    def __init__(self, fs: float = 64.0):
        self.fs = fs

    def detect_peaks(self, ppg_signal: np.ndarray) -> np.ndarray:
        """Detect R-peaks (systolic peaks) in PPG signal.

        Uses scipy.signal.find_peaks with adaptive threshold.
        Returns array of peak indices.
        """
        if len(ppg_signal) < 3:
            return np.array([], dtype=np.int64)

        sig = ppg_signal.astype(np.float64)
        # Normalize to zero mean, unit variance for robust thresholding
        std = sig.std()
        if std < 1e-10:
            return np.array([], dtype=np.int64)
        sig_norm = (sig - sig.mean()) / std

        # Minimum peak distance: 0.4 s corresponds to max 150 BPM
        min_dist = max(1, int(self.fs * 0.4))
        # Adaptive height threshold: 0.3 standard deviations above mean (already at 0)
        height = 0.3

        peaks, _ = scipy_signal.find_peaks(
            sig_norm,
            distance=min_dist,
            height=height,
        )
        return peaks.astype(np.int64)

    def extract_hrv(self, ppg_signal: np.ndarray) -> Dict:
        """Extract HRV features from PPG signal.

        Returns dict with:
        - mean_ibi_ms: mean inter-beat interval in ms
        - hr_bpm: heart rate in BPM
        - sdnn_ms: std of NN intervals
        - rmssd_ms: root mean square of successive differences
        - pnn50: fraction of successive differences > 50 ms
        - lf_power: low frequency HRV power (0.04-0.15 Hz)
        - hf_power: high frequency HRV power (0.15-0.4 Hz)
        - lf_hf_ratio: sympathovagal balance
        - n_beats: number of beats detected
        """
        peaks = self.detect_peaks(ppg_signal)

        if len(peaks) < 3:
            return self._empty_hrv(n_beats=len(peaks))

        # IBI in milliseconds
        ibi_ms = np.diff(peaks) / self.fs * 1000.0

        # Remove physiological outliers (250-2000 ms = 30-240 BPM)
        ibi_ms = ibi_ms[(ibi_ms >= 250) & (ibi_ms <= 2000)]

        if len(ibi_ms) < 2:
            return self._empty_hrv(n_beats=len(peaks))

        mean_ibi = float(ibi_ms.mean())
        hr_bpm = 60000.0 / mean_ibi
        sdnn = float(ibi_ms.std(ddof=1)) if len(ibi_ms) > 1 else 0.0
        successive = np.diff(ibi_ms)
        rmssd = float(np.sqrt((successive ** 2).mean())) if len(successive) > 0 else 0.0
        pnn50 = (
            float((np.abs(successive) > 50).sum()) / len(successive)
            if len(successive) > 0
            else 0.0
        )

        lf_power, hf_power, lf_hf_ratio = self._frequency_domain_hrv(ibi_ms)

        return {
            "mean_ibi_ms": round(mean_ibi, 2),
            "hr_bpm": round(float(hr_bpm), 2),
            "sdnn_ms": round(sdnn, 2),
            "rmssd_ms": round(rmssd, 2),
            "pnn50": round(pnn50, 4),
            "lf_power": round(float(lf_power), 6),
            "hf_power": round(float(hf_power), 6),
            "lf_hf_ratio": round(float(lf_hf_ratio), 4),
            "n_beats": int(len(peaks)),
        }

    def get_r_peak_times(self, ppg_signal: np.ndarray) -> np.ndarray:
        """Return R-peak times in seconds for HEP computation."""
        peaks = self.detect_peaks(ppg_signal)
        return peaks / self.fs

    # ── Private helpers ──────────────────────────────────────────────────────

    def _frequency_domain_hrv(self, ibi_ms: np.ndarray):
        """Estimate LF/HF power via FFT on uniformly resampled RR series.

        Returns (lf_power, hf_power, lf_hf_ratio).
        """
        try:
            target_fs = 4.0  # Hz — standard for HRV spectral analysis
            # Cumulative time axis in seconds
            t = np.cumsum(ibi_ms) / 1000.0
            n_samples = max(64, int((t[-1] - t[0]) * target_fs))
            t_uniform = np.linspace(t[0], t[-1], n_samples)
            ibi_interp = np.interp(t_uniform, t, ibi_ms)
            ibi_interp -= ibi_interp.mean()  # detrend

            freqs = np.fft.rfftfreq(n_samples, d=1.0 / target_fs)
            psd = np.abs(np.fft.rfft(ibi_interp)) ** 2
            freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

            lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
            hf_mask = (freqs >= 0.15) & (freqs <= 0.40)

            lf_power = float(psd[lf_mask].sum() * freq_res)
            hf_power = float(psd[hf_mask].sum() * freq_res)
            lf_hf = lf_power / (hf_power + 1e-10)
            return lf_power, hf_power, min(lf_hf, 20.0)
        except Exception as exc:
            log.debug("HRV frequency analysis failed: %s", exc)
            return 0.0, 0.0, 1.0

    @staticmethod
    def _empty_hrv(n_beats: int = 0) -> Dict:
        return {
            "mean_ibi_ms": 0.0,
            "hr_bpm": 0.0,
            "sdnn_ms": 0.0,
            "rmssd_ms": 0.0,
            "pnn50": 0.0,
            "lf_power": 0.0,
            "hf_power": 0.0,
            "lf_hf_ratio": 0.0,
            "n_beats": int(n_beats),
        }
