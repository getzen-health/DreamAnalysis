"""Interictal epileptiform discharge (IED) spike detector for epilepsy screening.

IED signatures: high-amplitude sharp waves (>100 µV), fast rise time (<70ms),
followed by slow wave. Detectable in frontal channels (AF7/AF8 on Muse 2).

Note: Clinical diagnosis requires full 10-20 EEG system. This is a 4-channel
screening tool only.

References:
    Scheuer et al. (2017) — automated IED detection (94% sensitivity)
    Prasanth et al. (2021) — lightweight CNN for spike detection
    Sharma et al. (2024) — consumer EEG for epilepsy screening
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List


class IEDDetector:
    AMPLITUDE_THRESHOLD = 100.0  # µV
    SPIKE_DURATION_MS_MAX = 200  # ms

    def predict(self, signals: np.ndarray, fs: float = 256.0) -> Dict:
        if signals.ndim == 1:
            signals = signals[np.newaxis, :]
        n_ch, n_samples = signals.shape

        spikes_detected: List[Dict] = []
        max_amplitude = 0.0

        for ch_idx in range(n_ch):
            ch = signals[ch_idx]
            # z-score normalize
            ch_z = (ch - np.mean(ch)) / (np.std(ch) + 1e-9)
            amp = np.abs(ch)
            max_amplitude = max(max_amplitude, float(np.max(amp)))

            # Detect candidates: amplitude > threshold
            candidates = np.where(amp > self.AMPLITUDE_THRESHOLD)[0]
            if len(candidates) > 0:
                # Group into events
                groups = []
                g = [candidates[0]]
                for i in candidates[1:]:
                    if i - g[-1] < int(fs * 0.2):  # within 200ms
                        g.append(i)
                    else:
                        groups.append(g)
                        g = [i]
                groups.append(g)

                for group in groups:
                    peak_idx = group[np.argmax(amp[group])]
                    spikes_detected.append({
                        "channel": int(ch_idx),
                        "peak_sample": int(peak_idx),
                        "peak_amplitude_uv": round(float(amp[peak_idx]), 2),
                        "time_sec": round(float(peak_idx) / fs, 3),
                    })

        # Spike rate
        duration_sec = n_samples / fs
        spike_rate = len(spikes_detected) / (duration_sec + 1e-9)

        # Risk assessment
        if len(spikes_detected) >= 3 or spike_rate > 1.0:
            risk = "high_risk"
        elif len(spikes_detected) >= 1:
            risk = "abnormal_activity"
        else:
            risk = "no_spikes_detected"

        # Kurtosis of signal (high = impulsive spikes)
        from scipy.stats import kurtosis
        k = float(kurtosis(signals[0]))

        return {
            "risk_category": risk,
            "spike_count": len(spikes_detected),
            "spike_rate_per_min": round(spike_rate * 60, 2),
            "max_amplitude_uv": round(max_amplitude, 2),
            "signal_kurtosis": round(k, 4),
            "spikes": spikes_detected[:10],  # cap at 10 for response size
            "note": "Screening only — clinical diagnosis requires full 10-20 EEG",
            "model_used": "threshold_spike_detector",
        }


_model = IEDDetector()
def get_model(): return _model
