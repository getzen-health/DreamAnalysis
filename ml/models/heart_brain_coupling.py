"""Heart-brain coupling model for Muse 2 EEG + PPG integration.

Computes heartbeat-evoked potentials (HEP), heart rate variability (HRV),
and their relationship to EEG band powers. Detects interoceptive awareness
states and tracks autonomic-cortical coherence over time.

The Muse 2 provides 4 EEG channels (TP9, AF7, AF8, TP10) at 256 Hz plus
PPG via BrainFlow ANCILLARY preset (enabled by board.config_board("p50")).

Feature-based approach -- no saved model needed.

References:
    Schandry (1981) -- Heart beat perception and emotional experience
    Park et al. (2014) -- Interoceptive awareness declines with age:
        HEP amplitude at frontal sites indexes body awareness
    Thayer & Lane (2009) -- Neurovisceral integration model:
        vagal tone (HF-HRV) reflects prefrontal cortex regulation
    Sel et al. (2017) -- HEP amplitude modulates emotional face processing
    Shaffer & Ginsberg (2017) -- HRV metrics overview (RMSSD, SDNN, LF/HF)
"""
from typing import Dict, List, Optional

import numpy as np
_trapezoid = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
from scipy import signal as scipy_signal

from processing.ppg_features import PPGFeatureExtractor
from processing.heart_brain import compute_hep


# ── Constants ────────────────────────────────────────────────────────────────

# Autonomic state thresholds (LF/HF ratio)
_SYMPATHETIC_THRESHOLD = 2.0   # LF/HF > 2 = sympathetic dominance
_PARASYMPATHETIC_THRESHOLD = 0.5  # LF/HF < 0.5 = parasympathetic dominance

# Interoceptive awareness thresholds
_HIGH_INTEROCEPTION_HEP = 3.0   # uV -- HEP amplitude above this = high awareness
_LOW_INTEROCEPTION_HEP = 1.0    # uV -- HEP amplitude below this = low awareness

# Band power frequency ranges (Hz)
_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
}

# Maximum history entries per user
_MAX_HISTORY = 1000


class HeartBrainCoupling:
    """Heart-brain coupling analysis combining EEG and PPG from Muse 2.

    Computes:
    - Heartbeat-evoked potentials (HEP): EEG locked to heartbeats
    - HRV metrics: RMSSD, SDNN, LF/HF ratio
    - EEG-HRV correlation: band power vs autonomic activity
    - Interoceptive awareness score (0-100)
    - Autonomic state classification
    - Autonomic-cortical coherence index
    """

    def __init__(self, fs_ppg: float = 64.0, max_history: int = _MAX_HISTORY):
        self._fs_ppg = fs_ppg
        self._max_history = max_history
        self._baselines: Dict[str, Dict] = {}
        self._history: Dict[str, List[Dict]] = {}

    def set_baseline(
        self,
        eeg_signals: np.ndarray,
        ppg_signal: np.ndarray,
        fs: float = 256.0,
        user_id: str = "default",
    ) -> Dict:
        """Record resting-state baseline for heart-brain coupling.

        Should be called during 2-3 min eyes-closed resting state.
        Establishes per-user baselines for HRV and EEG band powers
        used to normalize subsequent analyze() calls.

        Args:
            eeg_signals: (n_channels, n_samples) EEG array in uV.
            ppg_signal: (n_samples,) PPG signal.
            fs: EEG sampling rate in Hz.
            user_id: User identifier.

        Returns:
            Dict with baseline HRV, EEG band powers, and baseline_set flag.
        """
        eeg = np.asarray(eeg_signals, dtype=np.float64)
        ppg = np.asarray(ppg_signal, dtype=np.float64)
        if eeg.ndim == 1:
            eeg = eeg[np.newaxis, :]

        # Baseline HRV
        extractor = PPGFeatureExtractor(fs=self._fs_ppg)
        hrv = extractor.extract_hrv(ppg)

        # Baseline EEG band powers (average across channels)
        band_powers = self._compute_band_powers(eeg, fs)

        baseline = {
            "hrv": hrv,
            "band_powers": band_powers,
            "fs": fs,
        }
        self._baselines[user_id] = baseline

        return {
            "baseline_set": True,
            "baseline_hr_bpm": hrv["hr_bpm"],
            "baseline_rmssd": hrv["rmssd_ms"],
            "baseline_sdnn": hrv["sdnn_ms"],
            "baseline_lf_hf": hrv["lf_hf_ratio"],
            "baseline_alpha_power": band_powers.get("alpha", 0.0),
            "baseline_theta_power": band_powers.get("theta", 0.0),
        }

    def analyze(
        self,
        eeg_signals: np.ndarray,
        ppg_signal: np.ndarray,
        fs: float = 256.0,
        user_id: str = "default",
    ) -> Dict:
        """Full heart-brain coupling analysis.

        Combines HEP, HRV, EEG band powers, and their interactions
        into a comprehensive coupling assessment.

        Args:
            eeg_signals: (n_channels, n_samples) EEG array in uV.
            ppg_signal: (n_samples,) PPG signal.
            fs: EEG sampling rate in Hz.
            user_id: User identifier.

        Returns:
            Dict with:
            - hep_amplitude: mean HEP amplitude in uV
            - hrv_metrics: {rmssd, sdnn, lf_hf_ratio, hr_bpm}
            - coupling_strength: 0-1 overall coupling score
            - interoceptive_score: 0-100 body awareness score
            - autonomic_state: sympathetic / parasympathetic / balanced
            - coherence_index: 0-1 autonomic-cortical coherence
        """
        eeg = np.asarray(eeg_signals, dtype=np.float64)
        ppg = np.asarray(ppg_signal, dtype=np.float64)
        if eeg.ndim == 1:
            eeg = eeg[np.newaxis, :]

        # ── HRV ──────────────────────────────────────────────────────
        extractor = PPGFeatureExtractor(fs=self._fs_ppg)
        hrv = extractor.extract_hrv(ppg)
        r_peak_times = extractor.get_r_peak_times(ppg)

        hrv_metrics = {
            "rmssd": hrv["rmssd_ms"],
            "sdnn": hrv["sdnn_ms"],
            "lf_hf_ratio": hrv["lf_hf_ratio"],
            "hr_bpm": hrv["hr_bpm"],
        }

        # ── HEP ──────────────────────────────────────────────────────
        hep = compute_hep(eeg, r_peak_times, fs_eeg=fs)
        hep_amplitude = hep["hep_amplitude"]

        # ── EEG band powers ──────────────────────────────────────────
        band_powers = self._compute_band_powers(eeg, fs)

        # ── Coupling strength ────────────────────────────────────────
        coupling_strength = self._compute_coupling_strength(
            hep, hrv, band_powers
        )

        # ── Interoceptive score ──────────────────────────────────────
        interoceptive_score = self._compute_interoceptive_score(
            hep_amplitude, hrv, user_id
        )

        # ── Autonomic state ──────────────────────────────────────────
        autonomic_state = self._classify_autonomic_state(hrv)

        # ── Coherence index ──────────────────────────────────────────
        coherence_index = self._compute_coherence_index(
            hrv, band_powers, user_id
        )

        result = {
            "hep_amplitude": round(hep_amplitude, 4),
            "hrv_metrics": {k: round(v, 4) for k, v in hrv_metrics.items()},
            "coupling_strength": round(coupling_strength, 4),
            "interoceptive_score": round(interoceptive_score, 2),
            "autonomic_state": autonomic_state,
            "coherence_index": round(coherence_index, 4),
        }

        # Record history
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > self._max_history:
            self._history[user_id] = self._history[user_id][-self._max_history:]

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get session-level statistics for heart-brain coupling.

        Returns:
            Dict with n_analyses, mean coupling strength, mean interoceptive
            score, autonomic state distribution, and has_baseline flag.
        """
        history = self._history.get(user_id, [])
        if not history:
            return {
                "n_analyses": 0,
                "has_baseline": user_id in self._baselines,
            }

        couplings = [h["coupling_strength"] for h in history]
        intero_scores = [h["interoceptive_score"] for h in history]
        states = [h["autonomic_state"] for h in history]

        from collections import Counter
        state_counts = Counter(states)

        return {
            "n_analyses": len(history),
            "mean_coupling_strength": round(float(np.mean(couplings)), 4),
            "max_coupling_strength": round(float(np.max(couplings)), 4),
            "mean_interoceptive_score": round(float(np.mean(intero_scores)), 2),
            "autonomic_state_distribution": dict(state_counts),
            "has_baseline": user_id in self._baselines,
        }

    def get_history(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> List[Dict]:
        """Get analysis history for a user.

        Args:
            user_id: User identifier.
            last_n: If provided, return only the last N entries.

        Returns:
            List of analysis result dicts.
        """
        history = self._history.get(user_id, [])
        if last_n is not None:
            history = history[-last_n:]
        return history

    def reset(self, user_id: str = "default"):
        """Clear all state for a user (baseline, history)."""
        self._baselines.pop(user_id, None)
        self._history.pop(user_id, None)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _compute_band_powers(
        self, eeg: np.ndarray, fs: float
    ) -> Dict[str, float]:
        """Compute mean band power across channels via Welch PSD.

        Returns dict with delta, theta, alpha, beta powers.
        """
        n_channels = eeg.shape[0]
        band_totals = {band: 0.0 for band in _BANDS}

        for ch in range(n_channels):
            sig = eeg[ch]
            nperseg = min(len(sig), int(fs * 2))
            if nperseg < 4:
                continue

            try:
                freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=nperseg)
            except Exception:
                continue

            for band, (lo, hi) in _BANDS.items():
                mask = (freqs >= lo) & (freqs <= hi)
                if not np.any(mask):
                    continue
                power = float(
                    _trapezoid(psd[mask], freqs[mask])
                    if hasattr(np, "trapezoid")
                    else np.trapz(psd[mask], freqs[mask])
                )
                band_totals[band] += power

        # Average across channels
        if n_channels > 0:
            band_totals = {
                k: round(v / n_channels, 6) for k, v in band_totals.items()
            }

        return band_totals

    def _compute_coupling_strength(
        self, hep: Dict, hrv: Dict, band_powers: Dict[str, float]
    ) -> float:
        """Compute overall heart-brain coupling strength (0-1).

        Blends:
        - HEP quality (more averaged beats = more reliable coupling)
        - HRV-alpha correlation proxy (high HF-HRV + high alpha = coherent)
        - Heart rate regularity (low RMSSD variance = stable coupling)

        Returns float in [0, 1].
        """
        # HEP component: coupling_quality from compute_hep (0-1)
        hep_score = hep.get("coupling_quality", 0.0)

        # HRV-alpha coherence: high vagal tone (HF) + high alpha = coherent state
        alpha = band_powers.get("alpha", 0.0)
        hf = hrv.get("hf_power", 0.0)
        # Normalize both to soft 0-1 range via tanh
        alpha_norm = float(np.tanh(alpha * 5.0))
        hf_norm = float(np.tanh(hf * 0.001))
        hrv_alpha_score = (alpha_norm + hf_norm) / 2.0

        # Regularity component: RMSSD indicates beat-to-beat variability
        # Moderate RMSSD (20-50 ms) = healthy coupling; very low or very high = less coupled
        rmssd = hrv.get("rmssd_ms", 0.0)
        if rmssd < 1e-6:
            regularity_score = 0.0
        else:
            # Bell curve centered at 35 ms, sigma ~20
            regularity_score = float(np.exp(-((rmssd - 35.0) ** 2) / (2 * 20.0 ** 2)))

        # Weighted blend
        coupling = (
            0.40 * hep_score
            + 0.35 * hrv_alpha_score
            + 0.25 * regularity_score
        )
        return float(np.clip(coupling, 0.0, 1.0))

    def _compute_interoceptive_score(
        self, hep_amplitude: float, hrv: Dict, user_id: str
    ) -> float:
        """Compute interoceptive awareness score (0-100).

        Based on Schandry (1981) heartbeat detection paradigm.
        Higher HEP amplitude at frontal sites = better body awareness.
        Vagal tone (HF-HRV) modulates interoceptive sensitivity.

        Uses baseline normalization when available.
        """
        baseline = self._baselines.get(user_id)

        # HEP component (0-1): maps HEP amplitude to awareness
        if baseline and baseline["hrv"].get("hr_bpm", 0) > 0:
            # Baseline-normalized: ratio of current to resting HEP
            baseline_hep = max(
                baseline.get("band_powers", {}).get("alpha", 0.01), 0.01
            )
            hep_ratio = hep_amplitude / (baseline_hep + 1e-6)
            hep_score = float(np.clip(np.tanh(hep_ratio * 0.5), 0, 1))
        else:
            # Absolute thresholds
            if hep_amplitude >= _HIGH_INTEROCEPTION_HEP:
                hep_score = 1.0
            elif hep_amplitude <= _LOW_INTEROCEPTION_HEP:
                hep_score = float(np.clip(hep_amplitude / _LOW_INTEROCEPTION_HEP, 0, 1))
            else:
                hep_score = float(
                    (hep_amplitude - _LOW_INTEROCEPTION_HEP)
                    / (_HIGH_INTEROCEPTION_HEP - _LOW_INTEROCEPTION_HEP)
                )

        # Vagal tone component (0-1): HF-HRV reflects parasympathetic activity
        hf = hrv.get("hf_power", 0.0)
        vagal_score = float(np.clip(np.tanh(hf * 0.001), 0, 1))

        # Blend: 60% HEP, 40% vagal tone (Park et al., 2014)
        raw_score = 0.60 * hep_score + 0.40 * vagal_score

        return float(np.clip(raw_score * 100.0, 0.0, 100.0))

    def _classify_autonomic_state(self, hrv: Dict) -> str:
        """Classify autonomic nervous system state from HRV.

        Uses LF/HF ratio as the primary indicator of sympathovagal balance
        (Shaffer & Ginsberg, 2017).

        Returns one of: 'sympathetic', 'parasympathetic', 'balanced'.
        """
        lf_hf = hrv.get("lf_hf_ratio", 1.0)

        if lf_hf > _SYMPATHETIC_THRESHOLD:
            return "sympathetic"
        elif lf_hf < _PARASYMPATHETIC_THRESHOLD:
            return "parasympathetic"
        else:
            return "balanced"

    def _compute_coherence_index(
        self, hrv: Dict, band_powers: Dict[str, float], user_id: str
    ) -> float:
        """Compute autonomic-cortical coherence index (0-1).

        Measures alignment between cardiac autonomic activity and
        cortical oscillations. High coherence = synchronized
        heart-brain communication.

        Based on Thayer & Lane (2009) neurovisceral integration model:
        prefrontal alpha + vagal tone = regulated, coherent state.
        """
        # Alpha power contribution (cortical calm/regulation)
        alpha = band_powers.get("alpha", 0.0)
        theta = band_powers.get("theta", 0.0)
        beta = band_powers.get("beta", 0.0)

        # Alpha dominance over beta = regulated cortical state
        total_power = alpha + beta + theta + 1e-10
        alpha_frac = alpha / total_power

        # Vagal tone (HF-HRV) = parasympathetic regulation
        hf = hrv.get("hf_power", 0.0)
        lf = hrv.get("lf_power", 0.0)
        total_hf_lf = hf + lf + 1e-10
        hf_frac = hf / total_hf_lf

        # Heart rate regularity: moderate HR + moderate variability = coherent
        hr = hrv.get("hr_bpm", 0.0)
        rmssd = hrv.get("rmssd_ms", 0.0)
        if hr < 1.0:
            hr_score = 0.0
        else:
            # Penalize extreme HR (too slow or too fast)
            hr_score = float(np.exp(-((hr - 65.0) ** 2) / (2 * 15.0 ** 2)))

        # Coherence: aligned cortical + autonomic regulation
        coherence = (
            0.35 * float(np.clip(alpha_frac * 2.5, 0, 1))
            + 0.35 * float(np.clip(hf_frac * 2.0, 0, 1))
            + 0.30 * hr_score
        )
        return float(np.clip(coherence, 0.0, 1.0))
