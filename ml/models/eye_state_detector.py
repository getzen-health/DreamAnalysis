"""Eye state detector — classify eyes open vs closed from EEG.

Detects eye state using alpha power changes at frontal and temporal
sites. Eyes-closed produces strong alpha increase (Berger effect),
and eye blinks produce characteristic delta spikes at frontal sites.

Key markers:
- Alpha power increase: eyes closed → 2-5x alpha boost (Berger, 1929)
- Frontal delta spikes: eye blinks (100-400 µV, 200-400 ms)
- Alpha reactivity: the ratio of eyes-closed to eyes-open alpha

References:
    Berger (1929) — Original alpha rhythm discovery
    Rosinvil et al. (2020) — Alpha reactivity as cognitive marker
    Sabanci & Koklu (2015) — EEG eye state classification
"""
from typing import Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal


class EyeStateDetector:
    """Detect eye state (open/closed/blink) from EEG.

    Uses alpha power changes and frontal delta spikes to classify
    current eye state and measure alpha reactivity.
    """

    def __init__(self, fs: float = 256.0, blink_threshold: float = 75.0):
        """
        Args:
            fs: EEG sampling rate.
            blink_threshold: Amplitude threshold (µV) for blink detection.
        """
        self._fs = fs
        self._blink_threshold = blink_threshold
        self._baselines: Dict[str, Dict] = {}
        self._history: Dict[str, List[Dict]] = {}

    def calibrate(
        self,
        eyes_open_signals: np.ndarray,
        eyes_closed_signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Calibrate with eyes-open and eyes-closed segments.

        Args:
            eyes_open_signals: (n_channels, n_samples) EEG during eyes open.
            eyes_closed_signals: (n_channels, n_samples) EEG during eyes closed.
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with alpha reactivity and calibration status.
        """
        fs = fs or self._fs
        eo = np.asarray(eyes_open_signals, dtype=float)
        ec = np.asarray(eyes_closed_signals, dtype=float)
        if eo.ndim == 1:
            eo = eo.reshape(1, -1)
        if ec.ndim == 1:
            ec = ec.reshape(1, -1)

        eo_alpha = self._mean_alpha_power(eo, fs)
        ec_alpha = self._mean_alpha_power(ec, fs)

        reactivity = ec_alpha / (eo_alpha + 1e-10)
        threshold = (eo_alpha + ec_alpha) / 2

        self._baselines[user_id] = {
            "eo_alpha": eo_alpha,
            "ec_alpha": ec_alpha,
            "reactivity": reactivity,
            "threshold": threshold,
        }

        return {
            "calibrated": True,
            "eyes_open_alpha": round(eo_alpha, 6),
            "eyes_closed_alpha": round(ec_alpha, 6),
            "alpha_reactivity": round(reactivity, 4),
        }

    def detect(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Detect current eye state.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG.
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with eye_state, alpha_power, blink_detected, blink_count,
            alpha_reactivity_live, confidence.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        alpha_power = self._mean_alpha_power(signals, fs)
        blinks = self._detect_blinks(signals, fs)
        blink_count = len(blinks)

        baseline = self._baselines.get(user_id, {})
        has_calibration = bool(baseline)

        if has_calibration:
            threshold = baseline["threshold"]
            eo_alpha = baseline["eo_alpha"]
            ec_alpha = baseline["ec_alpha"]
            reactivity_live = alpha_power / (eo_alpha + 1e-10)

            if blink_count > 0:
                eye_state = "blink"
                confidence = 0.9
            elif alpha_power >= threshold:
                eye_state = "closed"
                # Confidence based on distance from threshold
                conf_raw = (alpha_power - threshold) / (ec_alpha - threshold + 1e-10)
                confidence = float(np.clip(0.5 + conf_raw * 0.4, 0.5, 0.95))
            else:
                eye_state = "open"
                conf_raw = (threshold - alpha_power) / (threshold - eo_alpha + 1e-10)
                confidence = float(np.clip(0.5 + conf_raw * 0.4, 0.5, 0.95))
        else:
            # Without calibration, use population defaults
            reactivity_live = 1.0
            if blink_count > 0:
                eye_state = "blink"
                confidence = 0.8
            elif alpha_power > 0.5:  # arbitrary without calibration
                eye_state = "closed"
                confidence = 0.5
            else:
                eye_state = "open"
                confidence = 0.5

        result = {
            "eye_state": eye_state,
            "alpha_power": round(alpha_power, 6),
            "blink_detected": blink_count > 0,
            "blink_count": blink_count,
            "blink_positions": blinks,
            "alpha_reactivity_live": round(reactivity_live, 4),
            "confidence": round(confidence, 4),
            "has_calibration": has_calibration,
        }

        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > 1000:
            self._history[user_id] = self._history[user_id][-1000:]

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get session statistics."""
        history = self._history.get(user_id, [])
        if not history:
            return {"n_epochs": 0, "has_calibration": user_id in self._baselines}

        states = [h["eye_state"] for h in history]
        blinks_total = sum(h["blink_count"] for h in history)
        state_counts = {}
        for s in states:
            state_counts[s] = state_counts.get(s, 0) + 1

        return {
            "n_epochs": len(history),
            "has_calibration": user_id in self._baselines,
            "state_distribution": state_counts,
            "total_blinks": blinks_total,
            "blink_rate_per_epoch": round(blinks_total / len(history), 2),
            "mean_alpha": round(float(np.mean([h["alpha_power"] for h in history])), 6),
        }

    def get_history(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> List[Dict]:
        """Get detection history."""
        history = self._history.get(user_id, [])
        if last_n:
            history = history[-last_n:]
        return history

    def reset(self, user_id: str = "default"):
        """Clear all data."""
        self._baselines.pop(user_id, None)
        self._history.pop(user_id, None)

    # ── Private helpers ─────────────────────────────────────────

    def _mean_alpha_power(self, signals: np.ndarray, fs: float) -> float:
        """Compute mean alpha (8-12 Hz) power across channels."""
        powers = []
        for ch in range(signals.shape[0]):
            nperseg = min(len(signals[ch]), int(fs * 2))
            if nperseg < 4:
                continue
            try:
                freqs, psd = scipy_signal.welch(signals[ch], fs=fs, nperseg=nperseg)
                mask = (freqs >= 8) & (freqs <= 12)
                if np.any(mask):
                    if hasattr(np, "trapezoid"):
                        powers.append(float(np.trapezoid(psd[mask], freqs[mask])))
                    else:
                        powers.append(float(np.trapz(psd[mask], freqs[mask])))
            except Exception:
                pass
        return float(np.mean(powers)) if powers else 0.0

    def _detect_blinks(self, signals: np.ndarray, fs: float) -> List[int]:
        """Detect eye blinks from frontal channels (AF7=ch1, AF8=ch2).

        Returns list of sample indices where blinks were detected.
        """
        # Use frontal channels if available
        frontal = [1, 2] if signals.shape[0] >= 3 else [0]
        blink_positions = []

        for ch in frontal:
            if ch >= signals.shape[0]:
                continue
            sig = signals[ch]
            # Blinks are large amplitude deflections
            abs_sig = np.abs(sig - np.median(sig))
            peaks, _ = scipy_signal.find_peaks(
                abs_sig, height=self._blink_threshold,
                distance=int(fs * 0.3)  # min 300ms between blinks
            )
            blink_positions.extend(peaks.tolist())

        # Deduplicate nearby positions
        if blink_positions:
            blink_positions = sorted(set(blink_positions))
            deduped = [blink_positions[0]]
            for pos in blink_positions[1:]:
                if pos - deduped[-1] > int(fs * 0.2):
                    deduped.append(pos)
            return deduped
        return []
