"""Motor imagery BCI classifier using mu/beta desynchronization.

Classifies imagined movements (left hand, right hand, both feet, rest)
from EEG event-related desynchronization (ERD) patterns. On Muse 2,
temporal channels TP9 (ch0) and TP10 (ch3) serve as the closest proxy
for the C3/C4 sensorimotor electrodes used in traditional motor imagery BCIs.

Motor imagery neuroscience:
- Left hand imagery  -> right hemisphere mu (8-12 Hz) suppression at TP10
- Right hand imagery -> left hemisphere mu suppression at TP9
- Feet imagery       -> bilateral mu/beta suppression (midline activation)
- Rest               -> minimal ERD, mu power near baseline

The laterality index captures the asymmetry of mu suppression:
  laterality = (ERD_left - ERD_right) / (|ERD_left| + |ERD_right|)
  Positive = left hemisphere suppression = right hand imagery
  Negative = right hemisphere suppression = left hand imagery

References:
    Pfurtscheller & Neuper (2001) — Motor imagery and direct brain-computer
        communication. Proceedings of the IEEE, 89(7), 1123-1134.
    Blankertz et al. (2008) — Optimizing spatial filters for robust EEG
        single-trial analysis. IEEE Signal Processing Magazine, 25(1), 41-56.
"""
from typing import Dict, List, Optional

import numpy as np
_trapezoid = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
from scipy import signal as scipy_signal

CLASSES = ("left_hand", "right_hand", "both_feet", "rest")

# Muse 2 channel mapping (BrainFlow board_id 38)
CH_TP9 = 0   # Left temporal  — proxy for C3
CH_AF7 = 1   # Left frontal
CH_AF8 = 2   # Right frontal
CH_TP10 = 3  # Right temporal — proxy for C4


class MotorImageryClassifier:
    """ERD-based motor imagery classifier for 4-channel Muse 2 EEG.

    Uses mu (8-12 Hz) and beta (13-30 Hz) desynchronization at TP9/TP10
    to classify imagined hand and foot movements.
    """

    def __init__(self, fs: float = 256.0):
        self._fs = fs
        # Per-user state: baselines, history, labels
        self._baselines: Dict[str, Dict] = {}
        self._history: Dict[str, List[Dict]] = {}
        self._labels: Dict[str, List[Optional[str]]] = {}

    # ── Public API ──────────────────────────────────────────────────

    def set_baseline(
        self,
        eeg_signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record resting-state baseline mu and beta power per channel.

        Args:
            eeg_signals: (n_channels, n_samples) EEG array, or (n_samples,).
            fs: Sampling rate in Hz.
            user_id: User identifier for multi-user support.

        Returns:
            Dict with baseline_set, mu_powers, beta_powers per channel.
        """
        fs = fs or self._fs
        signals = self._ensure_2d(eeg_signals)

        mu_powers = []
        beta_powers = []
        for ch in range(signals.shape[0]):
            mu_powers.append(self._band_power(signals[ch], fs, 8.0, 12.0))
            beta_powers.append(self._band_power(signals[ch], fs, 13.0, 30.0))

        self._baselines[user_id] = {
            "mu": np.array(mu_powers),
            "beta": np.array(beta_powers),
        }

        return {
            "baseline_set": True,
            "mu_powers": [round(p, 6) for p in mu_powers],
            "beta_powers": [round(p, 6) for p in beta_powers],
        }

    def classify(
        self,
        eeg_signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Classify imagined movement from EEG mu/beta desynchronization.

        Args:
            eeg_signals: (n_channels, n_samples) or (n_samples,) EEG array.
            fs: Sampling rate in Hz.
            user_id: User identifier.

        Returns:
            Dict with predicted_class, probabilities, confidence,
            laterality_index, mu_suppression, erd_map.
        """
        fs = fs or self._fs
        signals = self._ensure_2d(eeg_signals)
        n_ch = signals.shape[0]

        # Compute current mu and beta power per channel
        mu_current = np.array([
            self._band_power(signals[ch], fs, 8.0, 12.0) for ch in range(n_ch)
        ])
        beta_current = np.array([
            self._band_power(signals[ch], fs, 13.0, 30.0) for ch in range(n_ch)
        ])

        # Compute ERD relative to baseline (negative = desynchronization)
        baseline = self._baselines.get(user_id)
        if baseline is not None:
            mu_base = baseline["mu"][:n_ch]
            beta_base = baseline["beta"][:n_ch]
            # ERD% = (current - baseline) / baseline
            # Negative ERD = suppression (desynchronization)
            mu_erd = np.where(
                mu_base > 1e-12,
                (mu_current - mu_base) / mu_base,
                np.zeros(n_ch),
            )
            beta_erd = np.where(
                beta_base > 1e-12,
                (beta_current - beta_base) / beta_base,
                np.zeros(n_ch),
            )
        else:
            # Without baseline, use raw power ratios (less accurate)
            total_mu = np.sum(mu_current) + 1e-12
            mu_erd = mu_current / total_mu - 1.0 / n_ch
            total_beta = np.sum(beta_current) + 1e-12
            beta_erd = beta_current / total_beta - 1.0 / n_ch

        # Mu suppression per channel (how much ERD, as positive number)
        mu_suppression = {}
        ch_names = ["TP9", "AF7", "AF8", "TP10"]
        for i in range(min(n_ch, 4)):
            mu_suppression[ch_names[i]] = round(float(-mu_erd[i]), 4)

        # ERD map
        erd_map = {
            "mu": {ch_names[i]: round(float(mu_erd[i]), 4) for i in range(min(n_ch, 4))},
            "beta": {ch_names[i]: round(float(beta_erd[i]), 4) for i in range(min(n_ch, 4))},
        }

        # Laterality index from temporal channels (TP9 vs TP10)
        laterality_index = self._compute_laterality(mu_erd, n_ch)

        # Classify based on ERD features
        probs, confidence = self._compute_probabilities(
            mu_erd, beta_erd, laterality_index, n_ch
        )

        predicted_class = CLASSES[int(np.argmax([probs[c] for c in CLASSES]))]

        result = {
            "predicted_class": predicted_class,
            "probabilities": {c: round(float(probs[c]), 6) for c in CLASSES},
            "confidence": round(float(confidence), 4),
            "laterality_index": round(float(np.clip(laterality_index, -1, 1)), 4),
            "mu_suppression": mu_suppression,
            "erd_map": erd_map,
        }

        # Store in history
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > 1000:
            self._history[user_id] = self._history[user_id][-1000:]

        return result

    def submit_label(self, true_label: str, user_id: str = "default") -> None:
        """Submit ground-truth label for the most recent classification.

        Used for online accuracy tracking. The label is paired with the
        last classification result.

        Args:
            true_label: One of left_hand, right_hand, both_feet, rest.
            user_id: User identifier.
        """
        history = self._history.get(user_id, [])
        if not history:
            return

        if user_id not in self._labels:
            self._labels[user_id] = []

        # Pair label with classification index
        self._labels[user_id].append({
            "true_label": true_label,
            "predicted": history[-1]["predicted_class"],
        })

    def get_accuracy(self, user_id: str = "default") -> float:
        """Get online classification accuracy from submitted labels.

        Returns:
            Accuracy as float 0-1, or 0.0 if no labels submitted.
        """
        labels = self._labels.get(user_id, [])
        if not labels:
            return 0.0
        correct = sum(1 for l in labels if l["true_label"] == l["predicted"])
        return correct / len(labels)

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get classification session statistics.

        Returns:
            Dict with n_classifications, class_distribution, mean_confidence,
            mean_laterality, accuracy (if labels submitted).
        """
        history = self._history.get(user_id, [])
        if not history:
            return {
                "n_classifications": 0,
                "has_baseline": user_id in self._baselines,
            }

        distribution = {c: 0 for c in CLASSES}
        for h in history:
            distribution[h["predicted_class"]] += 1

        confidences = [h["confidence"] for h in history]
        lateralities = [h["laterality_index"] for h in history]

        stats = {
            "n_classifications": len(history),
            "class_distribution": distribution,
            "mean_confidence": round(float(np.mean(confidences)), 4),
            "mean_laterality": round(float(np.mean(lateralities)), 4),
            "has_baseline": user_id in self._baselines,
        }

        acc = self.get_accuracy(user_id)
        if acc > 0:
            stats["accuracy"] = round(acc, 4)

        return stats

    def get_history(
        self, last_n: Optional[int] = None, user_id: str = "default"
    ) -> List[Dict]:
        """Get classification history.

        Args:
            last_n: Return only the last N entries. None = return all.
            user_id: User identifier.

        Returns:
            List of classification result dicts.
        """
        history = self._history.get(user_id, [])
        if last_n is not None:
            return history[-last_n:]
        return list(history)

    def reset(self, user_id: str = "default") -> None:
        """Clear all state for a user (baseline, history, labels)."""
        self._baselines.pop(user_id, None)
        self._history.pop(user_id, None)
        self._labels.pop(user_id, None)

    # ── Private helpers ─────────────────────────────────────────────

    @staticmethod
    def _ensure_2d(signals: np.ndarray) -> np.ndarray:
        """Ensure signals are (n_channels, n_samples)."""
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)
        return signals

    @staticmethod
    def _band_power(
        signal: np.ndarray, fs: float, low: float, high: float
    ) -> float:
        """Compute band power via Welch PSD.

        Args:
            signal: 1D EEG signal.
            fs: Sampling rate.
            low: Lower band edge (Hz).
            high: Upper band edge (Hz).

        Returns:
            Integrated power in the band (uV^2/Hz * Hz = uV^2).
        """
        nperseg = min(len(signal), int(fs * 2))
        if nperseg < 4:
            return 0.0
        try:
            freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg)
        except Exception:
            return 0.0
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return 0.0
        if hasattr(np, "trapezoid"):
            return float(_trapezoid(psd[mask], freqs[mask]))
        return float(np.trapz(psd[mask], freqs[mask]))

    @staticmethod
    def _compute_laterality(mu_erd: np.ndarray, n_ch: int) -> float:
        """Compute laterality index from mu ERD.

        Uses TP9 (ch0, left) and TP10 (ch3, right) if available.
        laterality = (suppression_right - suppression_left) expressed as:
          (ERD_right - ERD_left) / (|ERD_left| + |ERD_right| + eps)

        Since ERD is negative for suppression:
          Positive = left hemisphere more suppressed = right hand imagery
          Negative = right hemisphere more suppressed = left hand imagery
        """
        if n_ch >= 4:
            erd_left = mu_erd[CH_TP9]    # TP9 — left hemisphere
            erd_right = mu_erd[CH_TP10]  # TP10 — right hemisphere
        elif n_ch >= 2:
            erd_left = mu_erd[0]
            erd_right = mu_erd[1]
        else:
            return 0.0

        denom = abs(erd_left) + abs(erd_right) + 1e-12
        return float((erd_right - erd_left) / denom)

    @staticmethod
    def _compute_probabilities(
        mu_erd: np.ndarray,
        beta_erd: np.ndarray,
        laterality: float,
        n_ch: int,
    ) -> tuple:
        """Compute class probabilities from ERD features.

        Feature logic:
        - Strong laterality -> hand classification (left or right)
        - Bilateral mu/beta suppression -> feet
        - Minimal ERD -> rest

        Returns:
            (probs dict, confidence float)
        """
        probs = {c: 0.0 for c in CLASSES}

        # Compute total mu ERD magnitude across channels
        mu_erd_mean = float(np.mean(mu_erd))
        beta_erd_mean = float(np.mean(beta_erd))
        total_erd = abs(mu_erd_mean) + abs(beta_erd_mean)

        # Bilateral suppression indicator (all channels suppressed similarly)
        if n_ch >= 4:
            mu_bilateral = float(
                np.mean([-mu_erd[CH_TP9], -mu_erd[CH_TP10]])
            )
            mu_bilateral_balance = 1.0 - abs(mu_erd[CH_TP9] - mu_erd[CH_TP10]) / (
                abs(mu_erd[CH_TP9]) + abs(mu_erd[CH_TP10]) + 1e-12
            )
        else:
            mu_bilateral = float(-mu_erd_mean)
            mu_bilateral_balance = 0.5

        abs_lat = abs(laterality)

        # Left hand: negative laterality (right hemisphere suppression)
        if laterality < 0:
            probs["left_hand"] = abs_lat * 0.7 + max(0, -mu_erd_mean) * 0.3
        else:
            probs["left_hand"] = max(0, 0.1 - abs_lat * 0.3)

        # Right hand: positive laterality (left hemisphere suppression)
        if laterality > 0:
            probs["right_hand"] = abs_lat * 0.7 + max(0, -mu_erd_mean) * 0.3
        else:
            probs["right_hand"] = max(0, 0.1 - abs_lat * 0.3)

        # Both feet: bilateral suppression with balanced laterality
        feet_score = max(0, mu_bilateral) * mu_bilateral_balance
        probs["both_feet"] = feet_score * (1.0 - abs_lat * 0.8)

        # Rest: minimal ERD
        rest_score = max(0, 1.0 - total_erd * 3.0)
        probs["rest"] = rest_score

        # Normalize to sum to 1
        total = sum(probs.values())
        if total > 1e-12:
            for c in CLASSES:
                probs[c] = probs[c] / total
        else:
            # Uniform if nothing stands out
            for c in CLASSES:
                probs[c] = 0.25

        # Confidence = how far the max probability is from uniform (0.25)
        max_prob = max(probs.values())
        confidence = float(np.clip((max_prob - 0.25) / 0.75, 0.0, 1.0))

        return probs, confidence
