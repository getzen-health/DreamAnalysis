"""Imagined speech command decoder from EEG signals.

Decodes imagined speech commands (e.g. "yes", "no", "stop", "go", "help")
from 4-channel Muse 2 EEG using spectral template matching. This is HIGHLY
experimental on consumer-grade 4-channel EEG — research-grade imagined speech
BCIs use 64-128 channels over speech-motor cortex and achieve only 55-65%
accuracy for 5 classes even with optimal placement.

Imagined speech neuroscience (relevant to Muse 2):
- Imagined speech produces mu (8-13 Hz) suppression at speech-motor areas.
  On Muse 2, AF7 (left frontal, ch1) is closest to Broca's area — the primary
  speech production cortex in the left hemisphere.
- Different imagined words produce subtly different spectral patterns, mainly
  in theta (4-8 Hz) and mu/alpha (8-13 Hz) bands at left-hemisphere channels.
- Left-hemisphere channels (AF7, TP9) show more activation during speech imagery
  than right-hemisphere channels (AF8, TP10).
- Frontal theta increases during effortful speech imagery (working memory load).
- Beta rebound (13-30 Hz increase) occurs after a speech imagery attempt ends.

Approach:
- During calibration: extract 20 spectral features (4 channels x 5 bands)
  per labeled example, store per class as templates.
- During decode: extract same features, compute cosine similarity to each
  class template (mean of calibration samples for that class).
- Predict class with highest similarity. Confidence = max_sim - second_max_sim.

Honest limitations:
- 55-65% accuracy for 5 classes is realistic (chance = 20%).
- Muse 2 has no electrodes over speech-motor cortex (C3/C4/Cz).
- AF7 is the only electrode remotely near Broca's area, and it's contaminated
  by EMG from frontalis muscle and Fpz reference noise.
- Results should NEVER be used for critical applications.

References:
    Nguyen et al. (2018) — Inferring imagined speech using EEG signals:
        a new approach using Riemannian manifold features. J Neural Eng.
    DaSalla et al. (2009) — Single-trial classification of vowel speech
        imagery using common spatial patterns. Neural Networks, 22(9).
    Cooney et al. (2020) — Evaluation of hyperparameter optimization in
        machine and deep learning methods for decoding imagined speech EEG.
        Sensors, 20(16), 4629.
"""
from typing import Dict, List, Optional, Tuple

import logging
import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)

# Muse 2 channel mapping (BrainFlow board_id 38)
CH_TP9 = 0   # Left temporal
CH_AF7 = 1   # Left frontal — closest to Broca's area
CH_AF8 = 2   # Right frontal
CH_TP10 = 3  # Right temporal

# Frequency bands for feature extraction
BANDS: List[Tuple[str, float, float]] = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
]

DEFAULT_COMMANDS = ("yes", "no", "stop", "go", "help")

DISCLAIMER = (
    "Imagined speech decoding from 4-channel consumer EEG is experimental. "
    "Accuracy is realistically 55-65% for 5 classes (chance = 20%). "
    "Not reliable for critical applications."
)

MAX_HISTORY = 500


class ImaginedSpeechDecoder:
    """Template-matching imagined speech decoder for 4-channel Muse 2 EEG.

    Uses spectral feature extraction (band powers per channel) and cosine
    similarity to user-calibrated templates to decode imagined speech commands.

    This is experimental. Muse 2 has no electrodes directly over speech-motor
    cortex. AF7 (left frontal) is the closest proxy to Broca's area but is
    heavily contaminated by frontalis EMG and Fpz reference noise.
    """

    def __init__(self, fs: float = 256.0, n_classes: int = 5):
        """Initialize the imagined speech decoder.

        Args:
            fs: Sampling rate in Hz (Muse 2 default: 256).
            n_classes: Maximum number of command classes to support.
        """
        self._fs = fs
        self._n_classes = n_classes
        # Per-user state
        self._calibration: Dict[str, Dict[str, List[np.ndarray]]] = {}
        self._templates: Dict[str, Dict[str, np.ndarray]] = {}
        self._history: Dict[str, List[Dict]] = {}
        self._decode_count: Dict[str, int] = {}

    # ── Public API ──────────────────────────────────────────────────

    def calibrate(
        self,
        signals: np.ndarray,
        label: str,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Add a labeled calibration sample for template building.

        Collects EEG examples for each command class. After collecting at least
        one example per class, templates are computed as the mean feature vector
        across all samples for that class.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array.
            label: Command label string (e.g. "yes", "no", "stop").
            fs: Sampling rate override.
            user_id: User identifier for multi-user support.

        Returns:
            Dict with calibrated, n_samples, n_classes_seen, classes_seen.
        """
        fs = fs or self._fs
        signals_2d = self._ensure_2d(signals)
        features = self._extract_features(signals_2d, fs)

        if user_id not in self._calibration:
            self._calibration[user_id] = {}

        if label not in self._calibration[user_id]:
            if len(self._calibration[user_id]) >= self._n_classes:
                logger.warning(
                    "Max classes (%d) reached for user %s. Ignoring label '%s'.",
                    self._n_classes, user_id, label,
                )
                classes_seen = sorted(self._calibration[user_id].keys())
                total = sum(
                    len(v) for v in self._calibration[user_id].values()
                )
                return {
                    "calibrated": len(classes_seen) >= 2,
                    "n_samples": total,
                    "n_classes_seen": len(classes_seen),
                    "classes_seen": classes_seen,
                }
            self._calibration[user_id][label] = []

        self._calibration[user_id][label].append(features)

        # Rebuild templates for this user
        self._rebuild_templates(user_id)

        classes_seen = sorted(self._calibration[user_id].keys())
        total_samples = sum(
            len(v) for v in self._calibration[user_id].values()
        )

        return {
            "calibrated": len(classes_seen) >= 2,
            "n_samples": total_samples,
            "n_classes_seen": len(classes_seen),
            "classes_seen": classes_seen,
        }

    def decode(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Decode imagined speech command from EEG.

        Extracts spectral features from the input EEG and computes cosine
        similarity against each calibrated class template. Returns the most
        similar class, or None if not calibrated.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array.
            fs: Sampling rate override.
            user_id: User identifier.

        Returns:
            Dict with predicted_command, confidence, probabilities,
            is_calibrated, n_calibration_samples, disclaimer.
        """
        fs = fs or self._fs
        signals_2d = self._ensure_2d(signals)
        features = self._extract_features(signals_2d, fs)

        templates = self._templates.get(user_id, {})
        cal_data = self._calibration.get(user_id, {})
        n_cal_samples = sum(len(v) for v in cal_data.values())
        is_calibrated = len(templates) >= 2

        if not is_calibrated:
            result = {
                "predicted_command": None,
                "confidence": 0.0,
                "probabilities": {},
                "is_calibrated": False,
                "n_calibration_samples": n_cal_samples,
                "disclaimer": DISCLAIMER,
            }
            self._append_history(user_id, result)
            return result

        # Compute cosine similarity to each template
        similarities = {}
        for label, template in templates.items():
            similarities[label] = self._cosine_similarity(features, template)

        # Convert similarities to probabilities via softmax
        probabilities = self._softmax_from_similarities(similarities)

        # Determine prediction and confidence
        sorted_sims = sorted(similarities.values(), reverse=True)
        best_label = max(similarities, key=similarities.get)
        if len(sorted_sims) >= 2:
            confidence = float(
                np.clip(sorted_sims[0] - sorted_sims[1], 0.0, 1.0)
            )
        else:
            confidence = float(np.clip(sorted_sims[0], 0.0, 1.0))

        result = {
            "predicted_command": best_label,
            "confidence": round(confidence, 4),
            "probabilities": {
                k: round(float(v), 6) for k, v in probabilities.items()
            },
            "is_calibrated": True,
            "n_calibration_samples": n_cal_samples,
            "disclaimer": DISCLAIMER,
        }

        self._append_history(user_id, result)
        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get session statistics for a user.

        Returns:
            Dict with n_decodings, n_calibration_samples, classes_available.
        """
        cal_data = self._calibration.get(user_id, {})
        n_cal = sum(len(v) for v in cal_data.values())
        classes_available = sorted(cal_data.keys())
        n_decodings = self._decode_count.get(user_id, 0)

        return {
            "n_decodings": n_decodings,
            "n_calibration_samples": n_cal,
            "classes_available": classes_available,
        }

    def get_history(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> List[Dict]:
        """Get decode history for a user.

        Args:
            user_id: User identifier.
            last_n: Return only the last N entries. None = all.

        Returns:
            List of decode result dicts.
        """
        history = self._history.get(user_id, [])
        if last_n is not None:
            return history[-last_n:]
        return list(history)

    def reset(self, user_id: str = "default") -> None:
        """Clear all state for a user (calibration, templates, history)."""
        self._calibration.pop(user_id, None)
        self._templates.pop(user_id, None)
        self._history.pop(user_id, None)
        self._decode_count.pop(user_id, None)

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
        signal_1d: np.ndarray, fs: float, low: float, high: float
    ) -> float:
        """Compute band power via Welch PSD.

        Args:
            signal_1d: 1D EEG signal.
            fs: Sampling rate.
            low: Lower band edge (Hz).
            high: Upper band edge (Hz).

        Returns:
            Integrated power in the band (uV^2).
        """
        nperseg = min(len(signal_1d), int(fs * 2))
        if nperseg < 4:
            return 0.0
        try:
            freqs, psd = scipy_signal.welch(
                signal_1d, fs=fs, nperseg=nperseg
            )
        except Exception:
            return 0.0
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return 0.0
        return float(_trapezoid(psd[mask], freqs[mask]))

    def _extract_features(
        self, signals: np.ndarray, fs: float
    ) -> np.ndarray:
        """Extract spectral features: band power per channel per band.

        For 4 channels and 5 bands, this produces a 20-element feature vector.
        Feature ordering: [ch0_delta, ch0_theta, ..., ch0_gamma, ch1_delta, ...]

        Args:
            signals: (n_channels, n_samples) EEG array.
            fs: Sampling rate.

        Returns:
            1D numpy array of features.
        """
        n_ch = signals.shape[0]
        features = []
        for ch in range(n_ch):
            for _, low, high in BANDS:
                features.append(self._band_power(signals[ch], fs, low, high))
        return np.array(features, dtype=float)

    def _rebuild_templates(self, user_id: str) -> None:
        """Rebuild class templates from calibration data.

        Each template is the element-wise mean of all feature vectors
        collected for that class.
        """
        cal_data = self._calibration.get(user_id, {})
        templates = {}
        for label, feature_list in cal_data.items():
            if feature_list:
                templates[label] = np.mean(feature_list, axis=0)
        self._templates[user_id] = templates

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Returns 0.0 if either vector has zero norm or if dimensions mismatch
        (e.g. 1-channel decode against 4-channel calibration).
        """
        if a.shape != b.shape:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _softmax_from_similarities(
        similarities: Dict[str, float],
    ) -> Dict[str, float]:
        """Convert cosine similarities to probability distribution via softmax.

        Uses temperature scaling (T=0.1) to sharpen the distribution,
        since cosine similarities for similar spectral patterns tend to
        cluster near 1.0.
        """
        if not similarities:
            return {}
        temperature = 0.1
        labels = list(similarities.keys())
        values = np.array([similarities[l] for l in labels])
        # Shift for numerical stability
        shifted = (values - np.max(values)) / temperature
        exp_vals = np.exp(shifted)
        total = np.sum(exp_vals)
        if total < 1e-12:
            # Uniform fallback
            uniform = 1.0 / len(labels)
            return {l: uniform for l in labels}
        probs = exp_vals / total
        return {l: float(probs[i]) for i, l in enumerate(labels)}

    def _append_history(self, user_id: str, result: Dict) -> None:
        """Append a decode result to history, capped at MAX_HISTORY."""
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > MAX_HISTORY:
            self._history[user_id] = self._history[user_id][-MAX_HISTORY:]

        if user_id not in self._decode_count:
            self._decode_count[user_id] = 0
        self._decode_count[user_id] += 1
