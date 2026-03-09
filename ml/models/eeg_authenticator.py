"""EEG-based biometric authentication via spectral fingerprinting.

Enrollment: record resting-state EEG, extract PSD template per channel.
Verification: extract PSD from a segment, cosine similarity to template.

Scientific basis:
  - EEG spectral features are individually unique and temporally stable
  - PSD (Power Spectral Density) across channels forms a biometric fingerprint
  - Cosine similarity in log-PSD space is robust to amplitude scaling

References:
    MDPI Sensors (2025) -- PSD features from resting EEG
    Expert Systems (2024) -- 99.73% EEG identification accuracy

Architecture:
    1. Extract PSD via Welch's method (2-sec windows) per channel
    2. Keep freq bins in 0.5-45 Hz range
    3. Log-transform power for better normality
    4. Concatenate across channels into single template vector
    5. Cosine similarity for verification/identification
"""

import numpy as np
from typing import Dict, List, Optional
from scipy.signal import welch


class EEGAuthenticator:
    """EEG-based biometric authentication via spectral fingerprinting.

    Enrollment: record 2-min resting EEG, extract PSD template per channel.
    Verification: extract PSD from 5-sec segment, cosine similarity to template.

    References:
        MDPI Sensors (2025) -- PSD features from resting EEG
        Expert Systems (2024) -- 99.73% EEG identification
    """

    def __init__(self, match_threshold: float = 0.85, n_channels: int = 4):
        self._templates: Dict[str, np.ndarray] = {}  # user_id -> PSD template
        self._match_threshold = match_threshold
        self._n_channels = n_channels
        self._freq_range = (0.5, 45.0)  # Hz

    def enroll(
        self,
        signals: np.ndarray,
        fs: float = 256.0,
        user_id: str = "default",
    ) -> dict:
        """Enroll a user with resting-state EEG.

        Args:
            signals: (n_channels, n_samples) EEG array, or 1D for single channel
            fs: Sampling rate in Hz
            user_id: User identifier

        Returns:
            dict with keys: enrolled, user_id, template_size, duration_sec,
            quality_score
        """
        signals = self._ensure_2d(signals)
        n_channels, n_samples = signals.shape
        duration_sec = n_samples / fs

        template = self._extract_psd_template(signals, fs)
        self._templates[user_id] = template

        quality_score = self._compute_quality(signals)

        return {
            "enrolled": True,
            "user_id": user_id,
            "template_size": len(template),
            "duration_sec": float(duration_sec),
            "quality_score": float(quality_score),
        }

    def verify(
        self,
        signals: np.ndarray,
        fs: float = 256.0,
        claimed_id: str = "default",
    ) -> dict:
        """Verify identity against enrolled template.

        Args:
            signals: (n_channels, n_samples) EEG array (min 5 sec recommended)
            fs: Sampling rate in Hz
            claimed_id: User ID to verify against

        Returns:
            dict with keys: match, similarity, claimed_id, threshold, confidence
        """
        if claimed_id not in self._templates:
            return {
                "match": False,
                "similarity": 0.0,
                "claimed_id": claimed_id,
                "threshold": self._match_threshold,
                "confidence": "low",
            }

        signals = self._ensure_2d(signals)
        probe_template = self._extract_psd_template(signals, fs)
        enrolled_template = self._templates[claimed_id]

        similarity = self._cosine_similarity(probe_template, enrolled_template)
        match = similarity >= self._match_threshold

        confidence = self._classify_confidence(similarity)

        return {
            "match": match,
            "similarity": float(similarity),
            "claimed_id": claimed_id,
            "threshold": self._match_threshold,
            "confidence": confidence,
        }

    def identify(self, signals: np.ndarray, fs: float = 256.0) -> dict:
        """Identify user from all enrolled templates.

        Returns:
            dict with keys: identified_user, similarity, all_scores, n_enrolled
        """
        if not self._templates:
            return {
                "identified_user": None,
                "similarity": 0.0,
                "all_scores": {},
                "n_enrolled": 0,
            }

        signals = self._ensure_2d(signals)
        probe_template = self._extract_psd_template(signals, fs)

        all_scores: Dict[str, float] = {}
        for uid, enrolled_template in self._templates.items():
            all_scores[uid] = float(
                self._cosine_similarity(probe_template, enrolled_template)
            )

        best_user = max(all_scores, key=all_scores.get)  # type: ignore[arg-type]
        best_score = all_scores[best_user]

        identified = best_user if best_score >= self._match_threshold else None

        return {
            "identified_user": identified,
            "similarity": float(best_score),
            "all_scores": all_scores,
            "n_enrolled": len(self._templates),
        }

    def get_enrolled_users(self) -> List[str]:
        """Return list of enrolled user IDs."""
        return list(self._templates.keys())

    def remove_user(self, user_id: str) -> bool:
        """Remove enrolled user template. Returns True if found and removed."""
        if user_id in self._templates:
            del self._templates[user_id]
            return True
        return False

    def _extract_psd_template(
        self, signals: np.ndarray, fs: float
    ) -> np.ndarray:
        """Extract flattened PSD template from multichannel EEG.

        Uses scipy.signal.welch with 2-sec windows.
        Keeps only freq bins within self._freq_range.
        Log-transforms power for better normality.
        Returns: 1D array of log-PSD values across channels.
        """
        signals = self._ensure_2d(signals)
        n_channels, n_samples = signals.shape

        # Welch parameters: 2-second window, 50% overlap
        nperseg = min(int(2.0 * fs), n_samples)
        noverlap = nperseg // 2

        all_psd = []
        for ch in range(n_channels):
            freqs, psd = welch(
                signals[ch],
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                window="hann",
            )
            # Keep only frequency bins within range
            freq_mask = (freqs >= self._freq_range[0]) & (
                freqs <= self._freq_range[1]
            )
            psd_band = psd[freq_mask]

            # Log-transform (add small epsilon to avoid log(0))
            log_psd = np.log10(psd_band + 1e-20)
            all_psd.append(log_psd)

        return np.concatenate(all_psd)

    def _compute_quality(self, signals: np.ndarray) -> float:
        """Compute enrollment quality score (0-1).

        Measures spectral concentration: ratio of power in dominant frequency
        bins vs total broadband power. A clean signal with strong spectral
        peaks has high concentration; a noisy signal has flat (uniform) PSD
        and lower concentration.

        Technically: 1 - (spectral_entropy / max_entropy). Clean signals
        have low spectral entropy (peaky PSD), noisy signals have high
        spectral entropy (flat PSD).
        """
        signals = self._ensure_2d(signals)
        n_channels, n_samples = signals.shape

        nperseg = min(int(2.0 * 256), n_samples)
        noverlap = nperseg // 2

        entropies = []
        for ch in range(n_channels):
            _, psd = welch(signals[ch], fs=256.0, nperseg=nperseg,
                           noverlap=noverlap, window="hann")
            # Normalize PSD to probability distribution
            psd_sum = psd.sum()
            if psd_sum < 1e-20:
                entropies.append(1.0)  # flat/dead signal = max entropy
                continue
            p = psd / psd_sum
            # Shannon entropy (avoid log(0))
            p_safe = p[p > 1e-20]
            entropy = -np.sum(p_safe * np.log(p_safe))
            max_entropy = np.log(len(psd))
            # Normalize to 0-1
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
            entropies.append(norm_entropy)

        mean_entropy = np.mean(entropies)
        # Quality = 1 - normalized entropy: peaky PSD = high quality
        quality = float(1.0 - mean_entropy)
        return float(np.clip(quality, 0.0, 1.0))

    def _classify_confidence(self, similarity: float) -> str:
        """Classify verification confidence level.

        high: similarity > threshold + 0.10
        medium: similarity >= threshold
        low: everything else (including marginal matches)
        """
        if similarity > self._match_threshold + 0.10:
            return "high"
        elif similarity >= self._match_threshold:
            return "medium"
        else:
            return "low"

    def _ensure_2d(self, signals: np.ndarray) -> np.ndarray:
        """Ensure signals are 2D (n_channels, n_samples)."""
        if signals.ndim == 1:
            return signals.reshape(1, -1)
        return signals

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(dot / (norm_a * norm_b))
