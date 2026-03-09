"""EEG-based passive biometric authentication via spectral fingerprinting.

Each person's resting EEG has highly stable spectral characteristics — especially
alpha peak frequency and beta/alpha band-power ratios. These serve as a biometric
signature that cannot be forged unlike passwords or physical biometrics.

Accuracy (feature-based PSD cosine similarity):
- 4-channel systems: 92-95% (MDPI Sensors, 2025 review)
- Deep learning variants: 99.73% (Expert Systems, 2024)

Key biomarkers:
- Alpha peak frequency: individually stable (r=0.53-0.66 test-retest)
- PSD shape 0.5-45 Hz: unique fingerprint per person
- Theta/alpha/beta band power ratios: low intra-subject variance

Privacy: only frequency-domain templates stored, never raw EEG.

References:
- MDPI Sensors 2025 (PMC12390388): PSD alpha/beta for passive authentication
- Expert Systems 2024: Wavelet-ResNet 99.73% accuracy
"""

import logging
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TEMPLATE_MIN_SECONDS = 10     # minimum enrollment data per segment
ENROLL_MIN_SEGMENTS = 3       # need ≥3 segments to build stable template
VERIFY_THRESHOLD = 0.75       # cosine similarity threshold for match
PSD_FREQ_LO = 0.5             # Hz — lowest frequency in PSD fingerprint
PSD_FREQ_HI = 45.0            # Hz — highest frequency
PSD_RESOLUTION = 0.5          # Hz per bin → 89 bins per channel (0.5-45 Hz)


# ── EEGAuthenticator ──────────────────────────────────────────────────────────

class EEGAuthenticator:
    """Passive EEG biometric authentication via PSD template matching.

    Usage:
        auth = EEGAuthenticator()

        # Enrollment (call 3+ times with resting EEG segments):
        auth.enroll(eeg_4ch, fs=256, user_id="alice")
        # ...repeat for ≥3 two-minute epochs...

        # Verification:
        result = auth.verify(eeg_4ch, fs=256, user_id="alice")
        # → {"match": True, "similarity": 0.912, "threshold": 0.75}
    """

    def __init__(self):
        self._lock = threading.Lock()
        # user_id → list of PSD vectors (enrollment templates)
        self._templates: Dict[str, List[np.ndarray]] = {}
        # user_id → averaged template (set after finalize_enrollment)
        self._mean_templates: Dict[str, np.ndarray] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def enroll(
        self,
        eeg: np.ndarray,
        fs: float = 256.0,
        user_id: str = "default",
    ) -> Dict:
        """Add one EEG segment to the enrollment template for user_id.

        Call ≥3 times with fresh resting-state epochs (eyes closed, relaxed).
        After the 3rd call, the template is finalized automatically.

        Args:
            eeg: shape (4, n_samples) or (n_samples,) — raw µV
            fs: sampling rate in Hz
            user_id: unique user identifier

        Returns:
            {"enrolled_segments": int, "template_ready": bool, "user_id": str}
        """
        psd = self._extract_psd_vector(eeg, fs)
        if psd is None:
            return {
                "enrolled_segments": self._segment_count(user_id),
                "template_ready": self._is_ready(user_id),
                "user_id": user_id,
                "error": "Signal too short for reliable PSD",
            }

        with self._lock:
            if user_id not in self._templates:
                self._templates[user_id] = []
            self._templates[user_id].append(psd)
            n_segs = len(self._templates[user_id])

            # Finalize template as soon as we have ≥3 segments
            if n_segs >= ENROLL_MIN_SEGMENTS:
                self._mean_templates[user_id] = np.mean(
                    self._templates[user_id], axis=0
                )
                logger.info(
                    "EEG biometric template finalized for user=%s (%d segments)",
                    user_id,
                    n_segs,
                )

        return {
            "enrolled_segments": n_segs,
            "template_ready": n_segs >= ENROLL_MIN_SEGMENTS,
            "user_id": user_id,
        }

    def verify(
        self,
        eeg: np.ndarray,
        fs: float = 256.0,
        user_id: str = "default",
    ) -> Dict:
        """Verify whether the EEG matches the enrolled template for user_id.

        Args:
            eeg: shape (4, n_samples) or (n_samples,) — raw µV
            fs: sampling rate in Hz
            user_id: user to verify against

        Returns:
            {
              "match": bool,
              "similarity": float (0-1, cosine similarity),
              "threshold": float,
              "template_ready": bool,
              "user_id": str
            }
        """
        if not self._is_ready(user_id):
            return {
                "match": False,
                "similarity": 0.0,
                "threshold": VERIFY_THRESHOLD,
                "template_ready": False,
                "user_id": user_id,
                "error": f"No template for user '{user_id}'. Enroll first.",
            }

        psd = self._extract_psd_vector(eeg, fs)
        if psd is None:
            return {
                "match": False,
                "similarity": 0.0,
                "threshold": VERIFY_THRESHOLD,
                "template_ready": True,
                "user_id": user_id,
                "error": "Signal too short for verification",
            }

        with self._lock:
            template = self._mean_templates[user_id]

        similarity = float(_cosine_similarity(psd, template))
        match = similarity >= VERIFY_THRESHOLD

        return {
            "match": match,
            "similarity": round(similarity, 4),
            "threshold": VERIFY_THRESHOLD,
            "template_ready": True,
            "user_id": user_id,
        }

    def identify(
        self,
        eeg: np.ndarray,
        fs: float = 256.0,
    ) -> Dict:
        """1-of-N identification: find the best-matching enrolled user.

        Returns:
            {
              "identified_user": str or None,
              "similarity": float,
              "threshold": float,
              "candidates": [{user_id, similarity}, ...] sorted desc
            }
        """
        psd = self._extract_psd_vector(eeg, fs)
        if psd is None:
            return {
                "identified_user": None,
                "similarity": 0.0,
                "threshold": VERIFY_THRESHOLD,
                "candidates": [],
                "error": "Signal too short",
            }

        with self._lock:
            if not self._mean_templates:
                return {
                    "identified_user": None,
                    "similarity": 0.0,
                    "threshold": VERIFY_THRESHOLD,
                    "candidates": [],
                    "error": "No enrolled users",
                }
            scores = [
                {"user_id": uid, "similarity": round(float(_cosine_similarity(psd, tmpl)), 4)}
                for uid, tmpl in self._mean_templates.items()
            ]

        scores.sort(key=lambda x: x["similarity"], reverse=True)
        best = scores[0]
        identified = best["user_id"] if best["similarity"] >= VERIFY_THRESHOLD else None

        return {
            "identified_user": identified,
            "similarity": best["similarity"],
            "threshold": VERIFY_THRESHOLD,
            "candidates": scores,
        }

    def delete_template(self, user_id: str) -> Dict:
        """Remove all stored templates for user_id (GDPR right-to-erasure)."""
        with self._lock:
            removed = user_id in self._templates
            self._templates.pop(user_id, None)
            self._mean_templates.pop(user_id, None)
        return {"user_id": user_id, "deleted": removed}

    def get_status(self) -> Dict:
        """Return overview of enrolled users and template readiness."""
        with self._lock:
            return {
                "enrolled_users": list(self._templates.keys()),
                "ready_users": list(self._mean_templates.keys()),
                "verify_threshold": VERIFY_THRESHOLD,
                "enroll_min_segments": ENROLL_MIN_SEGMENTS,
            }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_psd_vector(
        self, eeg: np.ndarray, fs: float
    ) -> Optional[np.ndarray]:
        """Extract flattened PSD vector across all channels.

        Returns a 1-D feature vector: n_channels × n_bins, log-domain.
        Uses numpy-only Welch approximation (averaged periodogram).
        """
        try:
            # Normalise shape
            if eeg.ndim == 1:
                channels = [eeg]
            elif eeg.ndim == 2:
                channels = [eeg[i] for i in range(eeg.shape[0])]
            else:
                return None

            n_samples = len(channels[0])
            min_samples = int(fs * TEMPLATE_MIN_SECONDS)
            if n_samples < min_samples:
                logger.debug(
                    "Signal too short: %d < %d samples", n_samples, min_samples
                )
                return None

            # Hann-windowed averaged periodogram (Welch-like)
            seg_len = min(int(fs * 4), n_samples)  # 4-second segments
            hop = seg_len // 2
            freqs = np.fft.rfftfreq(seg_len, d=1.0 / fs)
            mask = (freqs >= PSD_FREQ_LO) & (freqs <= PSD_FREQ_HI)

            psds = []
            for ch in channels:
                ch = ch.astype(np.float32)
                segments = []
                start = 0
                while start + seg_len <= n_samples:
                    window = np.hanning(seg_len)
                    seg = ch[start : start + seg_len] * window
                    psd_seg = np.abs(np.fft.rfft(seg)) ** 2
                    segments.append(psd_seg[mask])
                    start += hop
                if not segments:
                    return None
                ch_psd = np.mean(segments, axis=0)
                # Log-domain: better normality, reduces outlier influence
                ch_psd = np.log1p(ch_psd)
                psds.append(ch_psd)

            return np.concatenate(psds).astype(np.float32)

        except Exception as exc:
            logger.debug("PSD extraction failed: %s", exc)
            return None

    def _segment_count(self, user_id: str) -> int:
        with self._lock:
            return len(self._templates.get(user_id, []))

    def _is_ready(self, user_id: str) -> bool:
        with self._lock:
            return user_id in self._mean_templates


# ── Math helpers ──────────────────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


# ── Singleton ─────────────────────────────────────────────────────────────────

_authenticator: Optional[EEGAuthenticator] = None
_auth_lock = threading.Lock()


def get_eeg_authenticator() -> EEGAuthenticator:
    global _authenticator
    with _auth_lock:
        if _authenticator is None:
            _authenticator = EEGAuthenticator()
    return _authenticator
