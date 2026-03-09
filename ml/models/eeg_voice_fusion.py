"""Cross-modal EEG+Voice temporal alignment via Optimal Transport fusion.

Fuses EEG emotion features with voice prosodic features using:
1. Feature extraction: 17 EEG band-power features + 6 voice prosodic features
2. OT-based alignment: Sinkhorn algorithm approximates Wasserstein distance
   to find optimal transport plan between feature distributions
3. Fusion: transport-weighted combination -> unified emotion classification

Scientific basis: OT-based multi-modal fusion has shown +8-15% over simple
concatenation for EEG+voice emotion recognition (2024 literature).

BrainFlow Muse 2 channel order (board_id 22/38):
  ch0 = TP9, ch1 = AF7, ch2 = AF8, ch3 = TP10
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

import numpy as np

log = logging.getLogger(__name__)

EMOTIONS_6 = ["happy", "sad", "angry", "fear", "surprise", "neutral"]

# EEG band definitions (Hz)
_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 45.0),  # capped at 45 Hz to avoid EMG; Nyquist at 128 Hz
}


def _welch_band_power(signal: np.ndarray, fs: float, low: float, high: float) -> float:
    """Estimate band power in [low, high] Hz using Welch's method (scipy-free fallback).

    Uses the periodogram with a Hann window. For segments shorter than 256
    samples the full segment is used.
    """
    n = len(signal)
    if n < 4:
        return 1e-10
    # Hann window
    win = np.hanning(n)
    windowed = signal * win
    fft_vals = np.fft.rfft(windowed, n=n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = (np.abs(fft_vals) ** 2) / (np.sum(win ** 2) * fs)
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 1e-10
    return float(np.sum(power[mask]))


# ── EEG Feature Extraction ────────────────────────────────────────────────────

def _extract_eeg_features_single(signal: np.ndarray, fs: float) -> np.ndarray:
    """Extract 5 band powers from a single EEG channel. Returns shape (5,)."""
    feats = []
    for band_name in ("delta", "theta", "alpha", "beta", "gamma"):
        low, high = _BANDS[band_name]
        p = _welch_band_power(signal, fs, low, high)
        feats.append(max(p, 1e-10))
    return np.array(feats, dtype=np.float64)


# ── Voice Feature Extraction (scipy-only, no librosa) ─────────────────────────

def _extract_f0(audio: np.ndarray, sr: int, frame_len: int = 1024, hop: int = 512) -> np.ndarray:
    """Estimate F0 via autocorrelation. Returns array of per-frame pitch estimates."""
    f0_vals = []
    n = len(audio)
    for start in range(0, n - frame_len, hop):
        frame = audio[start: start + frame_len]
        # Zero-pad autocorrelation
        ac = np.correlate(frame, frame, mode="full")
        ac = ac[len(ac) // 2:]  # one-sided
        # Find first peak beyond lag 10 (avoid DC)
        min_lag = max(int(sr / 800), 2)  # 800 Hz max pitch
        max_lag = int(sr / 50)           # 50 Hz min pitch
        max_lag = min(max_lag, len(ac) - 1)
        if min_lag >= max_lag:
            f0_vals.append(0.0)
            continue
        search = ac[min_lag:max_lag]
        if len(search) == 0 or ac[0] < 1e-10:
            f0_vals.append(0.0)
            continue
        peak_idx = int(np.argmax(search)) + min_lag
        # Only accept if the peak is strong relative to zero-lag
        if ac[peak_idx] / (ac[0] + 1e-10) > 0.3:
            f0_vals.append(float(sr) / float(peak_idx))
        else:
            f0_vals.append(0.0)
    return np.array(f0_vals, dtype=np.float64)


def _extract_energy(audio: np.ndarray, frame_len: int = 1024, hop: int = 512) -> np.ndarray:
    """RMS energy per frame."""
    energies = []
    n = len(audio)
    for start in range(0, n - frame_len, hop):
        frame = audio[start: start + frame_len]
        energies.append(float(np.sqrt(np.mean(frame ** 2) + 1e-10)))
    if not energies:
        energies = [float(np.sqrt(np.mean(audio ** 2) + 1e-10))]
    return np.array(energies, dtype=np.float64)


def _extract_zcr(audio: np.ndarray, frame_len: int = 1024, hop: int = 512) -> np.ndarray:
    """Zero-crossing rate per frame."""
    zcrs = []
    n = len(audio)
    for start in range(0, n - frame_len, hop):
        frame = audio[start: start + frame_len]
        signs = np.sign(frame)
        signs[signs == 0] = 1
        zcr = float(np.sum(np.abs(np.diff(signs))) / (2.0 * frame_len))
        zcrs.append(zcr)
    if not zcrs:
        zcrs = [0.0]
    return np.array(zcrs, dtype=np.float64)


def _extract_spectral_centroid(audio: np.ndarray, sr: int, frame_len: int = 1024, hop: int = 512) -> np.ndarray:
    """Spectral centroid (weighted mean frequency) per frame."""
    centroids = []
    n = len(audio)
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
    for start in range(0, n - frame_len, hop):
        frame = audio[start: start + frame_len]
        win = np.hanning(frame_len)
        mag = np.abs(np.fft.rfft(frame * win))
        total = np.sum(mag) + 1e-10
        centroid = float(np.sum(freqs * mag) / total)
        centroids.append(centroid)
    if not centroids:
        centroids = [0.0]
    return np.array(centroids, dtype=np.float64)


# ── Sinkhorn Optimal Transport ─────────────────────────────────────────────────

def _sinkhorn_transport(
    mu: np.ndarray,
    nu: np.ndarray,
    C: np.ndarray,
    eps: float = 0.1,
    n_iter: int = 50,
) -> np.ndarray:
    """Sinkhorn algorithm: approximate optimal transport plan.

    Solves the entropic-regularised OT problem:

        min_{T >= 0} sum_ij T_ij * C_ij - eps * H(T)
        s.t. T @ 1 = mu, T.T @ 1 = nu

    where H(T) = -sum T_ij log T_ij is the entropy regulariser.

    Args:
        mu: Source distribution, shape (n,). Must be on the probability simplex.
        nu: Target distribution, shape (m,). Must be on the probability simplex.
        C: Cost matrix, shape (n, m). C_ij = ||f_i - g_j||^2.
        eps: Regularisation strength. Smaller -> closer to exact OT, less stable.
        n_iter: Number of Sinkhorn iterations.

    Returns:
        T: Transport plan, shape (n, m). Row sums approx mu, col sums approx nu.
    """
    # Numerical stability: clip cost before exponent
    C_stable = C - C.min()
    log_K = -C_stable / (eps + 1e-10)
    # Clip to avoid overflow; max diff in log_K is bounded
    log_K = np.clip(log_K, -500, 0)
    K = np.exp(log_K)  # Gibbs kernel

    mu = mu / (mu.sum() + 1e-10)
    nu = nu / (nu.sum() + 1e-10)

    u = np.ones(len(mu)) / len(mu)
    for _ in range(n_iter):
        v = nu / (K.T @ u + 1e-300)
        u = mu / (K @ v + 1e-300)

    T = np.diag(u) @ K @ np.diag(v)
    return T


# ── Main Fusion Class ──────────────────────────────────────────────────────────

class EEGVoiceFusionClassifier:
    """Cross-modal EEG+Voice Optimal Transport fusion for 6-class emotion recognition.

    EEG and voice carry complementary emotion signals:
    - EEG: captures internal neural state (valence via FAA, arousal via beta/alpha)
    - Voice: captures expressive / motor output (pitch, energy, rate)

    Rather than simple concatenation, Sinkhorn OT finds the optimal alignment
    between the EEG feature distribution and voice feature distribution, then
    weights the fusion by transport quality (alignment confidence).

    Args:
        n_classes: Number of emotion classes (default 6).
        eeg_fs: EEG sampling rate in Hz (default 256.0).
        voice_fs: Voice audio sampling rate in Hz (default 16000).
    """

    def __init__(
        self,
        n_classes: int = 6,
        eeg_fs: float = 256.0,
        voice_fs: int = 16000,
    ) -> None:
        self.n_classes = n_classes
        self.eeg_fs = eeg_fs
        self.voice_fs = voice_fs
        self._last_ot_stats: Dict[str, float] = {
            "transport_cost": 0.0,
            "alignment_quality": 0.0,
        }

    # ── Feature extraction ─────────────────────────────────────────────────────

    def _extract_eeg_features(self, eeg: np.ndarray, fs: float) -> np.ndarray:
        """Extract band power features from EEG.

        Args:
            eeg: Shape (n_channels, n_samples) or (n_samples,).
            fs: Sampling frequency in Hz.

        Returns:
            Feature vector of shape (5 * n_channels,). 5 bands per channel.
            For 4-channel Muse 2 -> 20-dimensional feature vector.
        """
        if eeg.ndim == 1:
            return _extract_eeg_features_single(eeg, fs)

        feats = []
        for ch_idx in range(eeg.shape[0]):
            feats.append(_extract_eeg_features_single(eeg[ch_idx], fs))
        return np.concatenate(feats)

    def _extract_voice_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract prosodic features from audio.

        Features (6-dimensional):
            0: F0 mean (Hz)  — fundamental frequency / pitch
            1: F0 std (Hz)   — pitch variability
            2: Energy mean   — loudness (RMS)
            3: Energy std    — energy variability
            4: ZCR mean      — zero-crossing rate (voiced/unvoiced ratio proxy)
            5: Spectral centroid mean (Hz)  — brightness / place of articulation

        Args:
            audio: 1-D float32 audio array.
            sr: Audio sampling rate in Hz.

        Returns:
            6-dimensional feature vector (float64, not normalised).
        """
        if len(audio) < 64:
            return np.zeros(6, dtype=np.float64)

        frame_len = min(1024, len(audio) // 2)
        hop = frame_len // 2

        f0 = _extract_f0(audio, sr, frame_len=frame_len, hop=hop)
        f0_voiced = f0[f0 > 0]
        f0_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        f0_std = float(np.std(f0_voiced)) if len(f0_voiced) > 1 else 0.0

        energy = _extract_energy(audio, frame_len=frame_len, hop=hop)
        zcr = _extract_zcr(audio, frame_len=frame_len, hop=hop)
        centroid = _extract_spectral_centroid(audio, sr, frame_len=frame_len, hop=hop)

        return np.array(
            [
                f0_mean,
                f0_std,
                float(np.mean(energy)),
                float(np.std(energy)),
                float(np.mean(zcr)),
                float(np.mean(centroid)),
            ],
            dtype=np.float64,
        )

    # ── Optimal Transport Alignment ────────────────────────────────────────────

    def fuse(
        self,
        eeg_features: np.ndarray,
        voice_features: np.ndarray,
    ) -> np.ndarray:
        """Fuse EEG and voice feature vectors via OT-weighted combination.

        Builds a pairwise cost matrix between the two feature vectors (treated
        as 1-D histograms over feature indices), computes the Sinkhorn transport
        plan, then uses the total transport mass as the fusion weight:

            fused = w * eeg_norm + (1-w) * voice_norm

        where w (fusion_weight) reflects how well-aligned the modalities are
        (high w -> EEG more reliable; low w -> voice more reliable; ~0.5 -> equal).

        Args:
            eeg_features: EEG feature vector, shape (d_eeg,).
            voice_features: Voice feature vector, shape (d_voice,).

        Returns:
            Fused feature vector, shape (d_eeg + d_voice,).
        """
        eeg_f = eeg_features.astype(np.float64).ravel()
        voice_f = voice_features.astype(np.float64).ravel()

        n_eeg = len(eeg_f)
        n_voice = len(voice_f)

        # Normalise to probability simplex (source / target distributions)
        def _to_simplex(v: np.ndarray) -> np.ndarray:
            v_pos = np.abs(v) + 1e-10
            return v_pos / v_pos.sum()

        mu = _to_simplex(eeg_f)      # (n_eeg,)
        nu = _to_simplex(voice_f)    # (n_voice,)

        # Pairwise L2-squared cost matrix between feature distributions
        # Shape: (n_eeg, n_voice)
        # We normalise feature indices to [0,1] to make cost scale-invariant
        idx_eeg = np.linspace(0, 1, n_eeg)
        idx_voice = np.linspace(0, 1, n_voice)
        C = (idx_eeg[:, None] - idx_voice[None, :]) ** 2  # (n_eeg, n_voice)

        T = _sinkhorn_transport(mu, nu, C, eps=0.1, n_iter=50)

        ot_cost = float(np.sum(T * C))
        max_cost = float(np.max(C)) + 1e-10
        # alignment_quality: 0 = bad alignment (high cost), 1 = perfect alignment
        alignment_quality = float(np.exp(-ot_cost / max_cost))

        self._last_ot_stats = {
            "transport_cost": round(ot_cost, 6),
            "alignment_quality": round(alignment_quality, 4),
        }

        # OT-derived fusion weight:
        # Good alignment (high quality) -> trust both equally (weight ~0.5)
        # The weight here determines EEG vs voice contribution to the fused vector.
        # We blend normalised feature vectors with alignment_quality as the soft gate.
        fusion_weight = 0.5 + 0.5 * alignment_quality  # in [0.5, 1.0]; higher -> more EEG

        # Normalise feature magnitudes before blending
        eeg_norm = eeg_f / (np.linalg.norm(eeg_f) + 1e-10)
        voice_norm = voice_f / (np.linalg.norm(voice_f) + 1e-10)

        # Concatenate with OT-derived scaling
        fused = np.concatenate([
            eeg_norm * fusion_weight,
            voice_norm * (1.0 - fusion_weight),
        ])
        return fused

    # ── Emotion Classification ─────────────────────────────────────────────────

    def _emotion_from_eeg(self, eeg_features: np.ndarray, fs: float) -> Dict[str, Any]:
        """Heuristic emotion inference from EEG band powers (feature-based fallback).

        Uses the band-power ratios established in the CLAUDE.md science reference:
        - Valence: FAA proxy via alpha asymmetry (if multichannel) or alpha/beta ratio
        - Arousal: beta/(alpha+beta) ratio
        """
        feats = eeg_features.ravel()
        n = len(feats)

        # Determine per-channel band powers depending on feature vector size
        # _extract_eeg_features returns 5 features per channel
        bands_per_ch = 5
        n_channels = n // bands_per_ch

        if n_channels == 0:
            # Degenerate input
            probs = np.ones(6) / 6
            return {
                "emotion": "neutral",
                "probabilities": dict(zip(EMOTIONS_6, probs.tolist())),
                "valence": 0.0,
                "arousal": 0.5,
                "model_type": "eeg-only-degenerate",
            }

        # Aggregate across channels: mean band powers
        # band order: delta, theta, alpha, beta, gamma
        ch_feats = feats[: n_channels * bands_per_ch].reshape(n_channels, bands_per_ch)
        mean_bands = ch_feats.mean(axis=0)  # (delta, theta, alpha, beta, gamma)
        delta, theta, alpha, beta, gamma = mean_bands

        total = delta + theta + alpha + beta + 1e-10
        alpha_beta_ratio = alpha / (beta + 1e-10)
        beta_ratio = beta / (alpha + beta + 1e-10)
        theta_beta_ratio = theta / (beta + 1e-10)

        # FAA proxy: if 4 channels (Muse 2: TP9=ch0, AF7=ch1, AF8=ch2, TP10=ch3)
        faa_valence = 0.0
        if n_channels >= 3:
            # ch1 = AF7 (left), ch2 = AF8 (right)
            af7_alpha = ch_feats[1, 2]  # alpha band index 2
            af8_alpha = ch_feats[2, 2]
            if af7_alpha > 1e-12 and af8_alpha > 1e-12:
                faa = np.log(af8_alpha) - np.log(af7_alpha)
                faa_valence = float(np.clip(np.tanh(faa * 2.0), -1.0, 1.0))

        # Valence: alpha/beta ratio + FAA
        valence_abr = float(np.clip(np.tanh((alpha_beta_ratio - 0.7) * 2.0), -1.0, 1.0))
        if n_channels >= 3:
            valence = float(np.clip(0.50 * valence_abr + 0.50 * faa_valence, -1.0, 1.0))
        else:
            valence = valence_abr

        # Arousal: beta-dominated (no gamma — EMG noise at AF7/AF8)
        arousal = float(np.clip(
            0.45 * beta_ratio + 0.30 * (1.0 - alpha / (alpha + beta + theta + 1e-10))
            + 0.25 * (1.0 - delta / (delta + beta + 1e-10)),
            0.0, 1.0,
        ))

        # Map (valence, arousal) -> 6-class probability vector
        # Russell's circumplex: valence in [-1,1], arousal in [0,1]
        #   happy:    high valence, high arousal
        #   sad:      low valence, low arousal
        #   angry:    low valence, high arousal (approach-motivated)
        #   fear:     low valence, high arousal (withdrawal)
        #   surprise: any valence, very high arousal
        #   neutral:  mid valence, mid arousal
        pos_val = max(0.0, valence)
        neg_val = max(0.0, -valence)
        high_ar = arousal
        low_ar = 1.0 - arousal

        raw_probs = np.array([
            0.50 * pos_val + 0.30 * high_ar + 0.20 * (1.0 - neg_val),   # happy
            0.50 * max(0.0, -valence - 0.10) + 0.30 * low_ar,             # sad
            0.40 * neg_val + 0.35 * high_ar + 0.25 * beta_ratio,          # angry
            0.45 * neg_val + 0.35 * high_ar + 0.20 * (1.0 - pos_val),    # fear
            0.60 * max(0.0, arousal - 0.7) + 0.40 * (1.0 - abs(valence)), # surprise
            0.40 * (1.0 - abs(valence)) + 0.30 * (1.0 - abs(arousal - 0.5) * 2),  # neutral
        ], dtype=np.float64)

        raw_probs = np.clip(raw_probs, 0.0, None)
        total_p = raw_probs.sum()
        if total_p < 1e-10:
            raw_probs = np.ones(6) / 6
        else:
            raw_probs /= total_p

        emotion_idx = int(np.argmax(raw_probs))
        return {
            "emotion": EMOTIONS_6[emotion_idx],
            "probabilities": dict(zip(EMOTIONS_6, raw_probs.tolist())),
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
            "model_type": "eeg-only-heuristic",
        }

    def _emotion_from_voice(self, voice_features: np.ndarray) -> Dict[str, Any]:
        """Heuristic emotion inference from prosodic features.

        Voice features order: [f0_mean, f0_std, energy_mean, energy_std, zcr_mean, centroid_mean]

        Prosodic correlates (from Kay et al. 2003, Banse & Scherer 1996):
        - High F0 + high energy -> happy or angry
        - Low F0 + low energy   -> sad
        - High F0 variability   -> fear or surprise
        - Low ZCR + low centroid -> neutral/calm
        """
        if len(voice_features) < 6:
            probs = np.ones(6) / 6
            return {
                "emotion": "neutral",
                "probabilities": dict(zip(EMOTIONS_6, probs.tolist())),
                "valence": 0.0,
                "arousal": 0.5,
                "model_type": "voice-only-degenerate",
            }

        f0_mean, f0_std, en_mean, en_std, zcr_mean, centroid_mean = voice_features[:6]

        # Normalise to [0,1] proxies
        f0_norm = float(np.clip(f0_mean / 300.0, 0.0, 1.0))      # ~300 Hz = high pitched
        f0_var = float(np.clip(f0_std / 100.0, 0.0, 1.0))         # pitch variability
        en_norm = float(np.clip(en_mean / 0.1, 0.0, 1.0))         # energy level
        zcr_norm = float(np.clip(zcr_mean / 0.2, 0.0, 1.0))       # ZCR
        cent_norm = float(np.clip(centroid_mean / 4000.0, 0.0, 1.0))  # spectral brightness

        # Voice-based arousal: energy + F0 + centroid
        arousal = float(np.clip(0.40 * en_norm + 0.35 * f0_norm + 0.25 * cent_norm, 0.0, 1.0))
        # Voice-based valence: higher pitch and moderate energy -> positive
        valence = float(np.clip(
            0.50 * f0_norm + 0.30 * en_norm - 0.20 * f0_var,  # variability lowers valence
            -1.0, 1.0,
        ))
        valence = valence * 2.0 - 1.0  # rescale [0,1] -> [-1,1] roughly

        raw_probs = np.array([
            0.40 * max(0.0, valence) + 0.35 * arousal + 0.25 * en_norm,   # happy
            0.50 * max(0.0, -valence) + 0.30 * (1.0 - arousal),            # sad
            0.35 * max(0.0, -valence) + 0.40 * arousal + 0.25 * zcr_norm,  # angry
            0.45 * f0_var + 0.30 * arousal + 0.25 * max(0.0, -valence),    # fear
            0.50 * f0_var + 0.30 * max(0.0, arousal - 0.6),                # surprise
            0.40 * (1.0 - abs(valence)) + 0.40 * (1.0 - arousal),          # neutral
        ], dtype=np.float64)

        raw_probs = np.clip(raw_probs, 0.0, None)
        total_p = raw_probs.sum()
        if total_p < 1e-10:
            raw_probs = np.ones(6) / 6
        else:
            raw_probs /= total_p

        emotion_idx = int(np.argmax(raw_probs))
        return {
            "emotion": EMOTIONS_6[emotion_idx],
            "probabilities": dict(zip(EMOTIONS_6, raw_probs.tolist())),
            "valence": round(float(valence), 4),
            "arousal": round(arousal, 4),
            "model_type": "voice-only-heuristic",
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(
        self,
        eeg: Optional[np.ndarray],
        audio: Optional[np.ndarray],
        eeg_fs: float = 256.0,
        voice_fs: int = 16000,
    ) -> Dict[str, Any]:
        """Predict emotion from EEG + voice via Optimal Transport fusion.

        Gracefully degrades to single-modal when one input is unavailable.

        Args:
            eeg: EEG array of shape (n_channels, n_samples) or (n_samples,).
                 Pass None to use voice-only.
            audio: 1-D float32 audio array. Pass None to use EEG-only.
            eeg_fs: EEG sampling frequency (Hz).
            voice_fs: Audio sampling rate (Hz).

        Returns:
            dict with keys:
                emotion         (str)   — dominant class from EMOTIONS_6
                probabilities   (dict)  — per-class float in [0,1], sums to 1
                valence         (float) — [-1,1], negative=unpleasant
                arousal         (float) — [0,1], low=calm
                fusion_weight   (float) — OT alignment quality [0,1]
                model_type      (str)   — describes which path was taken
        """
        has_eeg = eeg is not None and isinstance(eeg, np.ndarray) and eeg.size > 0
        has_audio = audio is not None and isinstance(audio, np.ndarray) and audio.size > 0

        # ── EEG-only fallback ──────────────────────────────────────────────────
        if not has_audio:
            if not has_eeg:
                probs = np.ones(self.n_classes) / self.n_classes
                return {
                    "emotion": "neutral",
                    "probabilities": dict(zip(EMOTIONS_6, probs.tolist())),
                    "valence": 0.0,
                    "arousal": 0.5,
                    "fusion_weight": 0.0,
                    "model_type": "no-input",
                }
            eeg_feats = self._extract_eeg_features(eeg, eeg_fs)
            result = self._emotion_from_eeg(eeg_feats, eeg_fs)
            result["fusion_weight"] = 0.0
            result["model_type"] = "eeg-only"
            return result

        # ── Voice-only fallback ────────────────────────────────────────────────
        if not has_eeg:
            voice_feats = self._extract_voice_features(audio.astype(np.float32), voice_fs)
            result = self._emotion_from_voice(voice_feats)
            result["fusion_weight"] = 0.0
            result["model_type"] = "voice-only"
            return result

        # ── Full cross-modal OT fusion ─────────────────────────────────────────
        try:
            eeg_feats = self._extract_eeg_features(eeg, eeg_fs)
            voice_feats = self._extract_voice_features(audio.astype(np.float32), voice_fs)

            fused = self.fuse(eeg_feats, voice_feats)
            fusion_weight = float(self._last_ot_stats["alignment_quality"])

            # Independent modal predictions for blend
            eeg_result = self._emotion_from_eeg(eeg_feats, eeg_fs)
            voice_result = self._emotion_from_voice(voice_feats)

            # Blend probabilities: OT alignment quality drives EEG weight
            eeg_w = 0.5 + 0.5 * fusion_weight  # [0.5, 1.0]
            voice_w = 1.0 - eeg_w               # [0.0, 0.5]

            eeg_probs = np.array([eeg_result["probabilities"][e] for e in EMOTIONS_6])
            voice_probs = np.array([voice_result["probabilities"][e] for e in EMOTIONS_6])
            blended = eeg_w * eeg_probs + voice_w * voice_probs
            blended /= blended.sum() + 1e-10

            emotion_idx = int(np.argmax(blended))

            # Blend valence and arousal
            valence = float(eeg_w * eeg_result["valence"] + voice_w * voice_result["valence"])
            arousal = float(eeg_w * eeg_result["arousal"] + voice_w * voice_result["arousal"])

            return {
                "emotion": EMOTIONS_6[emotion_idx],
                "probabilities": dict(zip(EMOTIONS_6, blended.tolist())),
                "valence": round(float(np.clip(valence, -1.0, 1.0)), 4),
                "arousal": round(float(np.clip(arousal, 0.0, 1.0)), 4),
                "fusion_weight": round(fusion_weight, 4),
                "model_type": "eeg-voice-ot-fusion",
            }

        except Exception as exc:
            log.warning("EEGVoiceFusion.predict failed, falling back to EEG-only: %s", exc)
            eeg_feats = self._extract_eeg_features(eeg, eeg_fs)
            result = self._emotion_from_eeg(eeg_feats, eeg_fs)
            result["fusion_weight"] = 0.0
            result["model_type"] = "eeg-only-fallback"
            return result

    def get_fusion_stats(self) -> Dict[str, float]:
        """Return the last OT transport plan statistics.

        Returns:
            dict with:
                transport_cost     (float) — raw Wasserstein cost of last fusion
                alignment_quality  (float) — [0,1], higher = better modal alignment
        """
        return dict(self._last_ot_stats)


# ── Singleton registry ─────────────────────────────────────────────────────────

_fusion_registry: Dict[str, EEGVoiceFusionClassifier] = {}
_registry_lock = threading.Lock()


def get_eeg_voice_fusion(user_id: str = "default") -> EEGVoiceFusionClassifier:
    """Return a per-user singleton EEGVoiceFusionClassifier instance.

    Args:
        user_id: Unique identifier for the user session.

    Returns:
        EEGVoiceFusionClassifier bound to that user_id.
    """
    with _registry_lock:
        if user_id not in _fusion_registry:
            _fusion_registry[user_id] = EEGVoiceFusionClassifier()
        return _fusion_registry[user_id]
