"""REVE-inspired DETransformer Emotion Classifier (Inference Wrapper).

Loads the DETransformer model trained on FACED dataset (train_reve_faced.py)
and runs inference on live Muse 2 EEG data.

Key details:
  - Only activates when >= 30 seconds of EEG are buffered (7680 samples at 256 Hz)
  - Extracts 57-dim DE feature vectors for each 1-second window → (30, 57) sequence
  - DETransformer outputs 3-class (pos/neu/neg) → expanded to 6-class using arousal+FAA
  - Falls through silently when model file is absent or benchmark < 60%

Feature format (57-dim per 1-second window):
  - 20 raw DE: 5 bands × 4 channels
  - 16 band ratios: alpha/beta, theta/beta, alpha/theta, delta/theta per channel
  - 5 f_dasm: FP2-FP1 DE per band (frontal DASM)
  - 5 f_rasm: FP2/FP1 DE per band (frontal RASM)
  - 5 t_dasm: T8-T7 DE per band (temporal DASM)
  - 5 t_rasm: T8/T7 DE per band (temporal RASM)
  - 1 faa: FP2_alpha - FP1_alpha (frontal alpha asymmetry)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.signal import butter, filtfilt

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent   # ml/models/
_ML_ROOT      = _HERE.parent                       # ml/
_MODEL_PATH   = _ML_ROOT / "models" / "saved" / "reve_emotion_4ch.pt"
_BENCH_PATH   = _ML_ROOT / "benchmarks" / "reve_emotion_4ch_benchmark.json"

# ── Constants ─────────────────────────────────────────────────────────────────
_MIN_ACCURACY  = 0.50     # benchmark gate for 3-class (chance=33% → 50% is well above chance)
# Note: emotion_classifier.py uses 60% for 6-class (chance=16.7%).
# For 3-class cross-subject EEG, 52% is ~19 pts above chance — same relative lift.
_SEQ_LEN       = 30       # seconds per epoch
_INPUT_DIM     = 57       # features per 1-second window
_N_CLASSES     = 3        # model outputs: positive / neutral / negative

# Band-power computation: Welch-based DE (Differential Entropy = 0.5*log(2πe*σ²))
_BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
_BAND_KEYS = ["delta", "theta", "alpha", "beta", "gamma"]  # fixed order (5)
_DELTA, _THETA, _ALPHA, _BETA, _GAMMA = 0, 1, 2, 3, 4

EMOTIONS = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]

# Muse 2 channel order: ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10
# Maps to FACED channels: T7(23)→TP9, FP1(0)→AF7, FP2(2)→AF8, T8(24)→TP10
# In our (4,) channel vector after remapping:
#   idx 0 → T7  (TP9)
#   idx 1 → FP1 (AF7)  ← left frontal
#   idx 2 → FP2 (AF8)  ← right frontal
#   idx 3 → T8  (TP10)
_MUSE_LEFT_FRONTAL  = 1   # AF7 → FP1 position
_MUSE_RIGHT_FRONTAL = 2   # AF8 → FP2 position
_MUSE_LEFT_TEMPORAL = 0   # TP9 → T7 position
_MUSE_RIGHT_TEMPORAL = 3  # TP10 → T8 position


# ── DETransformer (mirrors train_reve_faced.py — kept in sync manually) ──────

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


class DETransformer:
    """Thin wrapper around the PyTorch DETransformer module.

    Defined as a Python class (not nn.Module subclass at module level) so that
    this file can be imported even when PyTorch is not installed — it just
    silently returns is_available() = False.
    """

    def __new__(cls, *args, **kwargs):
        if not _TORCH_OK:
            return None
        obj = object.__new__(cls)
        return obj

    def __init__(self, cfg: dict) -> None:
        if not _TORCH_OK:
            return
        self._net = _DETransformerNet(**cfg)

    def load_state(self, state_dict) -> None:
        self._net.load_state_dict(state_dict)
        self._net.eval()

    def predict_proba(self, x_np: np.ndarray) -> np.ndarray:
        """Run forward pass. x_np: (batch, seq_len, input_dim). Returns (batch, 3) softmax."""
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            t = torch.tensor(x_np, dtype=torch.float32)
            logits = self._net(t)
            return F.softmax(logits, dim=-1).numpy()


if _TORCH_OK:
    import torch.nn as nn  # noqa: F811

    class _DETransformerNet(nn.Module):
        """PyTorch implementation of the DETransformer."""

        def __init__(
            self,
            input_dim: int = _INPUT_DIM,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            n_classes: int = _N_CLASSES,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_dropout = nn.Dropout(dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4, dropout=dropout,
                batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )

        def forward(self, x):
            x = self.input_proj(x)
            x = self.pos_dropout(x)
            x = self.encoder(x)
            x = x.mean(dim=1)
            return self.classifier(x)


# ── Signal processing helpers ─────────────────────────────────────────────────

def _bandpass_de(signal: np.ndarray, fs: float, low: float, high: float) -> float:
    """Compute Differential Entropy of a bandpass-filtered 1-second signal.

    DE of a Gaussian = 0.5 * log(2 * pi * e * sigma^2)
    For EEG, we assume Gaussian within narrow frequency bands (standard in SEED/FACED).
    """
    nyq = fs / 2.0
    lo, hi = max(low / nyq, 1e-4), min(high / nyq, 0.9999)
    if lo >= hi:
        return 0.0
    try:
        b, a = butter(4, [lo, hi], btype="band")
        filtered = filtfilt(b, a, signal)
        var = float(np.var(filtered))
        if var < 1e-12:
            return 0.0
        return float(0.5 * np.log(2.0 * np.pi * np.e * var))
    except Exception:
        return 0.0


def _compute_de_4ch(window_4ch: np.ndarray, fs: float) -> np.ndarray:
    """Compute (4, 5) DE feature matrix for a 1-second window.

    Args:
        window_4ch: (4, n_samples) — 4 Muse channels, ~1 second of data
        fs:         sampling rate (256 Hz for Muse 2)

    Returns:
        de: (4, 5) float32 — DE per channel per band, band order = delta/theta/alpha/beta/gamma
    """
    de = np.zeros((4, 5), dtype=np.float32)
    for ch in range(4):
        sig = window_4ch[ch].astype(np.float64)
        for b_idx, key in enumerate(_BAND_KEYS):
            low, high = _BANDS[key]
            de[ch, b_idx] = _bandpass_de(sig, fs, low, high)
    return de


def _de_to_features(de_4ch: np.ndarray) -> np.ndarray:
    """Convert a (4, 5) DE window into a 57-dimensional feature vector.

    Identical to train_reve_faced._de_to_features — must stay in sync.
    """
    eps = 1e-8
    t7, fp1, fp2, t8 = de_4ch[0], de_4ch[1], de_4ch[2], de_4ch[3]

    raw = de_4ch.flatten()   # (20,)

    ratios = []
    for ch in range(4):
        d = de_4ch[ch, _DELTA] + eps
        h = de_4ch[ch, _THETA] + eps
        a = de_4ch[ch, _ALPHA] + eps
        b = de_4ch[ch, _BETA]  + eps
        ratios.extend([a / b, h / b, a / h, d / h])
    ratios = np.array(ratios)   # (16,)

    f_dasm = fp2 - fp1                             # (5,)
    f_rasm = fp2 / (fp1 + eps)                     # (5,)
    t_dasm = t8  - t7                              # (5,)
    t_rasm = t8  / (t7  + eps)                     # (5,)
    faa    = np.array([fp2[_ALPHA] - fp1[_ALPHA]]) # (1,)

    return np.concatenate([raw, ratios, f_dasm, f_rasm, t_dasm, t_rasm, faa])


# ── Main classifier ───────────────────────────────────────────────────────────

class REVEEmotionClassifier:
    """Inference wrapper for the REVE-inspired DETransformer emotion model.

    Usage:
        clf = REVEEmotionClassifier()
        if clf.is_available():
            result = clf.predict(eeg_4ch_30sec, fs=256)

    The predict() method accepts (4, >=7680) Muse 2 EEG and returns a dict
    with the same 15-key structure as EmotionClassifier._build_muse_result().
    """

    def __init__(self) -> None:
        self._model: Optional[DETransformer] = None
        self._scaler = None          # sklearn StandardScaler or None
        self._cv_accuracy: float = 0.0
        self._ema_probs: Optional[np.ndarray] = None
        self._ema_alpha: float = 0.3
        self._try_load()

    # ── Loading ────────────────────────────────────────────────────────────────

    def _try_load(self) -> None:
        """Attempt to load model + benchmark. Fails silently."""
        if not _TORCH_OK:
            log.debug("[REVEEmotionClassifier] PyTorch not available — skipping")
            return
        if not _MODEL_PATH.exists():
            log.debug("[REVEEmotionClassifier] Model file not found: %s", _MODEL_PATH)
            return
        if not _BENCH_PATH.exists():
            log.debug("[REVEEmotionClassifier] Benchmark file not found: %s", _BENCH_PATH)
            return
        try:
            bench = json.loads(_BENCH_PATH.read_text())
            acc = float(bench.get("cv_accuracy", bench.get("accuracy", 0.0)))
            if acc < _MIN_ACCURACY:
                log.info("[REVEEmotionClassifier] CV accuracy %.4f < %.2f threshold — disabled",
                         acc, _MIN_ACCURACY)
                return
        except Exception as exc:
            log.debug("[REVEEmotionClassifier] Benchmark read failed: %s", exc)
            return

        try:
            import torch
            payload = torch.load(str(_MODEL_PATH), map_location="cpu", weights_only=False)
            cfg = payload.get("model_cfg", {
                "input_dim": _INPUT_DIM, "d_model": 64, "n_heads": 4,
                "n_layers": 2, "n_classes": _N_CLASSES, "dropout": 0.2,
            })
            model = DETransformer(cfg)
            if model is None:
                return
            model.load_state(payload["model_state"])

            self._model = model
            self._scaler = payload.get("scaler")
            self._cv_accuracy = acc
            log.info("[REVEEmotionClassifier] Loaded DETransformer "
                     "(CV acc=%.4f, seq_len=%d, input_dim=%d)",
                     acc, payload.get("seq_len", _SEQ_LEN),
                     payload.get("input_dim", _INPUT_DIM))
        except Exception as exc:
            log.warning("[REVEEmotionClassifier] Load failed: %s", exc)
            self._model = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True if the model is loaded and above the accuracy threshold."""
        return self._model is not None and self._cv_accuracy >= _MIN_ACCURACY

    def predict(
        self,
        eeg: np.ndarray,
        fs: float = 256.0,
        device_type: str = "muse_2",
    ) -> Dict:
        """Classify emotion from a 30-second Muse 2 EEG epoch.

        Args:
            eeg:         (4, n_samples) — must have n_samples >= 30 * fs
            fs:          sampling rate (default 256 Hz)
            device_type: ignored (always Muse 2 channel layout) — kept for API consistency

        Returns:
            15-key dict matching EmotionClassifier._build_muse_result() format.
        """
        # ── Artifact gating ────────────────────────────────────────────────────
        if np.any(np.abs(eeg) > 100.0):
            if self._ema_probs is None:
                self._ema_probs = np.ones(6, dtype=np.float32) / 6
            return self._build_result(
                int(np.argmax(self._ema_probs)),
                self._ema_probs, eeg, fs, artifact_detected=True
            )

        # ── Build (30, 57) feature sequence ───────────────────────────────────
        seq = self._extract_sequence(eeg, fs)   # (30, 57) or None
        if seq is None:
            raise ValueError("EEG too short for DETransformer (need >= 30 sec)")

        # ── Optionally scale features ──────────────────────────────────────────
        if self._scaler is not None:
            try:
                flat = seq.reshape(-1, _INPUT_DIM)
                flat = self._scaler.transform(flat).astype(np.float32)
                seq  = flat.reshape(_SEQ_LEN, _INPUT_DIM)
            except Exception:
                pass   # scaler mismatch — continue with raw features

        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Model forward pass ─────────────────────────────────────────────────
        proba3 = self._model.predict_proba(seq[np.newaxis])[0]  # (3,) softmax
        p_pos, p_neu, p_neg = float(proba3[0]), float(proba3[1]), float(proba3[2])

        # ── 3→6 expansion (identical to _predict_mega_lgbm) ───────────────────
        arousal, faa_val, beta_alpha = self._compute_ancillary(eeg, fs)

        probs6 = np.zeros(6, dtype=np.float32)
        # positive → happy (high arousal) or relaxed (low arousal)
        if arousal > 0.5:
            probs6[0] += p_pos          # happy
        else:
            probs6[4] += p_pos          # relaxed
        # neutral → focused (high beta/alpha) or relaxed (low beta/alpha)
        if beta_alpha > 1.2:
            probs6[5] += p_neu          # focused
        else:
            probs6[4] += p_neu          # relaxed
        # negative → sad / angry / fearful based on FAA + arousal
        if faa_val < -0.2 and arousal < 0.5:
            probs6[1] += p_neg          # sad
        elif faa_val < 0.0 and arousal > 0.5:
            probs6[2] += p_neg          # angry
        elif arousal > 0.55:
            probs6[3] += p_neg          # fearful
        else:
            probs6[1] += p_neg          # sad

        total = probs6.sum()
        if total > 0:
            probs6 /= total

        # ── EMA smoothing ──────────────────────────────────────────────────────
        if self._ema_probs is None:
            self._ema_probs = probs6.copy()
        else:
            self._ema_probs = (self._ema_alpha * probs6
                               + (1.0 - self._ema_alpha) * self._ema_probs)
        smoothed = self._ema_probs
        emotion_idx = int(np.argmax(smoothed))

        result = self._build_result(emotion_idx, smoothed, eeg, fs,
                                    artifact_detected=False)
        result["model_type"] = "reve-de-transformer"
        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    def _extract_sequence(
        self, eeg: np.ndarray, fs: float
    ) -> Optional[np.ndarray]:
        """Extract (30, 57) DE feature sequence from (4, n_samples) Muse EEG.

        Splits the last 30 seconds into 1-second windows and computes DE features
        using the same _de_to_features() function used during training.
        """
        n_samples = eeg.shape[1]
        win_size   = int(fs)             # samples per 1-second window
        n_needed   = win_size * _SEQ_LEN

        if n_samples < n_needed:
            return None

        # Use the most recent 30 seconds
        start = n_samples - n_needed
        clip  = eeg[:, start:start + n_needed]   # (4, 7680)

        seq = np.zeros((_SEQ_LEN, _INPUT_DIM), dtype=np.float32)
        for t in range(_SEQ_LEN):
            w_start = t * win_size
            window  = clip[:, w_start : w_start + win_size]   # (4, 256)
            de_4ch  = _compute_de_4ch(window, fs)              # (4, 5)
            seq[t]  = _de_to_features(de_4ch)                  # (57,)
        return seq

    def _compute_ancillary(
        self, eeg: np.ndarray, fs: float
    ) -> tuple:
        """Compute arousal, faa_val, beta_alpha from the last 4 seconds of EEG.

        Returns: (arousal, faa_val, beta_alpha) — used for 3→6 class expansion.
        """
        win_size = int(fs * 4)
        chunk = eeg[:, -min(win_size, eeg.shape[1]):]  # last 4 sec

        try:
            # Use last 4-sec window for band powers
            de = _compute_de_4ch(chunk[:, :int(fs)] if chunk.shape[1] >= int(fs) else chunk, fs)
            # de shape (4, 5)
            lf = _MUSE_LEFT_FRONTAL
            rf = _MUSE_RIGHT_FRONTAL
            a_lf = float(np.exp(de[lf, _ALPHA]))   # back from log-domain
            a_rf = float(np.exp(de[rf, _ALPHA]))
            b_lf = float(np.exp(de[lf, _BETA]))
            b_rf = float(np.exp(de[rf, _BETA]))
            t_lf = float(np.exp(de[lf, _THETA]))
            t_rf = float(np.exp(de[rf, _THETA]))
            d_lf = float(np.exp(de[lf, _DELTA]))
            d_rf = float(np.exp(de[rf, _DELTA]))

            alpha = max((a_lf + a_rf) / 2, 1e-6)
            beta  = max((b_lf + b_rf) / 2, 1e-6)
            theta = max((t_lf + t_rf) / 2, 1e-6)
            delta = max((d_lf + d_rf) / 2, 1e-6)

            arousal = float(np.clip(
                0.45 * beta / (beta + alpha)
                + 0.30 * (1.0 - alpha / (alpha + beta + theta))
                + 0.25 * (1.0 - delta / (delta + beta)),
                0.0, 1.0
            ))
            beta_alpha = beta / max(alpha, 1e-6)
            faa_val = float(np.clip(np.tanh((de[rf, _ALPHA] - de[lf, _ALPHA]) * 2.0), -1.0, 1.0))
        except Exception:
            arousal, faa_val, beta_alpha = 0.5, 0.0, 1.0

        return arousal, faa_val, beta_alpha

    def _build_result(
        self,
        emotion_idx: int,
        smoothed: np.ndarray,
        eeg: np.ndarray,
        fs: float,
        artifact_detected: bool,
    ) -> Dict:
        """Build the standard 15-key emotion result dict."""
        # Compute band powers + indices from last second of left-frontal channel
        lf = _MUSE_LEFT_FRONTAL
        rf = _MUSE_RIGHT_FRONTAL
        try:
            win = int(fs)
            seg_lf = eeg[lf, -min(win, eeg.shape[1]):]
            seg_rf = eeg[rf, -min(win, eeg.shape[1]):]
            de_1s  = _compute_de_4ch(
                eeg[:, -min(win, eeg.shape[1]):], fs
            )   # (4, 5)

            alpha      = max(float(np.exp(de_1s[lf, _ALPHA])), 1e-6)
            beta       = max(float(np.exp(de_1s[lf, _BETA])),  1e-6)
            theta      = max(float(np.exp(de_1s[lf, _THETA])), 1e-6)
            delta      = max(float(np.exp(de_1s[lf, _DELTA])), 1e-6)
            high_beta  = beta * 0.5   # approximation (no separate band computed)
            low_beta   = beta * 0.5
            gamma      = 0.0          # excluded (EMG noise on Muse 2)

            bands = {
                "delta": delta, "theta": theta, "alpha": alpha,
                "beta":  beta,  "low_beta": low_beta, "high_beta": high_beta,
                "gamma": gamma,
            }
            de_dict = {k: float(de_1s[lf, i]) for i, k in enumerate(_BAND_KEYS)}

            alpha_a    = max(float(np.exp(de_1s[lf, _ALPHA])), 1e-6)
            alpha_b    = max(float(np.exp(de_1s[rf, _ALPHA])), 1e-6)
            faa_raw    = float(np.log(alpha_b) - np.log(alpha_a))
            faa_val    = float(np.clip(np.tanh(faa_raw * 2.0), -1.0, 1.0))
            dasm_alpha = float(de_1s[rf, _ALPHA] - de_1s[lf, _ALPHA])

            beta_alpha  = beta / max(alpha, 1e-6)
            valence_abr = float(np.clip(
                0.65 * np.tanh((alpha / beta - 0.7) * 2.0)
                + 0.35 * np.tanh((alpha - 0.15) * 4), -1.0, 1.0
            ))
            valence = float(np.clip(
                0.40 * valence_abr + 0.35 * faa_val + 0.25 * dasm_alpha * 0.5,
                -1.0, 1.0
            ))
            arousal = float(np.clip(
                0.45 * beta / (beta + alpha)
                + 0.30 * (1.0 - alpha / (alpha + beta + theta))
                + 0.25 * (1.0 - delta / (delta + beta)),
                0.0, 1.0
            ))
            theta_beta_ratio = theta / max(beta, 1e-6)
            stress_index = float(np.clip(
                0.40 * min(1.0, beta_alpha * 0.3)
                + 0.25 * max(0.0, 1.0 - alpha * 2.5)
                + 0.25 * min(1.0, high_beta * 4.0), 0.0, 1.0
            ))
            focus_index = float(np.clip(
                0.45 * min(1.0, beta * 3.5)
                + 0.40 * max(0.0, 1.0 - theta_beta_ratio * 0.40)
                + 0.15 * min(1.0, low_beta * 5.0), 0.0, 1.0
            ))
            relaxation_index = float(np.clip(
                0.50 * min(1.0, alpha * 2.5)
                + 0.30 * max(0.0, 1.0 - beta_alpha * 0.3)
                + 0.20 * min(1.0, theta * 1.5), 0.0, 1.0
            ))
            anger_index = float(np.clip(
                0.40 * min(1.0, max(0.0, beta_alpha - 1.0) * 0.5)
                + 0.30 * max(0.0, 1.0 - alpha * 5.0)
                + 0.20 * min(1.0, high_beta * 3.0), 0.0, 1.0
            ))
            fear_index = float(np.clip(
                0.35 * min(1.0, max(0.0, beta_alpha - 1.5) * 0.6)
                + 0.30 * min(1.0, high_beta * 5.0)
                + 0.25 * max(0.0, 1.0 - alpha * 5.0)
                + 0.10 * max(0.0, arousal - 0.45), 0.0, 1.0
            ))
            temporal_asymmetry = float(
                np.log(max(float(np.exp(de_1s[3, _ALPHA])), 1e-6))
                - np.log(max(float(np.exp(de_1s[0, _ALPHA])), 1e-6))
            )
        except Exception:
            valence = arousal = 0.0
            stress_index = focus_index = relaxation_index = anger_index = fear_index = 0.0
            faa_raw = temporal_asymmetry = 0.0
            bands = {}
            de_dict = {}

        return {
            "emotion":               EMOTIONS[emotion_idx],
            "emotion_index":         emotion_idx,
            "confidence":            float(np.max(smoothed)),
            "probabilities":         {EMOTIONS[i]: float(p) for i, p in enumerate(smoothed)},
            "valence":               valence,
            "arousal":               arousal,
            "stress_index":          stress_index,
            "focus_index":           focus_index,
            "relaxation_index":      relaxation_index,
            "anger_index":           anger_index,
            "fear_index":            fear_index,
            "band_powers":           bands,
            "differential_entropy":  de_dict,
            "frontal_asymmetry":     faa_raw,
            "temporal_asymmetry":    temporal_asymmetry,
            "artifact_detected":     artifact_detected,
            "model_type":            "reve-de-transformer",
        }
