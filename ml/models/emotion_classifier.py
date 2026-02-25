"""CNN-LSTM Emotion Classifier from EEG signals.

Classifies EEG into 6 emotions: happy, sad, angry, fearful, relaxed, focused.
Also outputs valence (-1 to 1) and arousal (0 to 1) scores.

Uses neuroscience-backed feature-based classification with exponential
smoothing to prevent rapid state flickering. ONNX/sklearn models are
only used when their benchmark accuracy exceeds 60%.
"""

import numpy as np
from typing import Dict, Optional
from collections import deque
from processing.eeg_processor import (
    extract_band_powers, differential_entropy, extract_features, preprocess,
    compute_frontal_asymmetry, compute_dasm_rasm,
    compute_frontal_midline_theta,
)

# Amplitude threshold above which an epoch is flagged as artifact-contaminated.
# From Krigolson (2021): 75 µV catches most blinks (100-200 µV) and EMG bursts.
_ARTIFACT_THRESHOLD_UV = 75.0

EMOTIONS = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]

# Minimum benchmark accuracy to trust a trained model over feature-based.
# 60% threshold: the DEAP model at ~51% has severe class-imbalance bias toward "sad"
# (11 165 sad samples vs 1 769 focused) — below this threshold we use feature-based.
_MIN_MODEL_ACCURACY = 0.60

# ── Device-aware gamma masking ────────────────────────────────────────────────
# Gamma (30-50 Hz) is dominated by EMG (muscle artifact) on consumer dry-electrode
# EEG devices.  When one of these devices is active, gamma features are zeroed
# before the LGBM prediction so the model relies on alpha/beta/theta instead.
# Research-grade EEG (gel electrodes, controlled lab) uses gamma normally.

# Gamma feature indices in the 85-feature vector
# band-major layout: delta(0-15), theta(16-31), alpha(32-47), beta(48-63), gamma(64-79)
# DASM: delta(80) theta(81) alpha(82) beta(83) gamma(84)
_GAMMA_FEAT_IDX: list[int] = list(range(64, 80)) + [84]  # 17 features

# Consumer dry-electrode devices — gamma is EMG, not neural signal
_CONSUMER_EEG_DEVICES: frozenset[str] = frozenset({
    "muse_2", "muse_2_bled", "muse_s", "muse_s_bled",
    "muse_2016", "muse_2016_bled", "muse",
})


class EmotionClassifier:
    """EEG-based emotion classifier with ONNX/sklearn/feature-based inference.

    Priority: multichannel DEAP model → ONNX → sklearn .pkl → feature-based.
    Trained models are only used if their benchmark accuracy >= 60%.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.onnx_session = None
        self.sklearn_model = None
        self.feature_names = None
        self.scaler = None
        self.multichannel_model = False  # True when loaded pkl was trained with multichannel features
        self.model_type = "feature-based"
        self._benchmark_accuracy = 0.0

        # Exponential moving average for smoothing (prevents flickering)
        self._ema_probs = None  # smoothed probabilities
        self._ema_alpha = 0.35  # smoothing factor — 0.35 gives slightly faster response
        #                         than 0.30 while still suppressing rapid noise bursts
        self._history = deque(maxlen=10)  # recent band power snapshots

        # Device-aware gamma masking — set via set_device_type() on connect/disconnect
        self.device_type: Optional[str] = None

        # Mega cross-dataset LGBM with global PCA (DEAP+DREAMER+GAMEEMO+DENS, highest priority)
        self.mega_lgbm_model  = None
        self.mega_lgbm_scaler = None  # StandardScaler (85→85)
        self.mega_lgbm_pca    = None  # PCA (85→80)
        self._mega_lgbm_benchmark = 0.0
        self._try_load_mega_lgbm()
        # Muse-native LightGBM (85 raw features, no PCA)
        self.lgbm_muse_model = None
        self.lgbm_muse_scaler = None
        self._lgbm_muse_benchmark = 0.0
        self._try_load_lgbm_muse_model()
        # Try loading the DEAP-trained multichannel model first
        self._try_load_deap_model()

        if model_path and self.model_type == "feature-based":
            self._load_model(model_path)

    # ── Device-aware gamma masking ─────────────────────────────────────────────

    def set_device_type(self, device_type: Optional[str]) -> None:
        """Notify the classifier which EEG device is connected.

        Call this whenever a device connects or disconnects.  When a consumer
        dry-electrode device (Muse 2, Muse S, …) is active, gamma features are
        zeroed before LGBM inference because gamma is EMG on those devices.
        """
        self.device_type = device_type

    @property
    def _is_consumer_device(self) -> bool:
        """True when the active device records EMG-contaminated gamma."""
        if self.device_type is None:
            return False
        return self.device_type.lower() in _CONSUMER_EEG_DEVICES

    def _try_load_deap_model(self):
        """Try loading the DEAP-trained Muse 2 model (best accuracy)."""
        from pathlib import Path
        pkl_path = Path("models/saved/emotion_deap_muse.pkl")
        if not pkl_path.exists():
            return

        bench = self._read_deap_benchmark()
        if bench < _MIN_MODEL_ACCURACY:
            return

        try:
            import joblib
            data = joblib.load(pkl_path)
            self.sklearn_model = data["model"]
            self.feature_names = data["feature_names"]
            self.scaler = data.get("scaler")
            self.multichannel_model = bool(data.get("multichannel", False))
            self.model_type = "sklearn-deap"
            self._benchmark_accuracy = bench
        except Exception:
            pass

    def _try_load_mega_lgbm(self) -> None:
        """Load the mega cross-dataset LGBM (global PCA 85→80, DEAP+DREAMER+GAMEEMO+DENS)."""
        import json
        from pathlib import Path
        bench_path = Path("benchmarks/emotion_mega_lgbm_benchmark.json")
        model_path = Path("models/saved/emotion_mega_lgbm.pkl")
        if not bench_path.exists() or not model_path.exists():
            return
        try:
            bench_data = json.loads(bench_path.read_text())
            acc = float(bench_data.get("accuracy", 0.0))
            if acc < _MIN_MODEL_ACCURACY:
                return
            import pickle
            with open(model_path, "rb") as fh:
                payload = pickle.load(fh)
            self.mega_lgbm_model  = payload["model"]
            self.mega_lgbm_scaler = payload["scaler"]
            self.mega_lgbm_pca    = payload["pca"]
            self._mega_lgbm_benchmark = acc
            print(f"[EmotionClassifier] Loaded mega cross-dataset LGBM "
                  f"(CV acc={acc:.4f}, datasets={payload.get('datasets')})")
        except Exception:
            pass

    def _try_load_lgbm_muse_model(self) -> None:
        """Load the Muse-native LightGBM model (80 raw features, no PCA)."""
        import json
        import joblib
        from pathlib import Path
        bench_path = Path("benchmarks/emotion_lgbm_muse_live_benchmark.json")
        model_path = Path("models/saved/emotion_lgbm_muse_live.pkl")
        if not bench_path.exists() or not model_path.exists():
            return
        try:
            bench_data = json.loads(bench_path.read_text())
            acc = float(bench_data.get("accuracy", 0.0))
            if acc < _MIN_MODEL_ACCURACY:
                return
            payload = joblib.load(model_path)
            self.lgbm_muse_model = payload["model"]
            self.lgbm_muse_scaler = payload["scaler"]
            self._lgbm_muse_benchmark = acc
            print(f"[EmotionClassifier] Loaded Muse-native LightGBM (CV acc={acc:.4f})")
        except Exception:
            pass

    def _load_model(self, model_path: str):
        """Load model from file (ONNX or sklearn pkl)."""
        self._benchmark_accuracy = self._read_benchmark()

        if self._benchmark_accuracy < _MIN_MODEL_ACCURACY:
            return

        if model_path.endswith(".onnx"):
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(model_path)
                self.model_type = "onnx"
            except Exception:
                pass
        elif model_path.endswith(".pkl"):
            try:
                import joblib
                data = joblib.load(model_path)
                self.sklearn_model = data["model"]
                self.feature_names = data["feature_names"]
                self.scaler = data.get("scaler")
                self.multichannel_model = bool(data.get("multichannel", False))
                self.model_type = "sklearn"
            except Exception:
                pass

    @staticmethod
    def _read_benchmark() -> float:
        """Read the latest benchmark accuracy from disk."""
        import json
        from pathlib import Path
        bench_path = Path("benchmarks/emotion_classifier_benchmark.json")
        if bench_path.exists():
            try:
                data = json.loads(bench_path.read_text())
                return float(data.get("accuracy", 0))
            except Exception:
                pass
        return 0.0

    @staticmethod
    def _read_deap_benchmark() -> float:
        """Read the DEAP model benchmark accuracy (separate from the GBM classifier benchmark)."""
        import json
        from pathlib import Path
        bench_path = Path("benchmarks/emotion_deap_benchmark.json")
        if bench_path.exists():
            try:
                data = json.loads(bench_path.read_text())
                return float(data.get("accuracy", 0))
            except Exception:
                pass
        return 0.0

    def _extract_muse_live_features(self, eeg: np.ndarray, fs: float) -> np.ndarray:
        """Extract 85-feature vector for the Muse-native LightGBM model.

        Feature layout:
            [0:80]  80 band-power stats: 5 bands × 4 channels × 4 stats (mean,std,med,iqr)
                    Band-major order: Delta_TP9, Delta_AF7, Delta_AF8, Delta_TP10, Theta_TP9, ...
            [80:85] 5 DASM features: mean(AF8_band) - mean(AF7_band) for each of 5 bands
                    AF7=ch1 (left-frontal), AF8=ch2 (right-frontal)
        Matches train_cross_dataset_lgbm.extract_features() exactly.
        """
        WIN_SW = 128   # 0.5-sec sub-window at 256 Hz
        HOP    = 64    # 50 % overlap → ~30 sub-windows per 4-sec epoch

        n_samples = eeg.shape[1]
        powers: list = []

        for start in range(0, n_samples - WIN_SW + 1, HOP):
            end = start + WIN_SW
            sub_bands = []
            for ch in range(4):  # TP9=0, AF7=1, AF8=2, TP10=3
                try:
                    proc = preprocess(eeg[ch, start:end], fs)
                    bp   = extract_band_powers(proc, fs)
                    sub_bands.append([
                        bp.get("delta", 0.0),
                        bp.get("theta", 0.0),
                        bp.get("alpha", 0.0),
                        bp.get("beta",  0.0),
                        bp.get("gamma", 0.0),
                    ])
                except Exception:
                    sub_bands.append([0.0] * 5)
            # sub_bands: (4 channels, 5 bands) → transpose to (5 bands, 4 channels)
            powers.append(np.array(sub_bands, dtype=np.float32).T)

        if len(powers) == 0:
            return np.zeros(85, dtype=np.float32)

        arr = np.array(powers, dtype=np.float32)  # (n_subwins, 5, 4)

        feat: list = []
        # 80 band-power stats
        for band_idx in range(5):
            for ch_idx in range(4):
                vals = arr[:, band_idx, ch_idx]
                vals = vals[np.isfinite(vals)]
                if len(vals) < 2:
                    feat.extend([0.0, 0.0, 0.0, 0.0])
                else:
                    feat.extend([
                        float(np.mean(vals)),
                        float(np.std(vals)),
                        float(np.median(vals)),
                        float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                    ])

        # 5 DASM features: mean(AF8_band) - mean(AF7_band) per band
        # AF7=ch1 (left frontal), AF8=ch2 (right frontal)
        for band_idx in range(5):
            af8 = arr[:, band_idx, 2]
            af8 = af8[np.isfinite(af8)]
            af7 = arr[:, band_idx, 1]
            af7 = af7[np.isfinite(af7)]
            if len(af8) > 0 and len(af7) > 0:
                feat.append(float(np.mean(af8)) - float(np.mean(af7)))
            else:
                feat.append(0.0)

        return np.array(feat, dtype=np.float32)  # 85 features

    def _predict_lgbm_muse(self, eeg: np.ndarray, fs: float) -> Dict:
        """Run Muse-native LightGBM inference (85 features: 80 band-power + 5 DASM, no PCA).

        Expands the 3-class LGBM output (positive/neutral/negative)
        to the 6-class EMOTIONS list using ancillary EEG features.
        """
        # Artifact rejection — freeze EMA on bad epoch
        if np.any(np.abs(eeg) > _ARTIFACT_THRESHOLD_UV):
            if self._ema_probs is None:
                self._ema_probs = np.ones(6, dtype=np.float32) / 6
            return self._build_muse_result(
                int(np.argmax(self._ema_probs)), self._ema_probs,
                eeg, fs, artifact_detected=True
            )

        # Feature extraction + scale
        feat = self._extract_muse_live_features(eeg, fs)
        # Zero gamma features for consumer dry-electrode devices (gamma = EMG)
        if self._is_consumer_device:
            feat[_GAMMA_FEAT_IDX] = 0.0
        feat_scaled = self.lgbm_muse_scaler.transform(feat.reshape(1, -1))

        # LGBM predict → (positive=0, neutral=1, negative=2) probabilities
        proba3 = self.lgbm_muse_model.predict_proba(feat_scaled)[0]
        p_pos, p_neu, p_neg = float(proba3[0]), float(proba3[1]), float(proba3[2])

        # Ancillary features for 3→6 expansion
        try:
            proc_af7 = preprocess(eeg[1], fs)
            proc_af8 = preprocess(eeg[2], fs)
            bp7  = extract_band_powers(proc_af7, fs)
            bp8  = extract_band_powers(proc_af8, fs)
            a7   = max(bp7.get("alpha", 1e-6), 1e-6)
            a8   = max(bp8.get("alpha", 1e-6), 1e-6)
            alpha = (a7 + a8) / 2
            beta  = (max(bp7.get("beta",  1e-6), 1e-6)
                     + max(bp8.get("beta",  1e-6), 1e-6)) / 2
            theta = (max(bp7.get("theta", 1e-6), 1e-6)
                     + max(bp8.get("theta", 1e-6), 1e-6)) / 2
            delta = (max(bp7.get("delta", 1e-6), 1e-6)
                     + max(bp8.get("delta", 1e-6), 1e-6)) / 2
            faa   = float(np.log(a8) - np.log(a7))
        except Exception:
            alpha = beta = theta = delta = 0.1
            faa = 0.0

        beta_alpha = beta / max(alpha, 1e-6)
        arousal = float(np.clip(
            0.45 * beta  / max(beta + alpha, 1e-6)
            + 0.30 * (1 - alpha / max(alpha + beta + theta, 1e-6))
            + 0.25 * (1 - delta / max(delta + beta, 1e-6)),
            0, 1))
        faa_val = float(np.clip(np.tanh(faa * 2.0), -1, 1))

        # 3→6 class expansion
        probs6 = np.zeros(6, dtype=np.float32)
        # positive → happy (0) or relaxed (4)
        if arousal > 0.5:
            probs6[0] += p_pos
        else:
            probs6[4] += p_pos
        # neutral → focused (5) or relaxed (4)
        if beta_alpha > 1.2:
            probs6[5] += p_neu
        else:
            probs6[4] += p_neu
        # negative → sad (1), angry (2), fearful (3)
        if faa_val < -0.2 and arousal < 0.5:
            probs6[1] += p_neg   # sad
        elif faa_val < 0.0 and arousal > 0.5:
            probs6[2] += p_neg   # angry
        elif arousal > 0.55:
            probs6[3] += p_neg   # fearful
        else:
            probs6[1] += p_neg   # default → sad

        total = probs6.sum()
        if total > 0:
            probs6 /= total

        # EMA smoothing
        if self._ema_probs is None:
            self._ema_probs = probs6.copy()
        else:
            self._ema_probs = (self._ema_alpha * probs6
                               + (1 - self._ema_alpha) * self._ema_probs)

        smoothed = self._ema_probs
        emotion_idx = int(np.argmax(smoothed))
        return self._build_muse_result(emotion_idx, smoothed, eeg, fs, artifact_detected=False)

    def _build_muse_result(self, emotion_idx: int, smoothed: np.ndarray,
                           eeg: np.ndarray, fs: float,
                           artifact_detected: bool) -> Dict:
        """Build the standard 15-key return dict for the Muse-native LGBM path."""
        try:
            proc  = preprocess(eeg[1], fs)   # AF7
            bp    = extract_band_powers(proc, fs)
            de    = differential_entropy(proc, fs)
            alpha     = max(bp.get("alpha",    1e-6), 1e-6)
            beta      = max(bp.get("beta",     1e-6), 1e-6)
            theta     = max(bp.get("theta",    1e-6), 1e-6)
            delta     = max(bp.get("delta",    1e-6), 1e-6)
            high_beta = max(bp.get("high_beta",1e-6), 1e-6)
            low_beta  = max(bp.get("low_beta", 1e-6), 1e-6)
            gamma     = max(bp.get("gamma",    1e-6), 1e-6)
            bands = {k: float(v) for k, v in bp.items()
                     if k in ("delta","theta","alpha","beta","low_beta","high_beta","gamma")}
        except Exception:
            alpha = beta = theta = delta = high_beta = low_beta = gamma = 0.1
            bands = {}
            de = {}

        try:
            dasm_rasm = compute_dasm_rasm(eeg, fs, left_ch=1, right_ch=2)
        except Exception:
            dasm_rasm = {}
        try:
            fmt = compute_frontal_midline_theta(eeg[1], fs)
        except Exception:
            fmt = {}
        try:
            asym = compute_frontal_asymmetry(eeg, fs, left_ch=1, right_ch=2)
            faa_valence = float(asym.get("asymmetry_valence", 0.0))
        except Exception:
            faa_valence = 0.0

        beta_alpha = beta / max(alpha, 1e-6)
        dasm_alpha_val = float(dasm_rasm.get("dasm_alpha", 0.0)) * 0.5
        valence_abr = float(np.clip(
            0.65 * np.tanh((alpha / beta - 0.7) * 2.0)
            + 0.35 * np.tanh((alpha - 0.15) * 4), -1, 1))
        valence = float(np.clip(
            0.40 * valence_abr + 0.35 * faa_valence + 0.25 * dasm_alpha_val, -1, 1))
        arousal = float(np.clip(
            0.45 * beta  / max(beta + alpha, 1e-6)
            + 0.30 * (1 - alpha / max(alpha + beta + theta, 1e-6))
            + 0.25 * (1 - delta / max(delta + beta, 1e-6)),
            0, 1))

        dasm_beta_stress = float(dasm_rasm.get("dasm_beta", 0.0)) * 0.5
        theta_beta_ratio = theta / max(beta, 1e-6)
        stress_index = float(np.clip(
            0.40 * min(1, beta_alpha * 0.3)
            + 0.25 * max(0, 1 - alpha * 2.5)
            + 0.25 * min(1, high_beta * 4)
            + 0.10 * dasm_beta_stress, 0, 1))
        focus_index = float(np.clip(
            0.45 * min(1, beta * 3.5)
            + 0.40 * max(0, 1 - theta_beta_ratio * 0.40)
            + 0.15 * min(1, low_beta * 5), 0, 1))
        relaxation_index = float(np.clip(
            0.50 * min(1, alpha * 2.5)
            + 0.30 * max(0, 1 - beta_alpha * 0.3)
            + 0.20 * min(1, theta * 1.5), 0, 1))
        anger_index = float(np.clip(
            0.40 * min(1, max(0, beta_alpha - 1.0) * 0.5)
            + 0.30 * max(0, 1 - alpha * 5)
            + 0.20 * min(1, high_beta * 3)
            + 0.10 * min(1, gamma * 3), 0, 1))
        fear_index = float(np.clip(
            0.35 * min(1, max(0, beta_alpha - 1.5) * 0.6)
            + 0.30 * min(1, high_beta * 5)
            + 0.25 * max(0, 1 - alpha * 5)
            + 0.10 * max(0, arousal - 0.45), 0, 1))

        top_conf    = float(np.max(smoothed))
        emotion_lbl = EMOTIONS[emotion_idx]
        return {
            "emotion":               emotion_lbl,
            "emotion_index":         emotion_idx,
            "confidence":            top_conf,
            "probabilities":         {EMOTIONS[i]: float(p) for i, p in enumerate(smoothed)},
            "valence":               valence,
            "arousal":               arousal,
            "stress_index":          stress_index,
            "focus_index":           focus_index,
            "relaxation_index":      relaxation_index,
            "anger_index":           anger_index,
            "fear_index":            fear_index,
            "band_powers":           bands,
            "differential_entropy":  de,
            "dasm_rasm":             dasm_rasm,
            "frontal_midline_theta": fmt,
            "artifact_detected":     artifact_detected,
            "model_type":            "lgbm-muse",
        }

    def _predict_mega_lgbm(self, eeg: np.ndarray, fs: float) -> Dict:
        """Run mega cross-dataset LGBM inference (85 raw → PCA 80 → 3-class → 6-class).

        Reuses _extract_muse_live_features() and _build_muse_result() from the
        Muse-native LGBM path for identical output format.
        """
        if np.any(np.abs(eeg) > _ARTIFACT_THRESHOLD_UV):
            if self._ema_probs is None:
                self._ema_probs = np.ones(6, dtype=np.float32) / 6
            return self._build_muse_result(
                int(np.argmax(self._ema_probs)), self._ema_probs,
                eeg, fs, artifact_detected=True
            )

        feat = self._extract_muse_live_features(eeg, fs)   # 85-dim
        if self._is_consumer_device:
            feat[_GAMMA_FEAT_IDX] = 0.0

        # Global scaler → PCA → LGBM
        feat_sc  = self.mega_lgbm_scaler.transform(feat.reshape(1, -1))
        feat_pca = self.mega_lgbm_pca.transform(feat_sc)
        proba3   = self.mega_lgbm_model.predict_proba(feat_pca)[0]
        p_pos, p_neu, p_neg = float(proba3[0]), float(proba3[1]), float(proba3[2])

        # Ancillary features for 3→6 expansion (identical to _predict_lgbm_muse)
        try:
            proc_af7 = preprocess(eeg[1], fs)
            proc_af8 = preprocess(eeg[2], fs)
            bp7  = extract_band_powers(proc_af7, fs)
            bp8  = extract_band_powers(proc_af8, fs)
            a7   = max(bp7.get("alpha", 1e-6), 1e-6)
            a8   = max(bp8.get("alpha", 1e-6), 1e-6)
            alpha = (a7 + a8) / 2
            beta  = (max(bp7.get("beta",  1e-6), 1e-6) + max(bp8.get("beta",  1e-6), 1e-6)) / 2
            theta = (max(bp7.get("theta", 1e-6), 1e-6) + max(bp8.get("theta", 1e-6), 1e-6)) / 2
            delta = (max(bp7.get("delta", 1e-6), 1e-6) + max(bp8.get("delta", 1e-6), 1e-6)) / 2
            faa   = float(np.log(a8) - np.log(a7))
        except Exception:
            alpha = beta = theta = delta = 0.1
            faa = 0.0

        beta_alpha = beta / max(alpha, 1e-6)
        arousal    = float(np.clip(
            0.45 * beta / max(beta + alpha, 1e-6)
            + 0.30 * (1 - alpha / max(alpha + beta + theta, 1e-6))
            + 0.25 * (1 - delta / max(delta + beta, 1e-6)), 0, 1))
        faa_val = float(np.clip(np.tanh(faa * 2.0), -1, 1))

        probs6 = np.zeros(6, dtype=np.float32)
        if arousal > 0.5:
            probs6[0] += p_pos
        else:
            probs6[4] += p_pos
        if beta_alpha > 1.2:
            probs6[5] += p_neu
        else:
            probs6[4] += p_neu
        if faa_val < -0.2 and arousal < 0.5:
            probs6[1] += p_neg
        elif faa_val < 0.0 and arousal > 0.5:
            probs6[2] += p_neg
        elif arousal > 0.55:
            probs6[3] += p_neg
        else:
            probs6[1] += p_neg

        total = probs6.sum()
        if total > 0:
            probs6 /= total

        if self._ema_probs is None:
            self._ema_probs = probs6.copy()
        else:
            self._ema_probs = (self._ema_alpha * probs6
                               + (1 - self._ema_alpha) * self._ema_probs)
        smoothed    = self._ema_probs
        emotion_idx = int(np.argmax(smoothed))
        result = self._build_muse_result(emotion_idx, smoothed, eeg, fs, artifact_detected=False)
        result["model_type"] = "mega-lgbm-pca"
        return result

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Classify emotion from EEG signal.

        Args:
            eeg: 1D (single channel) or 2D (n_channels, n_samples) array.
            fs: Sampling frequency.
        """
        # Mega cross-dataset LGBM with global PCA — highest priority
        if (self.mega_lgbm_model is not None and eeg.ndim == 2 and eeg.shape[0] >= 4
                and self._mega_lgbm_benchmark >= _MIN_MODEL_ACCURACY):
            return self._predict_mega_lgbm(eeg, fs)
        # Muse-native LightGBM (85 raw features, no PCA)
        if (self.lgbm_muse_model is not None and eeg.ndim == 2 and eeg.shape[0] >= 4
                and self._lgbm_muse_benchmark >= _MIN_MODEL_ACCURACY):
            return self._predict_lgbm_muse(eeg, fs)
        # Multichannel DEAP model (requires exactly 4 Muse 2 channels: AF7, AF8, TP9, TP10)
        # Gated on accuracy — DEAP model at ~51% has severe class-imbalance bias toward "sad"
        # (11,165 sad samples vs 1,769 focused), so skip it unless it meets the 60% threshold.
        if (self.model_type == "sklearn-deap" and eeg.ndim == 2 and eeg.shape[0] >= 4
                and self._benchmark_accuracy >= _MIN_MODEL_ACCURACY):
            return self._predict_multichannel(eeg, fs)
        if self.onnx_session is not None and self._benchmark_accuracy >= _MIN_MODEL_ACCURACY:
            return self._predict_onnx(eeg if eeg.ndim == 1 else eeg[0], fs)
        if self.sklearn_model is not None and self._benchmark_accuracy >= _MIN_MODEL_ACCURACY:
            # Multichannel sklearn models (trained with DASM/RASM/FAA features) need the
            # full (n_channels, n_samples) array; single-channel models need eeg[0].
            if self.multichannel_model and eeg.ndim == 2:
                return self._predict_sklearn(eeg, fs)
            return self._predict_sklearn(eeg if eeg.ndim == 1 else eeg[0], fs)
        return self._predict_features(eeg, fs)  # pass full multichannel array for FAA

    # ────────────────────────────────────────────────────────────────
    # Multichannel DEAP-trained model (primary for Muse 2)
    # ────────────────────────────────────────────────────────────────

    def _predict_multichannel(self, channels: np.ndarray, fs: float) -> Dict:
        """Predict using the DEAP-trained multichannel model."""
        from training.train_deap_muse import extract_multichannel_features

        features = extract_multichannel_features(channels, fs)
        feat_vec = np.array([features.get(k, 0.0) for k in self.feature_names]).reshape(1, -1)

        if self.scaler is not None:
            feat_vec = self.scaler.transform(feat_vec)

        feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)

        probs = self.sklearn_model.predict_proba(feat_vec)[0]
        emotion_idx = int(np.argmax(probs))

        # EMA smoothing
        if self._ema_probs is None:
            self._ema_probs = probs
        else:
            self._ema_probs = self._ema_alpha * probs + (1 - self._ema_alpha) * self._ema_probs

        smoothed = self._ema_probs / (self._ema_probs.sum() + 1e-10)
        emotion_idx = int(np.argmax(smoothed))

        # Extract band powers from first channel for extra metrics
        processed = preprocess(channels[0], fs)
        bands = extract_band_powers(processed, fs)
        de = differential_entropy(processed, fs)

        alpha_raw     = bands.get("alpha",     0)
        beta_raw      = bands.get("beta",      0)
        theta_raw     = bands.get("theta",     0)
        gamma_raw     = bands.get("gamma",     0)
        delta_raw     = bands.get("delta",     0)
        high_beta_raw = bands.get("high_beta", 0)
        low_beta_raw  = bands.get("low_beta",  0)

        total_power = alpha_raw + beta_raw + theta_raw + gamma_raw + delta_raw
        if total_power < 1e-6:
            total_power = 1.0
        alpha     = alpha_raw     / total_power
        beta      = beta_raw      / total_power
        theta     = theta_raw     / total_power
        gamma     = gamma_raw     / total_power
        high_beta = high_beta_raw / total_power
        low_beta  = low_beta_raw  / total_power
        delta     = delta_raw     / total_power

        alpha_beta_ratio = alpha / max(beta, 1e-10)
        beta_alpha_ratio = beta / max(alpha, 1e-10)
        theta_beta_ratio = theta / max(beta, 1e-10)

        # Frontal alpha asymmetry (Davidson 1992): richer valence signal
        # For Muse 2 channel order: AF7(0), AF8(1), TP9(2), TP10(3)
        asymmetry = compute_frontal_asymmetry(channels, fs, left_ch=0, right_ch=1)
        asym_valence = asymmetry.get("asymmetry_valence", 0.0)  # -1 to +1

        # Blend alpha-beta ratio valence (60%) with frontal asymmetry (40%)
        # Asymmetry is a stronger valence signal when 4 channels are available.
        # Neutral baseline 0.8 (eyes-open resting: beta slightly > alpha for Muse).
        # Gamma removed from valence — it's typically < 0.05 on consumer EEG and
        # would constantly subtract from valence, falsely pushing toward negative.
        valence_abr = float(np.tanh((alpha_beta_ratio - 0.8) * 1.5) * 0.6)
        valence = float(np.clip(0.60 * valence_abr + 0.40 * asym_valence, -1, 1))
        # Arousal without gamma: frontal gamma at AF7/AF8 is frontalis EMG, not neural.
        arousal = float(np.clip(
            0.45 * beta / max(beta + alpha, 1e-10)
            + 0.30 * (1.0 - alpha / max(alpha + beta + theta, 1e-10))
            + 0.25 * (1.0 - delta / max(delta + beta, 1e-10)),
            0, 1
        ))

        stress_index = float(np.clip(
            0.40 * min(1, beta_alpha_ratio * 0.3) + 0.25 * max(0, 1 - alpha * 2.5)
            + 0.20 * min(1, high_beta * 4) + 0.15 * min(1, gamma * 2), 0, 1))
        focus_index = float(np.clip(
            0.40 * min(1, beta * 3.5) + 0.35 * max(0, 1 - theta_beta_ratio * 0.40)
            + 0.15 * min(1, low_beta * 5) + 0.10 * min(1, gamma * 2), 0, 1))
        relaxation_index = float(np.clip(
            0.50 * min(1, alpha * 2.5) + 0.30 * max(0, 1 - beta_alpha_ratio * 0.3)
            + 0.20 * min(1, theta * 1.5), 0, 1))
        anger_index = float(np.clip(
            0.35 * min(1, max(0, beta_alpha_ratio - 1.0) * 0.5) + 0.25 * min(1, gamma * 3)
            + 0.25 * max(0, 1 - alpha * 5) + 0.15 * min(1, high_beta * 3), 0, 1))
        fear_index = float(np.clip(
            0.35 * min(1, max(0, beta_alpha_ratio - 1.5) * 0.6) + 0.30 * min(1, high_beta * 5)
            + 0.25 * max(0, 1 - alpha * 5) + 0.10 * max(0, arousal - 0.45), 0, 1))

        return {
            "emotion": EMOTIONS[emotion_idx],
            "emotion_index": emotion_idx,
            "confidence": float(smoothed[emotion_idx]),
            "probabilities": {EMOTIONS[i]: float(p) for i, p in enumerate(smoothed)},
            "valence": valence,
            "arousal": arousal,
            "stress_index": stress_index,
            "focus_index": focus_index,
            "relaxation_index": relaxation_index,
            "anger_index": anger_index,
            "fear_index": fear_index,
            "band_powers": bands,
            "differential_entropy": de,
        }

    # ────────────────────────────────────────────────────────────────
    # Feature-based classifier (primary path for Muse 2)
    # ────────────────────────────────────────────────────────────────

    def _predict_features(self, eeg: np.ndarray, fs: float) -> Dict:
        """Neuroscience-backed emotion classification from band powers.

        Accepts 1D (single channel) or 2D (n_channels, n_samples) input.
        When multichannel Muse 2 data is provided (AF7=ch0, AF8=ch1), computes
        Frontal Alpha Asymmetry (FAA) per Davidson (1992) to improve valence accuracy.

        EEG-emotion correlates used:
        - Alpha (8-12 Hz): inversely related to cortical arousal; dominant in relaxation
        - Low-beta (12-20 Hz): active cognition, focus, motor planning
        - High-beta (20-30 Hz): anxiety, stress, fear — strongest differentiator for fearful
        - Beta (12-30 Hz): active thinking, concentration, anxiety when excessive
        - Theta (4-8 Hz): drowsiness, meditation, creative states
        - Gamma (30+ Hz): EXCLUDED from Muse 2 arousal/stress/valence — dominated by
          frontalis EMG artifact at AF7/AF8 sites (not neural). Only used as minimal
          discriminator in angry vs fearful distinction.
        - Delta (0.5-4 Hz): deep sleep, regeneration
        """
        # ── Artifact rejection ───────────────────────────────────
        # Reject epochs with extreme amplitude (eye blinks: 100-200 µV, EMG bursts).
        # Return the last EMA-smoothed state rather than a noisy artifact-driven result.
        max_amp = float(np.max(np.abs(eeg)))
        if max_amp > _ARTIFACT_THRESHOLD_UV:
            if self._ema_probs is not None:
                smoothed = self._ema_probs / (self._ema_probs.sum() + 1e-10)
                emotion_idx = int(np.argmax(smoothed))
                top_conf = float(smoothed[emotion_idx])
                emotion_label = EMOTIONS[emotion_idx] if top_conf >= 0.25 else "neutral"
                result = {e: float(smoothed[i]) for i, e in enumerate(EMOTIONS)}
                # Return frozen EMA state — do not update with artifact epoch
                return {
                    "emotion": emotion_label, "emotion_index": emotion_idx,
                    "confidence": top_conf, "probabilities": result,
                    "valence": 0.0, "arousal": 0.5,
                    "stress_index": 0.5, "focus_index": 0.5,
                    "relaxation_index": 0.5, "anger_index": 0.0, "fear_index": 0.0,
                    "band_powers": {}, "differential_entropy": {},
                    "dasm_rasm": {}, "frontal_midline_theta": {}, "artifact_detected": True,
                }
            # No prior EMA — return neutral with artifact flag
            neutral = {e: 1.0 / 6 for e in EMOTIONS}
            return {
                "emotion": "neutral", "emotion_index": 5,
                "confidence": 1.0 / 6, "probabilities": neutral,
                "valence": 0.0, "arousal": 0.5,
                "stress_index": 0.5, "focus_index": 0.5,
                "relaxation_index": 0.5, "anger_index": 0.0, "fear_index": 0.0,
                "band_powers": {}, "differential_entropy": {},
                "dasm_rasm": {}, "frontal_midline_theta": {}, "artifact_detected": True,
            }

        # ── Multichannel support ─────────────────────────────────
        # Keep original channels array for FAA; use channel 0 (AF7) for band powers.
        channels = eeg if eeg.ndim == 2 else None
        signal = eeg[0] if eeg.ndim == 2 else eeg

        processed = preprocess(signal, fs)
        bands = extract_band_powers(processed, fs)
        de = differential_entropy(processed, fs)

        alpha_raw     = bands.get("alpha",     0)
        beta_raw      = bands.get("beta",      0)
        theta_raw     = bands.get("theta",     0)
        gamma_raw     = bands.get("gamma",     0)
        delta_raw     = bands.get("delta",     0)
        high_beta_raw = bands.get("high_beta", 0)
        low_beta_raw  = bands.get("low_beta",  0)

        # Normalize to relative fractions that sum to 1.
        # Without this the absolute power values from BrainFlow (sum often 0.4-0.8)
        # make every formula threshold fire at different levels, producing flat probs.
        total_power = alpha_raw + beta_raw + theta_raw + gamma_raw + delta_raw
        if total_power < 1e-6:
            total_power = 1.0
        alpha     = alpha_raw     / total_power
        beta      = beta_raw      / total_power
        theta     = theta_raw     / total_power
        gamma     = gamma_raw     / total_power
        delta     = delta_raw     / total_power
        high_beta = high_beta_raw / total_power
        low_beta  = low_beta_raw  / total_power

        # high_beta_frac: fraction of beta that is 20-30 Hz (fear/anxiety marker).
        # Fearful/anxious states skew beta distribution toward high-beta.
        high_beta_frac = high_beta_raw / max(beta_raw, 1e-10)

        # Store snapshot for temporal analysis
        self._history.append(bands)

        # ── Frontal Alpha Asymmetry (FAA) ────────────────────────
        # FAA = ln(R_alpha) - ln(L_alpha) → positive = approach motivation / positive affect.
        # Davidson (1992). Primary valence biomarker when Muse 2 multichannel data available.
        # Note: FAA reflects approach motivation (left-lateralized), not pure positive valence —
        # anger (approach + negative valence) also shows left FAA. Blend with ABR for robustness.
        faa_valence = 0.0
        if channels is not None and channels.shape[0] >= 2:
            # BrainFlow Muse 2 channel order: ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10
            # FAA requires ch1 (AF7, left frontal) and ch2 (AF8, right frontal).
            asym = compute_frontal_asymmetry(channels, fs, left_ch=1, right_ch=2)
            faa_valence = asym.get("asymmetry_valence", 0.0)

        # ── DASM Alpha (DE-based asymmetry complement to FAA) ────
        # DASM_alpha = DE(AF8_alpha) - DE(AF7_alpha). Same directional meaning as FAA
        # but computed from differential entropy rather than raw log-power. Adds an
        # independent estimate of hemispheric alpha asymmetry for valence blending.
        # DASM_beta used to refine stress: right-dominant beta elevation = withdrawal stress.
        dasm_rasm: Dict = {}
        dasm_alpha_valence = 0.0
        dasm_beta_stress = 0.0
        if channels is not None and channels.shape[0] >= 3:
            dasm_rasm = compute_dasm_rasm(channels, fs, left_ch=1, right_ch=2)
            dasm_alpha_valence = float(
                np.clip(np.tanh(dasm_rasm.get("dasm_alpha", 0.0)), -1, 1)
            )
            # Positive dasm_beta = right-dominant beta = slight withdrawal/stress signal
            dasm_beta_stress = float(
                np.clip(np.tanh(dasm_rasm.get("dasm_beta", 0.0) * 0.5), 0, 1)
            )

        # ── Frontal Midline Theta (FMT) ──────────────────────────
        # FMT = ACC/mPFC theta (4-8 Hz). More reference-robust than FAA because
        # theta at Fpz is less sensitive to the Fpz reference electrode problem.
        # Left hemisphere FMT asymmetry → pleasant valence.
        # Used as a supplemental output in the emotion dict.
        fmt: Dict = {}
        if channels is not None and channels.shape[0] >= 2:
            try:
                fmt = compute_frontal_midline_theta(channels[1], fs)  # AF7 channel
            except Exception:
                fmt = {}

        # ── Valence (pleasantness) ──────────────────────────────
        # Alpha/beta ratio (ABR) as secondary valence signal.
        # Reference ratio 0.7: eyes-open resting naturally has beta > alpha (ratio ~0.5-0.8).
        alpha_beta_ratio = alpha / max(beta, 1e-10)
        valence_abr = (
            0.65 * np.tanh((alpha_beta_ratio - 0.7) * 2.0)
            + 0.35 * np.tanh((alpha - 0.15) * 4)
        )
        if channels is not None and channels.shape[0] >= 3 and dasm_rasm:
            # 3-signal blend: ABR + FAA + DASM_alpha (all three available).
            # DASM_alpha provides an independent DE-based asymmetry estimate.
            valence = float(np.clip(
                0.40 * valence_abr + 0.35 * faa_valence + 0.25 * dasm_alpha_valence,
                -1, 1
            ))
        elif channels is not None and channels.shape[0] >= 2:
            # 2-signal blend: ABR + FAA
            valence = float(np.clip(0.50 * valence_abr + 0.50 * faa_valence, -1, 1))
        else:
            valence = float(np.clip(valence_abr, -1, 1))

        # ── Arousal (activation level) ──────────────────────────
        # Beta indicates cortical activation; alpha + delta indicate deactivation.
        # Gamma EXCLUDED: at Muse 2 frontal sites (AF7/AF8), 30-100 Hz is dominated
        # by frontalis EMG artifact, not neural gamma. Including it artificially
        # inflates arousal whenever the user tenses their forehead or jaw.
        arousal_raw = (
            0.45 * beta / max(beta + alpha, 1e-10)                     # beta proportion (primary)
            + 0.30 * (1.0 - alpha / max(alpha + beta + theta, 1e-10))  # inverse alpha
            + 0.25 * (1.0 - delta / max(delta + beta, 1e-10))          # inverse delta
        )
        arousal = float(np.clip(arousal_raw, 0, 1))

        # ── Emotion probability estimation ──────────────────────
        # Based on the circumplex model with EEG-specific weightings
        probs = np.zeros(6)

        theta_beta_ratio = theta / max(beta, 1e-10)
        beta_alpha_ratio = beta / max(alpha, 1e-10)

        # Happy: positive valence, moderate-high arousal, good alpha.
        # Gamma removed — predominantly frontalis EMG at Muse 2 AF7/AF8 sites.
        probs[0] = (
            0.45 * max(0, valence)                     # primary: positive valence (includes FAA)
            + 0.30 * max(0, arousal - 0.3)             # moderate-high arousal
            + 0.25 * min(1, alpha_beta_ratio * 0.8)    # alpha > beta (calm-positive)
        )

        # Sad: negative valence + low arousal.
        # Threshold lowered from -0.25 → -0.10: with FAA now blended into valence,
        # negative states register earlier. Resting FAA ≈ 0 keeps neutral at ~0;
        # genuinely sad states show negative FAA pushing valence clearly below -0.10.
        probs[1] = (
            0.50 * max(0, -valence - 0.10)                  # needs negative valence
            + 0.30 * max(0, 0.35 - arousal)                 # must be truly low arousal
            + 0.20 * max(0, theta_beta_ratio - 2.0) * 0.4  # only VERY high theta/beta
        )

        # Angry: negative valence + elevated arousal + beta dominance.
        # Key differentiator from fearful: anger = fight/approach motivation.
        # Gamma weight drastically reduced — EMG at Muse 2 frontal sites creates
        # false anger spikes whenever the user clenches jaw or tenses forehead.
        probs[2] = (
            0.40 * max(0, -valence - 0.1)                       # negative valence (primary)
            + 0.30 * max(0, arousal - 0.45)                     # elevated arousal
            + 0.25 * min(1, max(0, beta_alpha_ratio - 1.3))     # beta dominance
            + 0.05 * min(1, gamma * 2.5)                        # minimal gamma (noisy)
        )

        # Fearful/Anxious: negative valence + elevated arousal + beta dominance.
        # Key differentiator from angry: fear = freeze/avoidance (high-beta dominant,
        # LESS gamma than angry). Substantially lowered thresholds to capture moderate fear.
        # high_beta (20-30 Hz) is the strongest EEG marker for anxiety/fear states.
        probs[3] = (
            0.30 * max(0, -valence - 0.15)                       # negative valence (lowered: -0.3→-0.15)
            + 0.25 * max(0, arousal - 0.50)                      # elevated arousal (lowered: 0.6→0.50)
            + 0.30 * min(1, max(0, beta_alpha_ratio - 1.8))      # beta dominance (lowered: 2.5→1.8)
            + 0.15 * min(1, high_beta * 5.0)                     # high-beta power: primary fear marker
        )

        # Relaxed: low arousal + dominant alpha OR high theta (meditative/drowsy).
        # Theta at 30-55% of power = meditative/drowsy waking state → relaxed, not fearful.
        probs[4] = (
            0.10 * max(0, valence * 0.5)                  # positive valence is a soft bonus
            + 0.20 * max(0, 0.6 - arousal)                # low arousal
            + 0.35 * min(1, alpha * 2.5)                   # dominant alpha (primary relaxation signal)
            + 0.20 * min(1, theta * 1.5)                   # high theta = meditative/drowsy → relaxed
            + 0.15 * max(0, 1 - beta_alpha_ratio * 0.5)   # low beta relative to alpha
        )

        # Focused: moderate-high arousal, strong beta (especially low-beta), low theta/beta.
        # Low-beta (12-20 Hz) is the working-memory and attentional band; stronger weight.
        # Slightly lowered arousal threshold — focused states can be calm but engaged.
        probs[5] = (
            0.10 * max(0, 0.5 + valence * 0.5)               # soft neutral-to-positive valence bonus
            + 0.25 * min(1, max(0, arousal - 0.30) * 2.0)    # moderate arousal (lowered: 0.35→0.30)
            + 0.40 * min(1, beta * 3.5)                       # strong beta (increased: 0.35→0.40)
            + 0.25 * max(0, 1 - theta_beta_ratio * 0.40)      # low theta/beta (adjusted multiplier)
        )

        # Softmax-like normalization (temperature=2.5 for clear sharpness).
        # Subtract max before exp to prevent overflow (numerically stable softmax).
        temp = 2.5
        scaled = probs * temp
        scaled -= scaled.max()  # shift so max is 0 → exp values in (0, 1]
        probs_exp = np.exp(scaled)
        probs = probs_exp / (probs_exp.sum() + 1e-10)

        # Exponential moving average smoothing
        if self._ema_probs is None:
            self._ema_probs = probs
        else:
            self._ema_probs = self._ema_alpha * probs + (1 - self._ema_alpha) * self._ema_probs

        smoothed = self._ema_probs / (self._ema_probs.sum() + 1e-10)
        emotion_idx = int(np.argmax(smoothed))

        # ── Mental state indices (0-1 scale) ────────────────────
        # Stress: high beta/alpha ratio + low alpha + high-beta + optional DASM beta.
        # DASM beta (10% weight): right-dominant beta correlates with withdrawal stress.
        # Gamma excluded — EMG noise at AF7/AF8 inflates stress artificially (jaw clenching).
        stress_index = float(np.clip(
            0.40 * min(1, beta_alpha_ratio * 0.3)
            + 0.25 * max(0, 1 - alpha * 2.5)
            + 0.25 * min(1, high_beta * 4)         # high-beta is the real stress marker
            + 0.10 * dasm_beta_stress,              # right-dominant beta = withdrawal stress
            0, 1
        ))

        # Focus: strong beta (especially low-beta), low theta/beta, moderate arousal.
        # Gamma excluded — EMG noise at AF7/AF8 produces false focus spikes.
        focus_index = float(np.clip(
            0.45 * min(1, beta * 3.5)
            + 0.40 * max(0, 1 - theta_beta_ratio * 0.40)
            + 0.15 * min(1, low_beta * 5),         # low-beta is the primary attentional band
            0, 1
        ))

        # Relaxation: high alpha, low beta, theta welcome
        relaxation_index = float(np.clip(
            0.50 * min(1, alpha * 2.5)
            + 0.30 * max(0, 1 - beta_alpha_ratio * 0.3)
            + 0.20 * min(1, theta * 1.5),
            0, 1
        ))

        # Anger: high beta/alpha + suppressed alpha + high-beta elevation.
        # Gamma weight minimized — EMG at Muse 2 frontal sites creates false anger.
        anger_index = float(np.clip(
            0.40 * min(1, max(0, beta_alpha_ratio - 1.0) * 0.5)
            + 0.30 * max(0, 1 - alpha * 5)
            + 0.20 * min(1, high_beta * 3)
            + 0.10 * min(1, gamma * 3),            # minimal weight (EMG noise)
            0, 1
        ))

        # Fear: high-beta dominance + extreme beta/alpha + suppressed alpha.
        # Distinguishes anxious/fearful from stressed-but-focused (which lacks the
        # beta/alpha elevation) and from angry (which has more gamma, less high-beta frac).
        fear_index = float(np.clip(
            0.35 * min(1, max(0, beta_alpha_ratio - 1.5) * 0.6)
            + 0.30 * min(1, high_beta * 5)        # high-beta is the strongest fear marker
            + 0.25 * max(0, 1 - alpha * 5)        # alpha suppression in fear
            + 0.10 * max(0, arousal - 0.45),      # elevated arousal (not pure freeze)
            0, 1
        ))

        # If max confidence is below threshold the EEG doesn't clearly match any
        # of the 6 trained emotions — label as "neutral" instead of forcing a wrong label.
        # With 6 classes, random baseline = 0.167. Threshold 0.25 means the top emotion
        # is at least 50% above chance before we commit to a label — prevents borderline
        # states (e.g. slightly negative valence) from being labeled as "sad".
        _CONFIDENCE_THRESHOLD = 0.25
        top_conf = float(smoothed[emotion_idx])
        emotion_label = EMOTIONS[emotion_idx] if top_conf >= _CONFIDENCE_THRESHOLD else "neutral"

        return {
            "emotion": emotion_label,
            "emotion_index": emotion_idx,
            "confidence": top_conf,
            "probabilities": {EMOTIONS[i]: float(p) for i, p in enumerate(smoothed)},
            "valence": valence,
            "arousal": arousal,
            "stress_index": stress_index,
            "focus_index": focus_index,
            "relaxation_index": relaxation_index,
            "anger_index": anger_index,
            "fear_index": fear_index,
            "band_powers": bands,
            "differential_entropy": de,
            "dasm_rasm": dasm_rasm,
            "frontal_midline_theta": fmt,
            "artifact_detected": False,
        }

    # ────────────────────────────────────────────────────────────────
    # ONNX inference (only used if benchmark >= 60%)
    # ────────────────────────────────────────────────────────────────

    def _predict_onnx(self, eeg: np.ndarray, fs: float) -> Dict:
        """ONNX model inference."""
        processed = preprocess(eeg, fs)
        bands = extract_band_powers(processed, fs)
        features_dict = extract_features(processed, fs)
        de = differential_entropy(processed, fs)
        features = np.array(list(features_dict.values()), dtype=np.float32).reshape(1, -1)

        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: features})
        emotion_idx = int(outputs[0][0])
        prob_map = outputs[1][0]
        n_classes = len(EMOTIONS)
        probs = [float(prob_map.get(i, 0.0)) for i in range(n_classes)]

        alpha_raw     = bands.get("alpha",     0)
        beta_raw      = bands.get("beta",      0)
        theta_raw     = bands.get("theta",     0)
        gamma_raw     = bands.get("gamma",     0)
        delta_raw     = bands.get("delta",     0)
        high_beta_raw = bands.get("high_beta", 0)
        low_beta_raw  = bands.get("low_beta",  0)

        total_power = alpha_raw + beta_raw + theta_raw + gamma_raw + delta_raw
        if total_power < 1e-6:
            total_power = 1.0
        alpha     = alpha_raw     / total_power
        beta      = beta_raw      / total_power
        theta     = theta_raw     / total_power
        gamma     = gamma_raw     / total_power
        high_beta = high_beta_raw / total_power
        low_beta  = low_beta_raw  / total_power

        delta     = delta_raw     / total_power
        beta_alpha_ratio = beta / max(alpha, 1e-10)
        theta_beta_ratio = theta / max(beta, 1e-10)

        valence = float(np.clip(
            0.65 * np.tanh((alpha / max(beta, 1e-10) - 0.7) * 2.0)
            + 0.35 * np.tanh((alpha - 0.15) * 4),
            -1, 1
        ))
        arousal = float(np.clip(
            0.45 * beta / max(beta + alpha, 1e-10)
            + 0.30 * (1.0 - alpha / max(alpha + beta + theta, 1e-10))
            + 0.25 * (1.0 - delta / max(delta + beta, 1e-10)),
            0, 1
        ))

        stress_index = float(np.clip(
            0.40 * min(1, beta_alpha_ratio * 0.3) + 0.25 * max(0, 1 - alpha * 2.5)
            + 0.20 * min(1, high_beta * 4) + 0.15 * min(1, gamma * 2), 0, 1))
        focus_index = float(np.clip(
            0.40 * min(1, beta * 3.5) + 0.35 * max(0, 1 - theta_beta_ratio * 0.40)
            + 0.15 * min(1, low_beta * 5) + 0.10 * min(1, gamma * 2), 0, 1))
        relaxation_index = float(np.clip(
            0.50 * min(1, alpha * 2.5) + 0.30 * max(0, 1 - beta_alpha_ratio * 0.3)
            + 0.20 * min(1, theta * 1.5), 0, 1))
        anger_index = float(np.clip(
            0.35 * min(1, max(0, beta_alpha_ratio - 1.0) * 0.5) + 0.25 * min(1, gamma * 3)
            + 0.25 * max(0, 1 - alpha * 5) + 0.15 * min(1, high_beta * 3), 0, 1))
        fear_index = float(np.clip(
            0.35 * min(1, max(0, beta_alpha_ratio - 1.5) * 0.6) + 0.30 * min(1, high_beta * 5)
            + 0.25 * max(0, 1 - alpha * 5) + 0.10 * max(0, arousal - 0.45), 0, 1))

        return {
            "emotion": EMOTIONS[emotion_idx] if emotion_idx < n_classes else "unknown",
            "emotion_index": emotion_idx,
            "confidence": float(probs[emotion_idx]) if emotion_idx < n_classes else 0.0,
            "probabilities": {EMOTIONS[i]: probs[i] for i in range(n_classes)},
            "valence": float(np.clip(valence, -1, 1)),
            "arousal": float(np.clip(arousal, 0, 1)),
            "stress_index": stress_index,
            "focus_index": focus_index,
            "relaxation_index": relaxation_index,
            "anger_index": anger_index,
            "fear_index": fear_index,
            "band_powers": bands,
            "differential_entropy": de,
        }

    # ────────────────────────────────────────────────────────────────
    # Sklearn inference (only used if benchmark >= 60%)
    # ────────────────────────────────────────────────────────────────

    def _predict_sklearn(self, eeg: np.ndarray, fs: float) -> Dict:
        """Sklearn model inference using extracted features.

        Supports both:
          - Legacy single-channel 17-feature models (self.multichannel_model = False)
          - Multichannel 41-feature models trained with DASM/RASM/FAA features (multichannel = True)
        """
        from processing.eeg_processor import extract_features_multichannel as _mc_feats

        # ── Feature extraction ────────────────────────────────────
        if self.multichannel_model and eeg.ndim == 2:
            # Multichannel model: extract full 41-feature vector, apply stored scaler
            mc_dict = _mc_feats(eeg, fs)
            feature_vector = np.array(list(mc_dict.values()), dtype=np.float32).reshape(1, -1)
            if self.scaler is not None:
                feature_vector = self.scaler.transform(feature_vector)
            # Band powers from first channel (AF7) for downstream indices
            signal_for_bands = eeg[1] if eeg.shape[0] > 1 else eeg[0]
        else:
            signal_for_bands = eeg if eeg.ndim == 1 else eeg
            processed = preprocess(signal_for_bands, fs)
            features = extract_features(processed, fs)
            try:
                feature_vector = np.array([features[k] for k in self.feature_names]).reshape(1, -1)
            except KeyError:
                return self._predict_features(eeg, fs)
            if self.scaler is not None:
                feature_vector = self.scaler.transform(feature_vector)
            signal_for_bands = processed

        processed = preprocess(signal_for_bands if signal_for_bands.ndim == 1
                                else signal_for_bands[0], fs)
        bands = extract_band_powers(processed, fs)
        de = differential_entropy(processed, fs)
        probs = self.sklearn_model.predict_proba(feature_vector)[0]
        emotion_idx = int(np.argmax(probs))

        alpha_raw     = bands.get("alpha",     0)
        beta_raw      = bands.get("beta",      0)
        theta_raw     = bands.get("theta",     0)
        gamma_raw     = bands.get("gamma",     0)
        delta_raw     = bands.get("delta",     0)
        high_beta_raw = bands.get("high_beta", 0)
        low_beta_raw  = bands.get("low_beta",  0)

        total_power = alpha_raw + beta_raw + theta_raw + gamma_raw + delta_raw
        if total_power < 1e-6:
            total_power = 1.0
        alpha     = alpha_raw     / total_power
        beta      = beta_raw      / total_power
        theta     = theta_raw     / total_power
        gamma     = gamma_raw     / total_power
        high_beta = high_beta_raw / total_power
        low_beta  = low_beta_raw  / total_power

        delta     = delta_raw     / total_power
        beta_alpha_ratio = beta / max(alpha, 1e-10)
        theta_beta_ratio = theta / max(beta, 1e-10)

        valence = float(np.clip(
            0.65 * np.tanh((alpha / max(beta, 1e-10) - 0.7) * 2.0)
            + 0.35 * np.tanh((alpha - 0.15) * 4),
            -1, 1
        ))
        arousal = float(np.clip(
            0.45 * beta / max(beta + alpha, 1e-10)
            + 0.30 * (1.0 - alpha / max(alpha + beta + theta, 1e-10))
            + 0.25 * (1.0 - delta / max(delta + beta, 1e-10)),
            0, 1
        ))

        stress_index = float(np.clip(
            0.40 * min(1, beta_alpha_ratio * 0.3) + 0.25 * max(0, 1 - alpha * 2.5)
            + 0.20 * min(1, high_beta * 4) + 0.15 * min(1, gamma * 2), 0, 1))
        focus_index = float(np.clip(
            0.40 * min(1, beta * 3.5) + 0.35 * max(0, 1 - theta_beta_ratio * 0.40)
            + 0.15 * min(1, low_beta * 5) + 0.10 * min(1, gamma * 2), 0, 1))
        relaxation_index = float(np.clip(
            0.50 * min(1, alpha * 2.5) + 0.30 * max(0, 1 - beta_alpha_ratio * 0.3)
            + 0.20 * min(1, theta * 1.5), 0, 1))
        anger_index = float(np.clip(
            0.35 * min(1, max(0, beta_alpha_ratio - 1.0) * 0.5) + 0.25 * min(1, gamma * 3)
            + 0.25 * max(0, 1 - alpha * 5) + 0.15 * min(1, high_beta * 3), 0, 1))
        fear_index = float(np.clip(
            0.35 * min(1, max(0, beta_alpha_ratio - 1.5) * 0.6) + 0.30 * min(1, high_beta * 5)
            + 0.25 * max(0, 1 - alpha * 5) + 0.10 * max(0, arousal - 0.45), 0, 1))

        return {
            "emotion": EMOTIONS[emotion_idx],
            "emotion_index": emotion_idx,
            "confidence": float(probs[emotion_idx]),
            "probabilities": {EMOTIONS[i]: float(p) for i, p in enumerate(probs)},
            "valence": float(np.clip(valence, -1, 1)),
            "arousal": float(np.clip(arousal, 0, 1)),
            "stress_index": stress_index,
            "focus_index": focus_index,
            "relaxation_index": relaxation_index,
            "anger_index": anger_index,
            "fear_index": fear_index,
            "band_powers": bands,
            "differential_entropy": de,
        }
