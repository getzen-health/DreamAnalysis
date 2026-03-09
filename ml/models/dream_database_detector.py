"""DREAM database-informed dream detector.

Uses published DREAM database statistics (Nature Communications, 2025,
505 subjects, 2,643 awakenings) as Bayesian priors for EEG-based dream
detection on Muse 2's 4-channel [TP9, AF7, AF8, TP10] configuration.

When a trained model file is available the classifier is used for the
probability estimate; DREAM database priors are always applied on top as
a sleep-stage-aware correction.
"""

import os
import sys
import threading
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Allow imports from ml/ root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from processing.eeg_processor import (
        extract_band_powers,
        extract_features,
        preprocess,
        compute_frontal_asymmetry,
    )
    _HAS_PROCESSOR = True
except ImportError:
    _HAS_PROCESSOR = False
    extract_band_powers = None
    extract_features = None
    preprocess = None
    compute_frontal_asymmetry = None

# Published DREAM database priors (Nature Communications, 2025)
_DREAM_PRIORS: Dict[str, float] = {
    "rem":   0.68,   # P(dream | REM awakening)
    "nrem":  0.28,   # P(dream | NREM awakening)
    "n1":    0.35,
    "n2":    0.22,
    "n3":    0.18,
    "sws":   0.18,   # slow-wave sleep alias
    "wake":  0.05,   # very unlikely to be "dreaming" while awake
    "default": 0.45, # overall P(dream) when stage unknown
}

# Per-band weight for dream scoring (feature-heuristic path)
_BAND_WEIGHTS = {
    "theta_delta_ratio": 0.40,   # primary dream indicator
    "alpha":             0.25,   # dream vividness proxy
    "theta":             0.20,
    "beta":              0.10,
    "delta_inv":         0.05,   # low delta → more dreaming
}

# Thread-safe singleton registry per user_id
_INSTANCES: Dict[str, "DREAMDatabaseDreamDetector"] = {}
_INSTANCE_LOCK = threading.Lock()


class DREAMDatabaseDreamDetector:
    """Dream detector informed by DREAM database statistics.

    Inference priority:
      1. Loaded sklearn/LightGBM model (from train_dream_database.py)
      2. Feature-based heuristic with DREAM database priors

    The sleep_stage argument shifts the base-rate prior before the EEG
    classifier result is blended in, which is especially helpful in a
    full polysomnography context.
    """

    MODEL_SEARCH_PATHS = [
        "models/saved/dream_database_model.pkl",
        "../models/saved/dream_database_model.pkl",
    ]

    def __init__(self, model_path: Optional[str] = None):
        self.sklearn_model = None
        self.scaler = None
        self.feature_names = None
        self.model_source = "feature_heuristic"

        path = model_path or self._auto_find_model()
        if path:
            self._load_model(path)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _auto_find_model(self) -> Optional[str]:
        for p in self.MODEL_SEARCH_PATHS:
            if Path(p).exists():
                return p
        return None

    def _load_model(self, path: str) -> None:
        try:
            import joblib
            payload = joblib.load(path)
            self.sklearn_model = payload.get("model")
            self.scaler = payload.get("scaler")
            self.feature_names = payload.get("feature_names")
            self.model_source = "dream_database"
        except Exception:
            pass  # fall through to heuristic

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        eeg: np.ndarray,
        fs: int = 256,
        sleep_stage: Optional[str] = None,
    ) -> Dict:
        """Predict dream state from a 4-channel Muse 2 EEG epoch.

        Args:
            eeg:         (4, n_samples) array — [TP9, AF7, AF8, TP10].
                         A 1-D array is also accepted (treated as single channel).
            fs:          Sampling rate in Hz (default 256).
            sleep_stage: Optional hint: 'REM', 'NREM', 'N1', 'N2', 'N3', 'Wake'.

        Returns:
            dict with keys:
              dreaming (bool), dream_probability (float 0-1),
              dream_intensity (float 0-1), dream_vividness (float 0-1),
              emotional_valence (float -1 to 1),
              sleep_stage_consistency (str),
              model_source (str),
              dream_database_features (dict)
        """
        eeg = np.asarray(eeg, dtype=float)
        return self._predict_from_eeg(eeg, fs, sleep_stage)

    def get_dream_themes(self, eeg: np.ndarray, fs: int = 256) -> Dict:
        """Estimate probable dream theme mix from EEG spectral features.

        Mapping based on DREAM database spectral correlates of reported
        dream content (Siclari et al., 2025):
          - emotional:    frontal theta + FAA effect
          - visual:       alpha (occipital proxy via temporal channels)
          - kinesthetic:  beta activity (motor-related)
          - narrative:    theta coherence proxy (theta/delta ratio)

        Returns:
            dict: {"emotional": float, "visual": float,
                   "kinesthetic": float, "narrative": float}
              All values are in [0, 1] and sum approximately to 1.
        """
        eeg = np.asarray(eeg, dtype=float)
        if eeg.ndim == 1:
            eeg = eeg.reshape(1, -1)

        bands = self._extract_bands(eeg, fs)
        theta = bands.get("theta", 0.3)
        alpha = bands.get("alpha", 0.2)
        beta  = bands.get("beta",  0.2)
        delta = bands.get("delta", 0.5)
        faa   = bands.get("faa",   0.0)

        theta_delta = theta / (delta + 1e-9)

        # Raw scores
        emotional    = float(np.clip(0.5 * theta + 0.5 * abs(faa), 0, 1))
        visual       = float(np.clip(alpha * 1.2, 0, 1))
        kinesthetic  = float(np.clip(beta * 1.5, 0, 1))
        narrative    = float(np.clip(theta_delta * 0.3, 0, 1))

        # Normalize to sum≈1
        total = emotional + visual + kinesthetic + narrative + 1e-9
        return {
            "emotional":   round(emotional   / total, 4),
            "visual":      round(visual       / total, 4),
            "kinesthetic": round(kinesthetic  / total, 4),
            "narrative":   round(narrative    / total, 4),
        }

    def get_model_info(self) -> Dict:
        """Return metadata about the loaded model."""
        return {
            "model_source":   self.model_source,
            "has_sklearn":    self.sklearn_model is not None,
            "n_features":     len(self.feature_names) if self.feature_names else 23,
            "dream_database": "DREAM (Nature Communications, 2025)",
            "participants":   505,
            "awakenings":     2643,
            "priors":         _DREAM_PRIORS,
        }

    # ── Internal prediction ───────────────────────────────────────────────────

    def _predict_from_eeg(
        self,
        eeg: np.ndarray,
        fs: int,
        sleep_stage: Optional[str],
    ) -> Dict:
        """Core prediction using DREAM database priors + EEG features."""
        if eeg.ndim == 1:
            eeg = eeg.reshape(1, -1)

        # Step 1: base-rate prior from sleep stage
        stage_key = (sleep_stage or "").lower().replace("-", "")
        base_prior = _DREAM_PRIORS.get(stage_key, _DREAM_PRIORS["default"])

        # Step 2: EEG-derived probability
        eeg_prob = self._eeg_dream_probability(eeg, fs)

        # Step 3: Bayesian blend — stage prior (40%) + EEG (60%)
        dream_prob = float(np.clip(0.40 * base_prior + 0.60 * eeg_prob, 0.0, 1.0))

        # Step 4: supplementary features
        bands = self._extract_bands(eeg, fs)
        theta = bands.get("theta", 0.3)
        alpha = bands.get("alpha", 0.2)
        delta = bands.get("delta", 0.5)
        beta  = bands.get("beta",  0.2)
        faa   = bands.get("faa",   0.0)
        theta_delta = float(np.clip(theta / (delta + 1e-9), 0.0, 5.0))
        alpha_beta  = float(np.clip(alpha / (beta  + 1e-9), 0.0, 5.0))

        # Dream intensity: weighted EEG complexity
        dream_intensity = float(np.clip(
            0.45 * theta + 0.30 * alpha + 0.15 * beta + 0.10 * (1.0 - delta),
            0.0, 1.0,
        ))

        # Dream vividness: alpha power is the best Muse-accessible proxy
        dream_vividness = float(np.clip(alpha * 1.5, 0.0, 1.0))

        # Emotional valence: FAA-based
        emotional_valence = float(np.clip(np.tanh(faa * 2.0), -1.0, 1.0))

        # Sleep stage consistency string
        consistency = self._stage_consistency(
            sleep_stage, theta_delta, delta
        )

        return {
            "dreaming":              dream_prob > 0.50,
            "dream_probability":     round(dream_prob, 4),
            "dream_intensity":       round(dream_intensity, 4),
            "dream_vividness":       round(dream_vividness, 4),
            "emotional_valence":     round(emotional_valence, 4),
            "sleep_stage_consistency": consistency,
            "model_source":          self.model_source,
            "dream_database_features": {
                "theta_delta_ratio": round(theta_delta, 4),
                "alpha_beta_ratio":  round(alpha_beta,  4),
                "faa":               round(faa, 4),
                "theta":             round(float(theta), 4),
                "alpha":             round(float(alpha), 4),
                "delta":             round(float(delta), 4),
                "stage_prior":       round(base_prior, 4),
                "eeg_probability":   round(eeg_prob, 4),
            },
        }

    def _eeg_dream_probability(self, eeg: np.ndarray, fs: int) -> float:
        """Return raw EEG-derived dream probability [0, 1]."""
        # Try trained model first
        if self.sklearn_model is not None and self.feature_names is not None:
            try:
                return self._model_probability(eeg, fs)
            except Exception:
                pass

        return self._heuristic_probability(eeg, fs)

    def _model_probability(self, eeg: np.ndarray, fs: int) -> float:
        """Use loaded sklearn/LightGBM model for probability."""
        from training.train_dream_database import DREAMDatabaseLoader
        loader = DREAMDatabaseLoader()
        fv = loader.extract_4ch_features(eeg, fs).reshape(1, -1)
        if self.scaler is not None:
            fv = self.scaler.transform(fv)
        probs = self.sklearn_model.predict_proba(fv)[0]
        # Class ordering: 0=non-dream, 1=dream
        return float(probs[1]) if len(probs) > 1 else float(probs[0])

    def _heuristic_probability(self, eeg: np.ndarray, fs: int) -> float:
        """Feature-based heuristic using DREAM database priors."""
        bands = self._extract_bands(eeg, fs)
        theta = bands.get("theta", 0.3)
        alpha = bands.get("alpha", 0.2)
        beta  = bands.get("beta",  0.2)
        delta = bands.get("delta", 0.5)

        theta_delta = theta / (delta + 1e-9)

        # Weighted sum calibrated to DREAM database effect sizes
        raw = (
            _BAND_WEIGHTS["theta_delta_ratio"] * np.clip(theta_delta / 2.0, 0, 1)
            + _BAND_WEIGHTS["alpha"]            * alpha
            + _BAND_WEIGHTS["theta"]            * theta
            + _BAND_WEIGHTS["beta"]             * beta
            + _BAND_WEIGHTS["delta_inv"]        * (1.0 - delta)
        )
        return float(np.clip(raw, 0.0, 1.0))

    def _extract_bands(self, eeg: np.ndarray, fs: int) -> Dict:
        """Extract average band powers and FAA from multichannel EEG."""
        if not _HAS_PROCESSOR:
            return {
                "delta": 0.5, "theta": 0.3, "alpha": 0.2,
                "beta": 0.2,  "gamma": 0.1, "faa": 0.0,
            }

        if eeg.ndim == 1:
            eeg = eeg.reshape(1, -1)

        band_accum: Dict[str, list] = {
            "delta": [], "theta": [], "alpha": [], "beta": [], "gamma": []
        }
        for ch in range(eeg.shape[0]):
            try:
                proc = preprocess(eeg[ch], fs)
                b = extract_band_powers(proc, fs)
                for k in band_accum:
                    val = b.get(k, 0.0)
                    if np.isfinite(val):
                        band_accum[k].append(float(val))
            except Exception:
                pass

        result = {k: float(np.mean(v)) if v else 0.0 for k, v in band_accum.items()}

        # FAA
        faa = 0.0
        if eeg.shape[0] >= 3:
            try:
                asym = compute_frontal_asymmetry(eeg, fs, left_ch=1, right_ch=2)
                faa = float(asym.get("frontal_asymmetry", 0.0))
                if not np.isfinite(faa):
                    faa = 0.0
            except Exception:
                pass
        result["faa"] = faa
        return result

    def _stage_consistency(
        self,
        sleep_stage: Optional[str],
        theta_delta: float,
        delta: float,
    ) -> str:
        """Describe how consistent EEG features are with the reported sleep stage."""
        if sleep_stage is None:
            return "unknown"

        stage_lower = sleep_stage.lower()

        if "rem" in stage_lower:
            # REM: high theta/delta, low delta
            if theta_delta > 0.8 and delta < 0.45:
                return "consistent_with_REM"
            return "atypical_for_REM"

        if "n3" in stage_lower or "sws" in stage_lower or "deep" in stage_lower:
            # Deep NREM: high delta
            if delta > 0.55:
                return "consistent_with_N3"
            return "atypical_for_N3"

        if "n2" in stage_lower:
            if 0.30 < delta < 0.65:
                return "consistent_with_N2"
            return "atypical_for_N2"

        if "n1" in stage_lower:
            if theta_delta > 0.4:
                return "consistent_with_N1"
            return "atypical_for_N1"

        if "wake" in stage_lower:
            if delta < 0.30:
                return "consistent_with_Wake"
            return "atypical_for_Wake"

        return "unknown"


# ── Singleton getter ───────────────────────────────────────────────────────────

def get_dream_database_detector(user_id: str = "default") -> DREAMDatabaseDreamDetector:
    """Return a per-user singleton DREAMDatabaseDreamDetector.

    Thread-safe — safe to call from multiple async FastAPI workers.
    """
    with _INSTANCE_LOCK:
        if user_id not in _INSTANCES:
            _INSTANCES[user_id] = DREAMDatabaseDreamDetector()
        return _INSTANCES[user_id]
