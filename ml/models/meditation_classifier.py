"""Meditation Depth Classifier from EEG signals.

Classifies meditation depth into 3 levels (reduced from 5 for better cross-subject accuracy):
  0: relaxed     — eyes-closed rest, alpha dominant, minimal meditation depth
  1: meditating  — sustained theta increase, alpha stabilization, reduced mental chatter
  2: deep        — strong theta dominance, beta suppressed, deep absorption

Scientific basis:
- Meditation increases frontal midline theta (Kubota et al., 2001)
- Alpha coherence increases during focused meditation (Travis & Shear, 2010)
- Advanced meditators show gamma power 25x higher than novices (Lutz et al., 2004)
- Theta/gamma coupling correlates with meditation expertise
- Default mode network (DMN) suppression = reduced delta/low-alpha

Meditation traditions mapped to EEG signatures:
- Focused attention (Shamatha): high alpha, moderate theta
- Open monitoring (Vipassana): high theta, wide alpha
- Non-dual (Dzogchen/Sahaj): gamma bursts, theta-gamma coupling
- Loving-kindness (Metta): alpha + gamma, high coherence

Reference: Lutz et al. (2004), Cahn & Polich (2006), Travis & Shear (2010)
"""

import numpy as np
from typing import Dict, Optional
from processing.eeg_processor import (
    extract_band_powers, extract_features, preprocess, compute_hjorth_parameters,
    spectral_entropy
)

MEDITATION_DEPTHS = ["relaxed", "meditating", "deep"]


class MeditationClassifier:
    """EEG-based meditation depth classifier."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_type = "feature-based"
        self.sklearn_model = None
        self.feature_names = None
        self.scaler = None
        self.baseline_alpha = None
        self.baseline_theta = None
        self.baseline_gamma = None
        # Track depth over session
        self._depth_history = []

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        if model_path.endswith(".pkl"):
            try:
                import joblib
                data = joblib.load(model_path)
                self.sklearn_model = data["model"]
                self.feature_names = data["feature_names"]
                self.scaler = data.get("scaler")
                self.model_type = "sklearn"
            except Exception:
                pass

    def calibrate(self, resting_eeg: np.ndarray, fs: float = 256.0):
        """Calibrate with eyes-closed resting EEG baseline."""
        processed = preprocess(resting_eeg, fs)
        bands = extract_band_powers(processed, fs)
        self.baseline_alpha = bands.get("alpha", 0.25)
        self.baseline_theta = bands.get("theta", 0.15)
        self.baseline_gamma = bands.get("gamma", 0.05)

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Classify meditation depth from EEG.

        Returns:
            Dict with 'depth', 'depth_index', 'meditation_score' (0-1),
            'confidence', 'tradition_match', and component scores.
        """
        if self.sklearn_model is not None:
            return self._predict_sklearn(eeg, fs)

        processed = preprocess(eeg, fs)
        bands = extract_band_powers(processed, fs)
        hjorth = compute_hjorth_parameters(processed)
        se = spectral_entropy(processed, fs)

        alpha = bands.get("alpha", 0)
        beta = bands.get("beta", 0)
        theta = bands.get("theta", 0)
        delta = bands.get("delta", 0)
        gamma = bands.get("gamma", 0)

        base_alpha = self.baseline_alpha or 0.25
        base_theta = self.baseline_theta or 0.15
        base_gamma = self.baseline_gamma or 0.05

        # === Meditation Components ===

        # 1. Alpha Stabilization (relaxation foundation)
        alpha_ratio = alpha / (base_alpha + 1e-10)
        alpha_stability = float(np.clip(np.tanh(alpha_ratio - 0.5), 0, 1))

        # 2. Theta Elevation (deepening meditation)
        theta_increase = theta / (base_theta + 1e-10) - 1.0
        theta_depth = float(np.clip(np.tanh(theta_increase), 0, 1))

        # 3. Beta Quieting (reduced mental chatter)
        beta_quiet = float(np.clip(1.0 - np.tanh(beta * 5), 0, 1))

        # 4. Delta Management (avoiding sleep — some delta is OK)
        # Too much delta = sleep, not meditation
        delta_balance = float(np.clip(1.0 - np.tanh(delta * 2 - 0.5), 0, 1))

        # 5. Gamma Transcendence (advanced meditation signature)
        gamma_ratio = gamma / (base_gamma + 1e-10)
        gamma_transcendence = float(np.clip(np.tanh(gamma_ratio - 1.5), 0, 1))

        # 6. Theta-Gamma Coupling (deep meditation marker)
        tg_coupling = theta * gamma * 100
        coupling_score = float(np.clip(np.tanh(tg_coupling), 0, 1))

        # 7. Spectral Narrowing (focused meditation = lower entropy)
        spectral_focus = float(np.clip(1.0 - se, 0, 1))

        # 8. Signal Calmness (low Hjorth activity = calm brain)
        activity = hjorth.get("activity", 0.01) if isinstance(hjorth, dict) else 0.01
        calmness = float(np.clip(1.0 - np.tanh(activity * 5), 0, 1))

        # === Overall Meditation Score ===
        meditation_score = float(np.clip(
            0.20 * alpha_stability +
            0.20 * theta_depth +
            0.15 * beta_quiet +
            0.10 * delta_balance +
            0.15 * gamma_transcendence +
            0.10 * coupling_score +
            0.05 * spectral_focus +
            0.05 * calmness,
            0, 1
        ))

        # Track depth
        self._depth_history.append(meditation_score)
        if len(self._depth_history) > 120:
            self._depth_history = self._depth_history[-120:]

        # Session duration effect (meditation deepens over time)
        session_minutes = len(self._depth_history) * 0.5  # ~30s per epoch
        session_bonus = float(np.clip(session_minutes / 20.0 * 0.1, 0, 0.1))

        adjusted_score = float(np.clip(meditation_score + session_bonus, 0, 1))

        # === Classify Depth (3-class) ===
        if adjusted_score >= 0.55:
            depth_idx = 2  # deep
        elif adjusted_score >= 0.25:
            depth_idx = 1  # meditating
        else:
            depth_idx = 0  # relaxed

        # === Tradition Match ===
        tradition_scores = {
            "focused_attention": float(alpha_stability * 0.5 + beta_quiet * 0.3 + spectral_focus * 0.2),
            "open_monitoring": float(theta_depth * 0.4 + alpha_stability * 0.3 + calmness * 0.3),
            "non_dual": float(gamma_transcendence * 0.4 + coupling_score * 0.3 + theta_depth * 0.3),
            "loving_kindness": float(alpha_stability * 0.3 + gamma_transcendence * 0.3 + calmness * 0.4),
        }
        best_tradition = max(tradition_scores, key=tradition_scores.get)

        # Confidence (3-class thresholds: 0.0 / 0.25 / 0.55 / 1.0)
        thresholds = [0.0, 0.25, 0.55, 1.0]
        mid = (thresholds[depth_idx] + thresholds[depth_idx + 1]) / 2
        dist = abs(adjusted_score - mid)
        range_size = thresholds[depth_idx + 1] - thresholds[depth_idx]
        confidence = float(np.clip(1.0 - dist / (range_size / 2 + 1e-10), 0.3, 0.95))

        return {
            "depth": MEDITATION_DEPTHS[depth_idx],
            "depth_index": depth_idx,
            "meditation_score": round(adjusted_score, 3),
            "confidence": round(confidence, 3),
            "tradition_match": best_tradition,
            "tradition_scores": {k: round(v, 3) for k, v in tradition_scores.items()},
            "session_minutes": round(session_minutes, 1),
            "components": {
                "alpha_stability": round(alpha_stability, 3),
                "theta_depth": round(theta_depth, 3),
                "beta_quiet": round(beta_quiet, 3),
                "delta_balance": round(delta_balance, 3),
                "gamma_transcendence": round(gamma_transcendence, 3),
                "theta_gamma_coupling": round(coupling_score, 3),
                "spectral_focus": round(spectral_focus, 3),
                "calmness": round(calmness, 3),
            },
            "band_powers": bands,
        }

    def _predict_sklearn(self, eeg: np.ndarray, fs: float) -> Dict:
        processed = preprocess(eeg, fs)
        features = extract_features(processed, fs)
        bands = extract_band_powers(processed, fs)
        fv = np.array([features.get(k, 0.0) for k in self.feature_names]).reshape(1, -1)
        if self.scaler is not None:
            fv = self.scaler.transform(fv)

        probs = self.sklearn_model.predict_proba(fv)[0]
        depth_idx = int(np.argmax(probs))

        return {
            "depth": MEDITATION_DEPTHS[min(depth_idx, 4)],
            "depth_index": min(depth_idx, 4),
            "meditation_score": round(float(np.dot(probs, np.linspace(0, 1, len(probs)))), 3),
            "confidence": round(float(probs[depth_idx]), 3),
            "tradition_match": "focused_attention",
            "tradition_scores": {},
            "session_minutes": 0.0,
            "components": {},
            "band_powers": bands,
        }

    def get_session_stats(self) -> Dict:
        """Get meditation session statistics."""
        if not self._depth_history:
            return {"n_epochs": 0}

        scores = np.array(self._depth_history)
        return {
            "n_epochs": len(scores),
            "session_minutes": round(len(scores) * 0.5, 1),
            "avg_depth": round(float(np.mean(scores)), 3),
            "max_depth": round(float(np.max(scores)), 3),
            "time_in_deep": round(float(np.mean(scores >= 0.6)) * 100, 1),
            "deepening_trend": round(float(
                np.mean(scores[-10:]) - np.mean(scores[:10])
            ) if len(scores) >= 20 else 0.0, 3),
        }
