"""Attention / Focus Classifier from EEG signals.

Classifies attention state into 4 levels:
  0: distracted   — mind-wandering, unfocused, high default mode network
  1: passive      — receiving information but not actively processing
  2: focused      — actively attending, sustained concentration
  3: hyperfocused  — deep concentration, tunnel vision, flow-adjacent

Distinguished from emotion-derived "focus_index" by using attention-specific
biomarkers rather than emotional valence/arousal mapping.

Scientific basis:
- Sustained attention: frontal-parietal beta coherence (Sauseng et al., 2005)
- Mind-wandering: default mode network alpha increase (Christoff et al., 2009)
- Attention lapses: theta/alpha bursts predict errors (O'Connell et al., 2009)
- Focused attention: high beta/theta ratio (Clarke et al., 2001)
- Used clinically for ADHD assessment: theta/beta ratio > 4.5 = attention deficit

Reference: Lubar (1991), Monastra et al. (2005), Arns et al. (2013)
"""

import numpy as np
from typing import Dict, Optional
from processing.eeg_processor import (
    extract_band_powers, extract_features, preprocess, compute_hjorth_parameters,
    spectral_entropy
)

ATTENTION_STATES = ["distracted", "passive", "focused", "hyperfocused"]


class AttentionClassifier:
    """EEG-based attention state classifier."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_type = "feature-based"
        self.sklearn_model = None
        self.feature_names = None
        self.scaler = None
        self.baseline_beta = None
        self.baseline_theta = None
        # Track attention stability over time
        self._attention_history = []

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

    def calibrate(self, focused_eeg: np.ndarray, fs: float = 256.0):
        """Calibrate with focused-state EEG for personalized thresholds."""
        signal = focused_eeg[0] if focused_eeg.ndim == 2 else focused_eeg
        processed = preprocess(signal, fs)
        bands = extract_band_powers(processed, fs)
        self.baseline_beta = bands.get("beta", 0.2)
        self.baseline_theta = bands.get("theta", 0.15)

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Classify attention state from EEG.

        Returns:
            Dict with 'state', 'attention_score' (0-1), 'confidence',
            'theta_beta_ratio', and component scores.
        """
        if self.sklearn_model is not None:
            try:
                return self._predict_sklearn(eeg, fs)
            except Exception:
                pass  # fall through to feature-based

        signal = eeg[0] if eeg.ndim == 2 else eeg
        processed = preprocess(signal, fs)
        bands = extract_band_powers(processed, fs)
        hjorth = compute_hjorth_parameters(processed)
        se = spectral_entropy(processed, fs)

        alpha = bands.get("alpha", 0)
        beta = bands.get("beta", 0)
        theta = bands.get("theta", 0)
        delta = bands.get("delta", 0)
        # Gamma (30-100 Hz) excluded: on Muse 2 AF7/AF8 it is predominantly EMG
        # artifact (jaw/forehead muscle). Using it would falsely inflate attention scores.
        low_beta = bands.get("low_beta", 0)   # 12-20 Hz: working memory and attention

        base_beta = self.baseline_beta or 0.2
        base_theta = self.baseline_theta or 0.15

        # === Attention Components ===

        # 1. Theta/Beta Ratio (ADHD gold standard, lower = better attention)
        theta_beta_ratio = theta / (beta + 1e-10)
        # Log-transform TBR for better linearity (TBR typically spans 0.5-8+)
        tbr_log = float(np.log1p(theta_beta_ratio))
        # Clinical threshold: TBR > 4.5 = attention deficit, ~log(5.5)≈1.7
        tbr_score = float(np.clip(1.0 - np.tanh(tbr_log / 1.5), 0, 1))

        # 2. Low-Beta Engagement (12-20 Hz: working memory and attentional control)
        # Using low_beta specifically is more accurate than broad beta for attention
        low_beta_increase = low_beta / max(base_beta * 0.6, 1e-10)  # low_beta ≈ 60% of beta
        beta_engagement = float(np.clip(np.tanh(low_beta_increase - 0.5), 0, 1))

        # 3. Alpha Idle penalty (high alpha = mind-wandering / default mode)
        # Moderated: alpha can also indicate relaxed readiness, so penalty is softer
        alpha_idle = float(np.clip(np.tanh(alpha * 3 - 0.3), 0, 1))
        alpha_focus_penalty = 1.0 - 0.8 * alpha_idle  # softer than (1 - alpha_idle)

        # 4. Spectral Concentration (focused = narrow-band beta, not broadband)
        # Low spectral entropy = concentrated in fewer bands = focused
        spectral_focus = float(np.clip(1.0 - se, 0, 1))

        # 5. Signal Stability (Hjorth mobility — low = stable = sustained attention)
        mobility = hjorth.get("mobility", 0.5) if isinstance(hjorth, dict) else 0.5
        signal_stability = float(np.clip(1.0 - np.tanh(mobility * 2), 0, 1))

        # 6. (Gamma processing removed — gamma is EMG artifact on Muse 2 AF7/AF8,
        #     not neural signal. It would inflate attention scores spuriously.)

        # 7. Delta Penalty (high delta = drowsiness / deep fatigue; incompatible with attention)
        # A high-delta EEG is almost certainly not focused — apply a downward pressure
        delta_penalty = float(np.clip(delta * 3, 0, 0.4))  # cap penalty at 0.4

        # === Overall Attention Score ===
        # Gamma processing removed (EMG artifact on Muse 2).
        # Its 0.09 weight redistributed: spectral_focus 0.13→0.18, signal_stability 0.10→0.14.
        attention_score = float(np.clip(
            0.25 * tbr_score +
            0.20 * beta_engagement +
            0.18 * alpha_focus_penalty +
            0.18 * spectral_focus +
            0.14 * signal_stability -
            0.05 * delta_penalty,
            0, 1
        ))

        # Track history for stability metric — extended to 40 for more reliable variance
        self._attention_history.append(attention_score)
        if len(self._attention_history) > 40:
            self._attention_history = self._attention_history[-40:]

        # Attention stability (low variance = sustained; use last 15 for responsiveness)
        if len(self._attention_history) >= 5:
            attention_stability = float(1.0 - min(1.0, np.std(self._attention_history[-15:]) * 5))
        else:
            attention_stability = 0.5

        # === Classify State ===
        if attention_score >= 0.75:
            state_idx = 3  # hyperfocused
        elif attention_score >= 0.50:
            state_idx = 2  # focused
        elif attention_score >= 0.25:
            state_idx = 1  # passive
        else:
            state_idx = 0  # distracted

        # Confidence
        thresholds = [0.0, 0.25, 0.50, 0.75, 1.0]
        mid = (thresholds[state_idx] + thresholds[state_idx + 1]) / 2
        dist = abs(attention_score - mid)
        range_size = thresholds[state_idx + 1] - thresholds[state_idx]
        confidence = float(np.clip(1.0 - dist / (range_size / 2 + 1e-10), 0.35, 0.95))

        return {
            "state": ATTENTION_STATES[state_idx],
            "state_index": state_idx,
            "attention_score": round(attention_score, 3),
            "confidence": round(confidence, 3),
            "theta_beta_ratio": round(float(theta_beta_ratio), 3),
            "attention_stability": round(attention_stability, 3),
            "components": {
                "tbr_score": round(tbr_score, 3),
                "beta_engagement": round(beta_engagement, 3),
                "alpha_focus_penalty": round(alpha_focus_penalty, 3),
                "spectral_focus": round(spectral_focus, 3),
                "signal_stability": round(signal_stability, 3),
                "delta_penalty": round(delta_penalty, 3),
            },
            "band_powers": bands,
        }

    def _predict_sklearn(self, eeg: np.ndarray, fs: float) -> Dict:
        signal = eeg[0] if eeg.ndim == 2 else eeg
        processed = preprocess(signal, fs)
        features = extract_features(processed, fs)
        bands = extract_band_powers(processed, fs)
        fv = np.array([features.get(k, 0.0) for k in self.feature_names]).reshape(1, -1)
        if self.scaler is not None:
            fv = self.scaler.transform(fv)

        probs = self.sklearn_model.predict_proba(fv)[0]
        state_idx = int(np.argmax(probs))
        attention_score = float(
            probs[1] * 0.33 + probs[2] * 0.66 + probs[3] * 1.0
        ) if len(probs) >= 4 else float(probs[state_idx])

        tbr = float(bands.get("theta", 0.15) / (bands.get("beta", 0.01) + 1e-10))

        return {
            "state": ATTENTION_STATES[min(state_idx, 3)],
            "state_index": min(state_idx, 3),
            "attention_score": round(attention_score, 3),
            "confidence": round(float(probs[state_idx]), 3),
            "theta_beta_ratio": round(tbr, 3),
            "attention_stability": 0.5,
            "components": {
                "tbr_score": round(float(1.0 - np.tanh(tbr / 3.0)), 3),
                "beta_engagement": round(float(bands.get("beta", 0.15)), 3),
                "alpha_focus_penalty": round(float(1.0 - bands.get("alpha", 0.2)), 3),
                "spectral_focus": 0.5,
                "signal_stability": 0.5,
            },
            "band_powers": bands,
        }
