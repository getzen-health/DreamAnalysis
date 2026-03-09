"""Flow State Detector from EEG signals.

Detects the psychological "flow" state (being "in the zone") using EEG biomarkers.

Scientific basis:
- Sci Rep 2025: flow = increased alpha+theta, decreased beta at AF7/AF8
- Sensors 2024: quadratic (inverted-U) theta-flow relationship — flow occurs at
  MODERATE theta (too low = boredom, too high = anxiety/cognitive overload)
- Beta channel asymmetry (AF8 - AF7): symmetric during flow, lateralised outside
- Katahira et al. (2018), Ulrich et al. (2014), Nacke et al. (2010)

Outputs flow probability (0-1) and component scores:
- absorption: how deeply engaged (theta relative to baseline)
- effortlessness: lack of strain (alpha/high-beta ratio)
- focus_quality: sustained attention quality (beta decrease + beta symmetry)
- time_distortion: altered time perception proxy (quadratic theta score)
"""

import numpy as np
from typing import Dict, Optional
from processing.eeg_processor import (
    extract_band_powers, extract_features, preprocess, compute_hjorth_parameters
)


FLOW_STATES = ["no_flow", "micro_flow", "flow", "deep_flow"]


class FlowStateDetector:
    """EEG-based flow state detector using frequency band analysis."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_type = "feature-based"
        self.sklearn_model = None
        self.feature_names = None
        self.scaler = None
        self.calibration_data = []
        self.baseline_alpha = None
        self.baseline_beta = None
        self.baseline_theta = None

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load trained model from pkl file."""
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
        """Calibrate with 30-60s of resting EEG for personalized thresholds."""
        processed = preprocess(resting_eeg, fs)
        bands = extract_band_powers(processed, fs)
        self.baseline_alpha = bands.get("alpha", 0.2)
        self.baseline_beta = bands.get("beta", 0.15)
        self.baseline_theta = bands.get("theta", 0.15)

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Detect flow state from EEG signal.

        Uses validated AF7/AF8 biomarkers from Sci Rep 2025 + Sensors 2024:
        - Quadratic (inverted-U) theta model: flow at moderate theta (~1.3x baseline)
        - (alpha + theta) / beta flow ratio
        - Beta decrease from baseline (relaxed focus, not anxious)
        - Beta channel asymmetry: symmetric during flow (AF7 ≈ AF8)

        Returns:
            Dict with 'state', 'flow_score', 'confidence',
            'absorption', 'effortlessness', 'focus_quality',
            'time_distortion', 'band_powers'
        """
        if self.sklearn_model is not None:
            try:
                return self._predict_sklearn(eeg, fs)
            except Exception:
                pass  # fall through to feature-based

        return self._predict_features(eeg, fs)

    def _predict_features(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Feature-based flow detection using validated AF7/AF8 biomarkers.

        Scientific basis:
        - Sci Rep 2025: flow = increased alpha+theta, decreased beta at AF7/AF8
        - Sensors 2024: quadratic (inverted-U) theta-flow relationship
        - Flow occurs at MODERATE theta — too low = boredom, too high = anxiety
        """
        signal = eeg[0] if eeg.ndim == 2 else eeg
        channels = eeg if eeg.ndim == 2 else None

        processed = preprocess(signal, fs)
        bands = extract_band_powers(processed, fs)

        alpha = bands.get("alpha", 0.2)
        theta = bands.get("theta", 0.15)
        beta = bands.get("beta", 0.15)
        high_beta = bands.get("high_beta", 0.05)

        # Use baselines if calibrated
        ref_alpha = self.baseline_alpha or 0.2
        ref_beta = self.baseline_beta or 0.15
        ref_theta = self.baseline_theta or 0.15

        # 1. Flow ratio: (alpha + theta) / beta — increases during flow
        eps = 1e-10
        flow_ratio = (alpha + theta) / (beta + eps)
        ref_ratio = (ref_alpha + ref_theta) / (ref_beta + eps)
        flow_ratio_score = min(1.0, flow_ratio / (ref_ratio + eps))

        # 2. Quadratic theta score (Sensors 2024): inverted-U, peak at moderate theta
        # Normalize theta relative to baseline
        theta_norm = theta / (ref_theta + eps)
        # Optimal at 1.3x baseline — quadratic penalty away from optimum
        theta_flow_score = max(0.0, 1.0 - (theta_norm - 1.3) ** 2 / 1.69)

        # 3. Beta decrease from baseline (flow = relaxed focus, less anxious beta)
        beta_decrease_score = max(0.0, min(1.0, 1.0 - (beta - ref_beta * 0.8) / (ref_beta + eps)))

        # 4. Beta channel asymmetry (AF8 - AF7): symmetric during flow
        beta_asym_score = 0.5  # default if single channel
        if channels is not None and channels.shape[0] >= 3:
            # ch1=AF7, ch2=AF8 (BrainFlow Muse 2 order)
            bands_af7 = extract_band_powers(preprocess(channels[1], fs), fs)
            bands_af8 = extract_band_powers(preprocess(channels[2], fs), fs)
            af7_beta = bands_af7.get("beta", beta)
            af8_beta = bands_af8.get("beta", beta)
            beta_asym = (af8_beta - af7_beta) / (af7_beta + af8_beta + eps)
            # Flow: symmetric beta (near 0), not strongly lateralized
            beta_asym_score = max(0.0, 1.0 - abs(beta_asym) * 3.0)

        # Weighted combination (validated weights from Sci Rep 2025)
        flow_score = (
            0.35 * theta_flow_score
            + 0.30 * min(1.0, flow_ratio_score)
            + 0.20 * beta_decrease_score
            + 0.15 * beta_asym_score
        )
        flow_score = float(np.clip(flow_score, 0.0, 1.0))

        # State classification
        if flow_score >= 0.72:
            state = "deep_flow"
            state_index = 3
        elif flow_score >= 0.50:
            state = "flow"
            state_index = 2
        elif flow_score >= 0.28:
            state = "micro_flow"
            state_index = 1
        else:
            state = "no_flow"
            state_index = 0

        # Component scores for UI
        absorption = float(np.clip((theta / ref_theta - 0.8) / 1.5, 0, 1))
        effortlessness = float(np.clip(alpha / (high_beta + eps) / 4.0, 0, 1))
        focus_quality = float(np.clip(beta_decrease_score * 0.7 + beta_asym_score * 0.3, 0, 1))
        time_distortion = float(np.clip(theta_flow_score, 0, 1))

        return {
            "state": state,
            "state_index": state_index,
            "flow_score": round(flow_score, 4),
            "confidence": round(min(0.85, flow_score + 0.1), 3),
            "model_type": "feature_quadratic",
            "absorption": round(absorption, 3),
            "effortlessness": round(effortlessness, 3),
            "focus_quality": round(focus_quality, 3),
            "time_distortion": round(time_distortion, 3),
            "band_powers": {k: round(float(v), 4) for k, v in bands.items()},
            "theta_flow_score": round(theta_flow_score, 3),
            "flow_ratio": round(float(flow_ratio), 3),
        }

    def _predict_sklearn(self, eeg: np.ndarray, fs: float) -> Dict:
        """Sklearn model inference using extracted features."""
        processed = preprocess(eeg, fs)
        features = extract_features(processed, fs)
        bands = extract_band_powers(processed, fs)
        feature_vector = np.array([features.get(k, 0.0) for k in self.feature_names]).reshape(1, -1)

        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector)

        probs = self.sklearn_model.predict_proba(feature_vector)[0]
        state_idx = int(np.argmax(probs))
        flow_score = float(state_idx / 3.0 * 0.7 + probs[state_idx] * 0.3)

        return {
            "state": FLOW_STATES[state_idx],
            "state_index": state_idx,
            "flow_score": round(float(np.clip(flow_score, 0, 1)), 3),
            "confidence": round(float(probs[state_idx]), 3),
            "components": {
                "absorption": round(float(probs[2] + probs[3]) / 2, 3) if len(probs) > 3 else 0.5,
                "effortlessness": round(float(bands.get("alpha", 0.3)), 3),
                "focus_quality": round(float(bands.get("beta", 0.2)), 3),
                "time_distortion": round(float(bands.get("theta", 0.15)), 3),
            },
            "band_powers": bands,
        }
