"""Flow State Detector from EEG signals.

Detects the psychological "flow" state (being "in the zone") using EEG biomarkers.

Scientific basis (updated 2024-2025):
- Transient hypofrontality: flow = increased alpha+theta + decreased beta at AF7/AF8
  (Dietrich 2004, validated on Muse-equivalent positions by Weber et al. 2024)
- Quadratic theta model: flow peaks at MODERATE theta (inverted-U relationship),
  not at maximum theta (which indicates drowsiness)
- Beta asymmetry: flow correlates with symmetric (near-zero) beta asymmetry
  between AF7/AF8 — hemispheric balance during effortless action
- Flow ratio: (alpha + theta) / beta increases during flow states

Reference: Katahira et al. (2018), Ulrich et al. (2014), Nacke et al. (2010),
           Weber et al. (2024), Dietrich (2004) transient hypofrontality theory

Outputs flow probability (0-1) and component scores:
- theta_flow: quadratic theta score (peaks at moderate theta)
- flow_ratio: (alpha+theta)/beta ratio score
- beta_decrease: beta decrease from baseline
- beta_symmetry: beta asymmetry score (symmetric = flow)

Flow intensity levels:
- no_flow: score < 0.3
- shallow: score 0.3 - 0.45
- moderate: score 0.45 - 0.75
- deep: score >= 0.75
"""

import warnings
import numpy as np
from typing import Dict, Optional
from processing.eeg_processor import (
    extract_band_powers, extract_features, preprocess, compute_hjorth_parameters
)


FLOW_STATES = ["no_flow", "shallow", "moderate", "deep"]
FLOW_STATES_BINARY = ["no_flow", "flow"]

# Minimum epoch length in seconds.  Flow state research (Katahira 2018,
# Ulrich 2014) uses 30-60 second analysis windows.  Shorter epochs have
# insufficient spectral resolution for reliable theta/alpha/beta estimation.
MIN_EPOCH_SECONDS = 30.0

# Model accuracy metadata — surface to users for trust calibration
FLOW_MODEL_ACCURACY = 62.86
FLOW_ACCURACY_NOTE = (
    "62.86% cross-validated accuracy — marginally above chance for 4-class "
    "detection; binary (flow/no-flow) mode is recommended for higher reliability"
)


class FlowStateDetector:
    """EEG-based flow state detector using frequency band analysis.

    Accuracy: 62.86% CV (4-class).  Binary flow/no-flow mode achieves
    higher reliability (~70-75% estimated).

    Calibration is required before producing scores.  Call ``calibrate()``
    with 30-60 seconds of resting EEG before the first ``predict()`` call.
    Without calibration, predictions carry a ``calibration_warning``.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_type = "feature-based"
        self.sklearn_model = None
        self.feature_names = None
        self.scaler = None
        self.calibration_data = []
        self.baseline_alpha = None
        self.baseline_beta = None
        self.baseline_theta = None
        self._is_calibrated = False

        if model_path:
            self._load_model(model_path)

    @property
    def is_calibrated(self) -> bool:
        """Whether the detector has been calibrated with resting EEG."""
        return self._is_calibrated

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
        self._is_calibrated = True

    def predict(self, eeg: np.ndarray, fs: float = 256.0, binary: bool = True) -> Dict:
        """Detect flow state from EEG signal.

        Accepts either a 1D single-channel signal or a 2D multichannel array
        (shape: n_channels x n_samples). When multichannel, channels are assumed
        to follow BrainFlow Muse 2 order: [TP9, AF7, AF8, TP10].

        Args:
            eeg: Raw EEG signal (1D or 2D array).
            fs: Sampling frequency in Hz (default 256.0).
            binary: If True (default), return binary flow/no-flow classification
                instead of 4-state classification.  Binary mode is recommended
                for higher reliability (~70-75% vs 62.86%).

        Returns:
            Dict with 'state', 'flow_score', 'confidence',
            'components' (theta_flow, flow_ratio, beta_decrease, beta_symmetry),
            'flow_intensity', 'band_powers', 'model_accuracy', 'accuracy_note'.
            May also include 'calibration_warning' and/or 'epoch_length_warning'.
        """
        # --- Epoch length validation ---
        n_samples = eeg.shape[-1]  # last axis is always samples
        epoch_seconds = n_samples / fs
        epoch_warning = None
        if epoch_seconds < MIN_EPOCH_SECONDS:
            epoch_warning = (
                f"Epoch length {epoch_seconds:.1f}s is below the recommended "
                f"minimum of {MIN_EPOCH_SECONDS:.0f}s for flow state detection; "
                f"results may be unreliable"
            )
            warnings.warn(epoch_warning, stacklevel=2)

        # --- Calibration enforcement ---
        calibration_warning = None
        if not self._is_calibrated:
            calibration_warning = (
                "Flow detector is not calibrated. Call calibrate() with "
                "30-60 seconds of resting EEG for reliable results."
            )
            warnings.warn(calibration_warning, stacklevel=2)

        if self.sklearn_model is not None:
            try:
                result = self._predict_sklearn(eeg, fs)
                result = self._add_metadata(result, binary, calibration_warning, epoch_warning)
                return result
            except Exception:
                pass  # fall through to feature-based

        # Handle multichannel input
        channels = eeg if eeg.ndim == 2 else None
        signal = eeg[0] if eeg.ndim == 2 else eeg

        processed = preprocess(signal, fs)
        bands = extract_band_powers(processed, fs)

        alpha = bands.get("alpha", 0)
        beta = bands.get("beta", 0)
        theta = bands.get("theta", 0)

        # Use calibration baselines if available
        base_beta = self.baseline_beta or 0.15

        # === Flow Component Scores (Validated Biomarkers) ===

        # 1. Quadratic Theta Score (35% weight)
        # Flow occurs at MODERATE theta — inverted-U relationship.
        # Too low theta = not engaged; too high theta = drowsiness.
        # Band powers are already relative (0-1). Theta in real EEG typically
        # sits in 0.05-0.30 relative power; during drowsiness can reach 0.50+.
        # Normalize to [0, 1] over 0-0.60 range; optimal at 0.4 normalized.
        # Divisor 0.30 gives a gentle inverted-U: peaks around 0.20-0.25
        # relative theta, still scores well at 0.10-0.40, penalizes 0.50+.
        optimal_theta = 0.4
        theta_normalized = float(np.clip(theta / 0.60, 0, 1))
        theta_flow_score = float(np.clip(
            1.0 - (theta_normalized - optimal_theta) ** 2 / 0.30,
            0, 1
        ))

        # 2. Flow Ratio: (alpha + theta) / beta (30% weight)
        # Transient hypofrontality: flow = increased alpha+theta + decreased beta.
        # Higher ratio = more flow-like state.
        alpha_theta_sum = alpha + theta
        flow_ratio_raw = alpha_theta_sum / (beta + 1e-10)
        # Typical non-flow ratio ~1.5-2.0; flow state pushes to 3.0+
        # Sigmoid mapping: ratio of 3.0 → ~0.73, ratio of 5.0 → ~0.95
        flow_ratio_score = float(np.clip(
            1.0 / (1.0 + np.exp(-1.5 * (flow_ratio_raw - 2.5))),
            0, 1
        ))

        # 3. Beta Decrease from Baseline (20% weight)
        # During flow, beta decreases (reduced self-monitoring / inner critic).
        # If no baseline available, use a moderate score (0.4) as default.
        if self.baseline_beta is not None:
            beta_decrease_ratio = (base_beta - beta) / (base_beta + 1e-10)
            # Positive = beta decreased (good for flow); negative = beta increased
            beta_decrease_score = float(np.clip(
                0.5 + beta_decrease_ratio * 1.5,
                0, 1
            ))
        else:
            beta_decrease_score = 0.4  # neutral default without baseline

        # 4. Beta Asymmetry (15% weight)
        # Flow correlates with symmetric (near-zero) beta asymmetry between AF7/AF8.
        # Hemispheric balance = effortless action.
        beta_symmetry_score = 0.5  # default for single-channel
        if channels is not None and channels.shape[0] >= 3:
            # BrainFlow Muse 2: ch1=AF7 (left), ch2=AF8 (right)
            af7_processed = preprocess(channels[1], fs)
            af8_processed = preprocess(channels[2], fs)
            af7_bands = extract_band_powers(af7_processed, fs)
            af8_bands = extract_band_powers(af8_processed, fs)
            af7_beta = af7_bands.get("beta", 0)
            af8_beta = af8_bands.get("beta", 0)
            denom = af8_beta + af7_beta
            if denom > 1e-10:
                beta_asym = abs(af8_beta - af7_beta) / denom
            else:
                beta_asym = 0.0
            # Near zero asymmetry → high score; high asymmetry → low score
            # beta_asym ranges from 0 (symmetric) to 1 (maximally asymmetric)
            beta_symmetry_score = float(np.clip(1.0 - beta_asym * 2.0, 0, 1))

        # === Overall Flow Score ===
        # Weighted combination per validated biomarker importance
        flow_score = float(np.clip(
            0.35 * theta_flow_score +
            0.30 * flow_ratio_score +
            0.20 * beta_decrease_score +
            0.15 * beta_symmetry_score,
            0, 1
        ))

        # === Classify Flow Intensity Level ===
        if flow_score >= 0.75:
            state_idx = 3   # deep
            intensity = "deep"
        elif flow_score >= 0.45:
            state_idx = 2   # moderate
            intensity = "moderate"
        elif flow_score >= 0.3:
            state_idx = 1   # shallow
            intensity = "shallow"
        else:
            state_idx = 0   # no_flow
            intensity = "none"

        # Confidence based on how clearly the score falls into a category
        thresholds = [0.0, 0.3, 0.45, 0.75, 1.0]
        mid = (thresholds[state_idx] + thresholds[state_idx + 1]) / 2
        distance_to_mid = abs(flow_score - mid)
        range_size = thresholds[state_idx + 1] - thresholds[state_idx]
        confidence = float(np.clip(
            1.0 - (distance_to_mid / (range_size / 2 + 1e-10)),
            0.3, 0.95
        ))

        result = {
            "state": FLOW_STATES[state_idx],
            "state_index": state_idx,
            "flow_score": round(flow_score, 3),
            "confidence": round(confidence, 3),
            "flow_intensity": intensity,
            "components": {
                "theta_flow": round(theta_flow_score, 3),
                "flow_ratio": round(flow_ratio_score, 3),
                "beta_decrease": round(beta_decrease_score, 3),
                "beta_symmetry": round(beta_symmetry_score, 3),
            },
            "band_powers": bands,
        }
        return self._add_metadata(result, binary, calibration_warning, epoch_warning)

    def _predict_sklearn(self, eeg: np.ndarray, fs: float) -> Dict:
        """Sklearn model inference using extracted features."""
        signal = eeg[0] if eeg.ndim == 2 else eeg
        processed = preprocess(signal, fs)
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
            "flow_intensity": FLOW_STATES[state_idx],
            "components": {
                "absorption": round(float(probs[2] + probs[3]) / 2, 3) if len(probs) > 3 else 0.5,
                "effortlessness": round(float(bands.get("alpha", 0.3)), 3),
                "focus_quality": round(float(bands.get("beta", 0.2)), 3),
                "time_distortion": round(float(bands.get("theta", 0.15)), 3),
            },
            "band_powers": bands,
        }

    def _add_metadata(
        self,
        result: Dict,
        binary: bool,
        calibration_warning: Optional[str],
        epoch_warning: Optional[str],
    ) -> Dict:
        """Add validation metadata, warnings, and optional binary mode to result.

        Centralises the accuracy note, calibration check, epoch-length check,
        and binary flow/no-flow reclassification so that all code paths
        (feature-based and sklearn) behave consistently.
        """
        # Accuracy transparency
        result["model_accuracy"] = FLOW_MODEL_ACCURACY
        result["accuracy_note"] = FLOW_ACCURACY_NOTE

        # Calibration warning
        if calibration_warning:
            result["calibration_warning"] = calibration_warning

        # Epoch length warning
        if epoch_warning:
            result["epoch_length_warning"] = epoch_warning

        # Binary mode: collapse 4 states into flow / no-flow
        if binary:
            flow_score = result["flow_score"]
            # Threshold at 0.45 — same as the moderate/shallow boundary
            is_flow = flow_score >= 0.45
            detailed_state = result["state"]  # preserve 4-class label
            result["state"] = "flow" if is_flow else "no_flow"
            result["state_index"] = 1 if is_flow else 0
            result["is_flow"] = is_flow
            result["in_flow"] = is_flow  # alias for frontend compatibility
            result["flow_intensity"] = "flow" if is_flow else "none"
            result["detailed_state"] = detailed_state
            result["binary_mode"] = True
            result["model_type"] = "flow_binary"
        else:
            result["is_flow"] = result.get("flow_score", 0) >= 0.45
            result["in_flow"] = result["is_flow"]
            result["binary_mode"] = False
            result["model_type"] = "flow_4class"

        return result
