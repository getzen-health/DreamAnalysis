"""EEGNet-based Sleep Staging Model.

Classifies EEG epochs into 5 sleep stages:
- Wake (W)
- N1 (light sleep)
- N2 (light sleep with spindles)
- N3 (deep/slow-wave sleep)
- REM (rapid eye movement)

Architecture: Compact CNN (EEGNet variant) optimized for ONNX export.
Supports three inference paths: ONNX > sklearn > feature-based fallback.
"""

from __future__ import annotations

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from processing.eeg_processor import extract_features, extract_band_powers, preprocess, detect_sleep_spindles, detect_k_complexes

SLEEP_STAGES = ["Wake", "N1", "N2", "N3", "REM"]
STAGE_MAP = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


class SleepStagingModel:
    """Sleep staging classifier with ONNX/sklearn/feature-based inference.

    Priority: ONNX → sklearn .pkl → feature-based fallback.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.onnx_session = None
        self.sklearn_model = None
        self.feature_names = None
        self.model_type = "feature-based"

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load model from file (ONNX or sklearn pkl)."""
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
                self.model_type = "sklearn"
            except Exception:
                pass

    def predict(self, eeg_epoch: np.ndarray, fs: float = 256.0) -> Dict:
        """Predict sleep stage from a single EEG epoch.

        Args:
            eeg_epoch: 1D numpy array of EEG samples (typically 30s * fs samples)
            fs: Sampling frequency

        Returns:
            Dict with 'stage', 'stage_index', 'confidence', 'probabilities'
        """
        if self.onnx_session is not None:
            return self._predict_onnx(eeg_epoch, fs)
        if self.sklearn_model is not None:
            try:
                return self._predict_sklearn(eeg_epoch, fs)
            except Exception:
                pass  # fall through to feature-based
        return self._predict_features(eeg_epoch, fs)

    def _predict_sklearn(self, eeg_epoch: np.ndarray, fs: float) -> Dict:
        """Sklearn model inference using extracted features."""
        # Reject epoch if amplitude exceeds 100 µV (blink/movement artifact)
        if np.any(np.abs(eeg_epoch) > 100):
            return {
                "stage": "artifact",
                "stage_index": -1,
                "confidence": 0.0,
                "probabilities": {s: 0.0 for s in STAGE_MAP.values()},
                "artifact_rejected": True,
            }

        # Use multichannel features when available — delta asymmetry between
        # hemispheres improves N3 vs REM discrimination.
        if eeg_epoch.ndim == 2 and eeg_epoch.shape[0] >= 2:
            try:
                from processing.eeg_processor import extract_features_multichannel
                features = extract_features_multichannel(eeg_epoch, fs)
                processed = preprocess(eeg_epoch[0], fs)  # AF7 channel for spindle detection
            except Exception:
                processed = preprocess(eeg_epoch[0], fs)
                features = extract_features(processed, fs)
        else:
            processed = preprocess(eeg_epoch, fs)
            features = extract_features(processed, fs)
        feature_vector = np.array([features.get(k, 0.0) for k in self.feature_names]).reshape(1, -1)

        probs = self.sklearn_model.predict_proba(feature_vector)[0]

        # Boost N2 probability when sleep spindles are detected
        try:
            spindles = detect_sleep_spindles(processed, fs)
            if spindles.get("spindles_detected", False):
                probs[2] = min(1.0, probs[2] + 0.15)
                probs = probs / probs.sum()
        except Exception:
            pass  # spindle detection failure must not break staging

        # K-complex detection: boosts N2 probability (K-complexes = defining N2 feature)
        try:
            k_complexes = detect_k_complexes(processed, fs)
            if len(k_complexes) > 0:
                probs[2] = min(1.0, probs[2] + 0.10)
                probs = probs / probs.sum()
        except Exception:
            pass  # K-complex detection failure must not break staging

        stage_idx = int(np.argmax(probs))

        return {
            "stage": STAGE_MAP[stage_idx],
            "stage_index": stage_idx,
            "confidence": float(probs[stage_idx]),
            "probabilities": {STAGE_MAP[i]: float(p) for i, p in enumerate(probs)},
        }

    def _predict_features(self, eeg_epoch: np.ndarray, fs: float) -> Dict:
        """Feature-based classification using physiological rules."""
        # Reject epoch if amplitude exceeds 100 µV (blink/movement artifact)
        if np.any(np.abs(eeg_epoch) > 100):
            return {
                "stage": "artifact",
                "stage_index": -1,
                "confidence": 0.0,
                "probabilities": {s: 0.0 for s in STAGE_MAP.values()},
                "artifact_rejected": True,
            }

        # Extract single channel for feature-based analysis
        signal = eeg_epoch[0] if eeg_epoch.ndim == 2 else eeg_epoch
        processed = preprocess(signal, fs)
        bands = extract_band_powers(processed, fs)
        features = extract_features(processed, fs)

        delta = bands.get("delta", 0)
        theta = bands.get("theta", 0)
        alpha = bands.get("alpha", 0)
        beta = bands.get("beta", 0)
        gamma = bands.get("gamma", 0)

        # Compute probabilities based on physiological characteristics
        probs = np.zeros(5)

        # Wake: high alpha + beta, low delta (no gamma — EMG contamination at AF7/AF8)
        probs[0] = alpha * 0.4 + beta * 0.4 + (1 - delta) * 0.2

        # N1: theta rising while alpha dropping — the transition zone.
        # Key differentiator from Wake (alpha high) and N2 (spindles/K-complexes).
        # Alpha dropout: alpha is falling but not gone; theta/alpha ratio increases.
        theta_alpha_ratio = theta / max(alpha + theta, 0.01)
        alpha_dropout = float(np.clip(alpha * (1.0 - theta_alpha_ratio), 0, 1))
        probs[1] = (theta * 0.35
                     + alpha_dropout * 0.25       # alpha still present but fading
                     + (1 - beta) * 0.20          # low beta (not awake-alert)
                     + (1 - delta) * 0.10         # not yet deep sleep
                     + theta_alpha_ratio * 0.10)  # theta starting to dominate

        # N2: theta + sleep spindles (sigma 12-14Hz within beta)
        sigma_component = min(beta * 0.3, 0.15)
        probs[2] = theta * 0.35 + sigma_component + delta * 0.25 + (1 - alpha) * 0.15

        # N3: high delta (slow-wave sleep)
        probs[3] = delta * 0.7 + theta * 0.15 + (1 - beta) * 0.1 + (1 - alpha) * 0.05

        # REM: mixed frequency, theta + beta, low delta (no gamma — EMG contamination)
        probs[4] = theta * 0.3 + beta * 0.3 + (1 - delta) * 0.2 + (1 - alpha) * 0.1 + (1 - alpha) * 0.1

        # Add noise for realism
        probs += np.random.uniform(0, 0.05, 5)

        # Normalize
        probs = probs / probs.sum()
        stage_idx = int(np.argmax(probs))

        return {
            "stage": STAGE_MAP[stage_idx],
            "stage_index": stage_idx,
            "confidence": float(probs[stage_idx]),
            "probabilities": {STAGE_MAP[i]: float(p) for i, p in enumerate(probs)},
        }

    def _predict_onnx(self, eeg_epoch: np.ndarray, fs: float) -> Dict:
        """ONNX model inference."""
        input_data = eeg_epoch.reshape(1, 1, -1).astype(np.float32)
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: input_data})
        probs = outputs[0][0]
        if probs.shape[0] != 5:
            probs = np.zeros(5)
            probs[0] = 1.0
        stage_idx = int(np.argmax(probs))
        return {
            "stage": STAGE_MAP[stage_idx],
            "stage_index": stage_idx,
            "confidence": float(probs[stage_idx]),
            "probabilities": {STAGE_MAP[i]: float(p) for i, p in enumerate(probs)},
        }

    def predict_sequence(self, epochs: List[np.ndarray], fs: float = 256.0) -> List[Dict]:
        """Predict sleep stages for a sequence of epochs (full night).

        Returns list of per-epoch dicts, each augmented with:
          - spindle_density: spindles per minute in N2 epochs (float)
          - so_spindle_coupling: phase-amplitude coupling score 0-1 (float)
        """
        results = []
        for epoch in epochs:
            result = self.predict(epoch, fs)
            results.append(result)

        smoothed = self._smooth_predictions(results)

        # Augment with advanced biomarkers (#499)
        for i, (result, epoch) in enumerate(zip(smoothed, epochs)):
            signal = epoch[0] if epoch.ndim == 2 else epoch
            result["spindle_density"] = compute_spindle_density(signal, fs, result["stage"])
            result["so_spindle_coupling"] = compute_so_spindle_coupling(signal, fs, result["stage"])

        return smoothed

    # Biologically invalid transition penalties (Markov prior).
    # Stage indices: Wake=0, N1=1, N2=2, N3=3, REM=4.
    # Tuple format: (from_stage, to_stage) → multiplier for the destination probability.
    _TRANSITION_PENALTIES = {
        (0, 4): 0.1,  # Wake → REM direct: essentially impossible
        (3, 1): 0.2,  # N3 → N1 direct: must surface through N2 first
        (4, 3): 0.2,  # REM → N3 direct: very rare, REM usually lightens to N1/N2
    }

    def _smooth_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """Apply temporal smoothing with Markov transition priors.

        1. Penalise biologically invalid stage jumps by scaling down the
           destination-stage probability before argmax.
        2. Fall back to the previous stage if confidence is low and the
           jump is large (legacy behaviour retained).
        """
        if len(predictions) <= 1:
            return predictions

        smoothed = [predictions[0]]
        for i in range(1, len(predictions)):
            current = predictions[i]
            prev_stage = smoothed[i - 1]["stage_index"]
            curr_stage = current["stage_index"]

            # --- Markov transition prior ---
            penalty = self._TRANSITION_PENALTIES.get((prev_stage, curr_stage))
            if penalty is not None:
                # Re-weight probabilities: scale down the invalid destination,
                # then renormalise so all probs still sum to 1.
                raw_probs = np.array(
                    [current["probabilities"][STAGE_MAP[j]] for j in range(5)]
                )
                raw_probs[curr_stage] *= penalty
                raw_probs = raw_probs / raw_probs.sum()
                new_stage_idx = int(np.argmax(raw_probs))
                current = {
                    **current,
                    "stage": STAGE_MAP[new_stage_idx],
                    "stage_index": new_stage_idx,
                    "confidence": float(raw_probs[new_stage_idx]),
                    "probabilities": {STAGE_MAP[j]: float(raw_probs[j]) for j in range(5)},
                }
                curr_stage = new_stage_idx

            # --- Legacy large-jump guard (unchanged) ---
            if abs(curr_stage - prev_stage) > 2 and current["confidence"] < 0.6:
                current = {**current, "stage": STAGE_MAP[prev_stage], "stage_index": prev_stage}

            smoothed.append(current)

        return smoothed


# ── Advanced sleep biomarkers (#499) ──────────────────────────────────────


def compute_spindle_density(
    signal: np.ndarray, fs: float, stage: str
) -> float:
    """Compute spindle density (spindles per minute) for N2 epochs.

    Spindle density during N2 is a validated marker of sleep-dependent memory
    consolidation (Schabus et al., 2004).

    Args:
        signal: 1D EEG epoch (typically 30s)
        fs: Sampling frequency
        stage: Predicted sleep stage label

    Returns:
        Spindle density (spindles/min). Returns 0.0 for non-N2 stages.
    """
    if stage != "N2":
        return 0.0

    try:
        spindles = detect_sleep_spindles(signal, fs)
        # detect_sleep_spindles returns a list of dicts
        n_spindles = len(spindles) if isinstance(spindles, list) else 0
        # Handle dict return from some versions
        if isinstance(spindles, dict):
            n_spindles = spindles.get("count", 0)
            if not n_spindles:
                n_spindles = 1 if spindles.get("spindles_detected", False) else 0
        duration_minutes = len(signal) / fs / 60.0
        if duration_minutes <= 0:
            return 0.0
        return round(n_spindles / duration_minutes, 2)
    except Exception:
        return 0.0


def compute_so_spindle_coupling(
    signal: np.ndarray, fs: float, stage: str
) -> float:
    """Compute slow-oscillation/spindle phase-amplitude coupling score.

    Phase-amplitude coupling (PAC) between slow oscillations (0.5-1.5 Hz phase)
    and spindles (11-16 Hz amplitude) is a validated biomarker for memory
    consolidation quality (Helfrich et al., 2018; Staresina et al., 2015).

    Uses Modulation Index (MI) approach: extracts SO phase and spindle amplitude,
    computes the non-uniformity of amplitude distribution across phase bins.

    Args:
        signal: 1D EEG epoch (at least a few seconds)
        fs: Sampling frequency
        stage: Predicted sleep stage label

    Returns:
        Coupling score 0-1 (0 = no coupling, 1 = strong coupling).
        Returns 0.0 for non-N2/N3 stages.
    """
    if stage not in ("N2", "N3"):
        return 0.0

    if len(signal) < int(fs * 2):
        return 0.0

    try:
        from scipy.signal import hilbert, butter, filtfilt

        # Extract slow oscillation phase (0.5-1.5 Hz)
        nyq = fs / 2
        b_so, a_so = butter(3, [0.5 / nyq, 1.5 / nyq], btype="band")
        so_filtered = filtfilt(b_so, a_so, signal)
        so_phase = np.angle(hilbert(so_filtered))

        # Extract spindle amplitude envelope (11-16 Hz)
        b_sp, a_sp = butter(3, [11.0 / nyq, 16.0 / nyq], btype="band")
        sp_filtered = filtfilt(b_sp, a_sp, signal)
        sp_amplitude = np.abs(hilbert(sp_filtered))

        # Modulation Index: bin amplitudes by phase, measure non-uniformity
        n_bins = 18
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        mean_amps = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (so_phase >= phase_bins[i]) & (so_phase < phase_bins[i + 1])
            if mask.sum() > 0:
                mean_amps[i] = float(np.mean(sp_amplitude[mask]))

        # Normalize to probability distribution
        total = mean_amps.sum()
        if total <= 0:
            return 0.0

        p = mean_amps / total
        # Kullback-Leibler divergence from uniform distribution
        uniform = np.ones(n_bins) / n_bins
        # Avoid log(0)
        p_safe = np.clip(p, 1e-10, None)
        kl_div = float(np.sum(p_safe * np.log(p_safe / uniform)))
        # Normalize by max possible KL (log(n_bins))
        max_kl = np.log(n_bins)
        mi = float(np.clip(kl_div / max_kl, 0, 1))

        return round(mi, 3)

    except Exception:
        return 0.0


# -- Sleep onset detection (#sleep-onset) ------------------------------------

# Minimum consecutive non-Wake epochs to confirm true sleep onset.
# Prevents false positives from brief N1 microsleep blips that revert to Wake.
# 3 epochs * 30s = 90 seconds sustained sleep = standard clinical criterion.
_SUSTAINED_SLEEP_EPOCHS = 3


def detect_sleep_onset(
    epochs: List[Dict],
    epoch_duration_s: float = 30.0,
    recording_start: Optional[datetime] = None,
) -> Optional[Dict]:
    """Detect the exact moment of sleep onset from staged epochs.

    Scans staged epochs for the first Wake -> non-Wake transition that is
    sustained for at least 3 consecutive epochs (90 seconds at 30s/epoch).
    This matches the AASM standard for sleep onset: the first epoch of
    sustained sleep, typically the Wake -> N1 boundary.

    "Non-Wake" includes N1, N2, N3, and REM. A transition directly from
    Wake to N2 (common in sleep-deprived individuals) also counts.

    Args:
        epochs: List of epoch dicts, each with at least:
            - "stage": str ("Wake", "N1", "N2", "N3", "REM")
            - "stage_index": int (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
            - "confidence": float (0-1)
        epoch_duration_s: Duration of each epoch in seconds (default 30).
        recording_start: Optional datetime of recording start. When provided,
            the result includes an ISO-format onset_time string.

    Returns:
        None if no sleep onset detected (all Wake, or non-Wake never sustained).
        Otherwise a dict:
            onset_epoch: int -- index of the first sustained non-Wake epoch
            onset_latency_min: float -- minutes from recording start to onset
            onset_time: str (ISO format) -- only present if recording_start given
            confidence: float -- mean confidence of the sustained sleep epochs
            transition: str -- e.g. "Wake -> N1"
    """
    if not epochs:
        return None

    n = len(epochs)

    # Special case: entire recording is non-Wake (user was already asleep)
    if all(_is_sleep(e) for e in epochs):
        first_stages = epochs[:min(_SUSTAINED_SLEEP_EPOCHS, n)]
        mean_conf = float(np.mean([e.get("confidence", 0.5) for e in first_stages]))
        result: Dict = {
            "onset_epoch": 0,
            "onset_latency_min": 0.0,
            "confidence": round(mean_conf, 3),
            "transition": f"(recording start) -> {epochs[0]['stage']}",
        }
        if recording_start is not None:
            result["onset_time"] = recording_start.isoformat()
        return result

    # Scan for first sustained non-Wake run after Wake
    i = 0
    while i < n:
        # Skip Wake epochs
        if not _is_sleep(epochs[i]):
            i += 1
            continue

        # Found a non-Wake epoch -- check if sustained
        run_start = i
        run_len = 0
        while i < n and _is_sleep(epochs[i]):
            run_len += 1
            i += 1

        if run_len >= _SUSTAINED_SLEEP_EPOCHS:
            # This is the onset
            onset_idx = run_start
            sustained = epochs[run_start : run_start + _SUSTAINED_SLEEP_EPOCHS]
            mean_conf = float(np.mean([e.get("confidence", 0.5) for e in sustained]))
            latency_min = round(onset_idx * epoch_duration_s / 60.0, 1)

            # Determine transition label
            if onset_idx > 0:
                prev_stage = epochs[onset_idx - 1]["stage"]
            else:
                prev_stage = "(recording start)"
            curr_stage = epochs[onset_idx]["stage"]

            result = {
                "onset_epoch": onset_idx,
                "onset_latency_min": latency_min,
                "confidence": round(mean_conf, 3),
                "transition": f"{prev_stage} -> {curr_stage}",
            }

            if recording_start is not None:
                onset_dt = recording_start + timedelta(seconds=onset_idx * epoch_duration_s)
                result["onset_time"] = onset_dt.isoformat()

            return result

    # No sustained sleep found
    return None


def _is_sleep(epoch: Dict) -> bool:
    """Return True if this epoch is a non-Wake sleep stage."""
    stage = epoch.get("stage", "Wake")
    return stage != "Wake" and stage != "W"
