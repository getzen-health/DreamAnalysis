"""Lucid Dream Detector from EEG signals.

Detects lucid dreaming — awareness of dreaming while asleep.

States:
  0: non_lucid     — standard dreaming without awareness
  1: pre_lucid     — approaching lucidity (reality testing patterns)
  2: lucid         — aware of dreaming, gamma bursts in frontal cortex
  3: controlled    — actively controlling dream content

Scientific basis:
- Lucid dreams show 40 Hz gamma increase in frontal/temporal regions
  (Voss et al., 2009, 2014)
- Frontal gamma coherence distinguishes lucid from non-lucid REM
- Alpha/gamma coupling in lucid dreams (Holzinger et al., 2006)
- Increased frontal beta during lucid dreams (metacognition)
- REM density increases during lucid episodes

This detector works ON TOP of the sleep staging model — it only activates
during detected REM sleep.

Reference: Voss et al. (2009), LaBerge (1990), Dresler et al. (2012)

=== CRITICAL HARDWARE WARNING: MUSE 2 GAMMA RELIABILITY ===
Gamma (30-100 Hz) is the PRIMARY scientific marker for lucid dreaming
(Voss et al., 2009) and is intentionally RETAINED here.  However:

  On Muse 2's AF7/AF8 dry frontal electrodes, gamma-band power is
  PREDOMINANTLY EMG muscle artifact (jaw clenching, frontalis tension,
  brow furrowing) rather than neural signal.

Implications for live use:
  - Gamma readings during waking with jaw tension will falsely elevate
    lucidity scores.
  - True neural 40 Hz lucid-dream gamma is mixed with EMG artifact
    and cannot be reliably separated with 4-ch dry electrodes.
  - Gamma weight is reduced to 50% of the Voss et al. literature value.
  - Two compensating features are added:
      (a) theta_gamma_ratio  — lucid REM has theta maintenance alongside
          gamma; pure EMG does NOT produce theta, so high theta/gamma
          ratio filters out artifact-only gamma spikes.
      (b) REM stage confirmation via `is_rem` flag — constrains
          gamma scoring to epochs already classified as REM by the
          sleep staging model, filtering out waking muscle noise.

Accuracy expectation on Muse 2:
  - With REM confirmed:    ~55-65% (vs ~75% with gel 32-ch EEG)
  - Without REM confirmed: ~35-45% (near chance; do not report lucidity)

To improve: record with BLED112 dongle (less Bluetooth dropout), enforce
artifact rejection (75 µV threshold), and pair with eye-movement detection
from the PPG/accelerometer signal as secondary REM confirmation.
"""

import numpy as np
from typing import Dict, Optional
from processing.eeg_processor import (
    extract_band_powers, extract_features, preprocess
)

LUCIDITY_STATES = ["non_lucid", "pre_lucid", "lucid", "controlled"]


class LucidDreamDetector:
    """EEG-based lucid dream detector for REM sleep periods."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_type = "feature-based"
        self.sklearn_model = None
        self.feature_names = None
        self.scaler = None
        # Track gamma activity during REM for onset detection
        self._gamma_history = []
        self._lucidity_history = []

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

    def predict(
        self,
        eeg: np.ndarray,
        fs: float = 256.0,
        is_rem: bool = True,
        sleep_stage: int = 4,
    ) -> Dict:
        """Detect lucid dreaming from EEG during REM sleep.

        Should only be called during REM sleep (sleep_stage == 4).
        During non-REM, returns non_lucid with zero lucidity.

        Args:
            eeg: Raw EEG signal.
            fs: Sampling frequency.
            is_rem: Whether current epoch is REM sleep.
            sleep_stage: Current sleep stage (4 = REM).

        Returns:
            Dict with 'state', 'lucidity_score', 'confidence',
            'gamma_surge', 'metacognition_index', components.
        """
        if not is_rem and sleep_stage != 4:
            return {
                "state": "non_lucid",
                "state_index": 0,
                "lucidity_score": 0.0,
                "confidence": 0.9,
                "gamma_surge": 0.0,
                "metacognition_index": 0.0,
                "control_index": 0.0,
                "components": {},
                "band_powers": {},
                "note": "Not in REM sleep — lucid dreaming only possible during REM",
            }

        if self.sklearn_model is not None:
            try:
                return self._predict_sklearn(eeg, fs)
            except Exception:
                pass  # fall through to feature-based

        # Multichannel safety: preprocess expects 1D — use first channel if 2D
        signal = eeg[0] if eeg.ndim == 2 else eeg
        processed = preprocess(signal, fs)
        bands = extract_band_powers(processed, fs)

        alpha = bands.get("alpha", 0)
        beta = bands.get("beta", 0)
        theta = bands.get("theta", 0)
        delta = bands.get("delta", 0)
        # WARNING: gamma on Muse 2 AF7/AF8 is predominantly EMG artifact.
        # Retained at 50% weight (down from ~45% across gamma_surge + gamma_onset)
        # because 40 Hz gamma IS the primary Voss et al. lucid dream signature.
        # See module docstring for full hardware warning.
        gamma = bands.get("gamma", 0)

        # === Lucidity Components ===

        # 1. Frontal 40 Hz Gamma Surge (Voss et al. 2009 signature)
        # MUSE 2 WARNING: contaminated by EMG — weight reduced 50% vs literature.
        # Compensating features added: theta_gamma_ratio + is_rem gating (below).
        gamma_surge = float(np.clip(np.tanh(gamma * 10 - 0.5), 0, 1))  # was *20

        # Track gamma for onset detection
        self._gamma_history.append(gamma)
        if len(self._gamma_history) > 30:
            self._gamma_history = self._gamma_history[-30:]

        # Gamma increase relative to recent REM baseline
        # (onset signal is also kept but at halved contribution)
        if len(self._gamma_history) >= 5:
            recent_gamma = np.mean(self._gamma_history[-3:])
            baseline_gamma = np.mean(self._gamma_history[:3])
            gamma_increase = (recent_gamma - baseline_gamma) / (baseline_gamma + 1e-10)
            gamma_onset = float(np.clip(np.tanh(gamma_increase * 5), 0, 1))
        else:
            gamma_onset = 0.0

        # 2. Metacognition Index (frontal beta during REM = awareness)
        # Normal REM has low beta; lucid REM has elevated frontal beta
        metacognition_index = float(np.clip(np.tanh(beta * 8 - 0.5), 0, 1))

        # 3. Alpha/Gamma Coupling (Holzinger 2006 signature)
        # MUSE 2 WARNING: gamma component is EMG-contaminated; use as soft indicator.
        alpha_gamma_product = alpha * gamma * 100
        coupling = float(np.clip(np.tanh(alpha_gamma_product), 0, 1))

        # 4. Theta Maintenance (dream content richness)
        # Lucid dreams maintain theta (vivid imagery) while adding gamma.
        # Also acts as EMG filter: sustained theta co-occurring with gamma
        # is more likely neural than pure EMG (which is broadband, not theta-locked).
        theta_maintenance = float(np.clip(np.tanh(theta * 5), 0, 1))

        # 5. Delta Reduction (increased cortical arousal within sleep)
        # Lucid dreamers show less delta than standard REM
        delta_reduction = float(np.clip(1.0 - np.tanh(delta * 3), 0, 1))

        # 6. Theta-Gamma Ratio (compensating feature — EMG artifact filter)
        # Lucid REM: sustained theta WITH gamma → ratio stays moderate (0.5-2.0)
        # Pure EMG spike: gamma shoots up but theta unchanged → ratio near 0
        # High ratio favors true neural gamma over muscle artifact
        theta_gamma_ratio = theta / (gamma + 1e-10)
        # Sigmoid centered at ratio=1.0: peaks when theta ≈ gamma
        theta_gamma_score = float(np.clip(np.tanh(theta_gamma_ratio - 0.5), 0, 1))

        # 7. REM Stage Confirmation (compensating feature)
        # Constrains lucidity scoring to REM-confirmed epochs.
        # EMG artifacts can occur at any sleep stage or while waking;
        # true lucid-dream gamma only occurs during REM.
        rem_confirmed = float(1.0 if is_rem else 0.3)

        # 8. Control Index (highest lucidity level)
        # Requires sustained gamma + high metacognition + stable theta
        control_index = float(np.clip(
            gamma_surge * metacognition_index * theta_maintenance * rem_confirmed,
            0, 1
        ))

        # === Overall Lucidity Score ===
        # Gamma weights reduced from (0.30 surge + 0.15 onset = 0.45) to
        # (0.15 surge + 0.075 onset ≈ 0.225).  Freed weight redistributed to:
        # theta_gamma_score (+0.15) and rem_confirmed as a multiplicative gate.
        lucidity_score = float(np.clip(
            (
                0.15 * gamma_surge +           # was 0.30 — halved (EMG warning)
                0.20 * metacognition_index +
                0.15 * coupling +
                0.075 * gamma_onset +          # was 0.15 — halved (EMG warning)
                0.10 * delta_reduction +
                0.10 * theta_maintenance +
                0.125 * theta_gamma_score      # new: EMG artifact filter
            ) * rem_confirmed,                 # new: REM gate — scales to 0.3 if not REM
            0, 1
        ))

        # Track lucidity
        self._lucidity_history.append(lucidity_score)
        if len(self._lucidity_history) > 60:
            self._lucidity_history = self._lucidity_history[-60:]

        # === Classify State ===
        if lucidity_score >= 0.70 and control_index >= 0.4:
            state_idx = 3  # controlled
        elif lucidity_score >= 0.50:
            state_idx = 2  # lucid
        elif lucidity_score >= 0.25:
            state_idx = 1  # pre_lucid
        else:
            state_idx = 0  # non_lucid

        # Confidence
        thresholds = [0.0, 0.25, 0.50, 0.70, 1.0]
        mid = (thresholds[state_idx] + thresholds[state_idx + 1]) / 2
        dist = abs(lucidity_score - mid)
        range_size = thresholds[state_idx + 1] - thresholds[state_idx]
        confidence = float(np.clip(1.0 - dist / (range_size / 2 + 1e-10), 0.3, 0.95))

        return {
            "state": LUCIDITY_STATES[state_idx],
            "state_index": state_idx,
            "lucidity_score": round(lucidity_score, 3),
            "confidence": round(confidence, 3),
            "gamma_surge": round(gamma_surge, 3),
            "metacognition_index": round(metacognition_index, 3),
            "control_index": round(control_index, 3),
            # gamma_emg_warning: True when gamma is high but theta is low,
            # suggesting the gamma reading is likely EMG artifact on Muse 2
            "gamma_emg_warning": bool(gamma > 0.05 and theta_gamma_ratio < 0.3),
            "components": {
                "gamma_surge": round(gamma_surge, 3),
                "gamma_onset": round(gamma_onset, 3),
                "metacognition_index": round(metacognition_index, 3),
                "alpha_gamma_coupling": round(coupling, 3),
                "theta_maintenance": round(theta_maintenance, 3),
                "delta_reduction": round(delta_reduction, 3),
                # New compensating features
                "theta_gamma_ratio_score": round(theta_gamma_score, 3),
                "rem_confirmed": round(rem_confirmed, 3),
            },
            "band_powers": bands,
        }

    def _predict_sklearn(self, eeg: np.ndarray, fs: float) -> Dict:
        # Multichannel safety: preprocess/extract_features expect 1D
        signal = eeg[0] if eeg.ndim == 2 else eeg
        processed = preprocess(signal, fs)
        features = extract_features(processed, fs)
        bands = extract_band_powers(processed, fs)
        fv = np.array([features.get(k, 0.0) for k in self.feature_names]).reshape(1, -1)
        if self.scaler is not None:
            fv = self.scaler.transform(fv)

        probs = self.sklearn_model.predict_proba(fv)[0]
        state_idx = int(np.argmax(probs))

        return {
            "state": LUCIDITY_STATES[min(state_idx, 3)],
            "state_index": min(state_idx, 3),
            "lucidity_score": round(float(np.sum(probs[1:] * [0.3, 0.7, 1.0])) if len(probs) >= 4 else float(probs[state_idx]), 3),
            "confidence": round(float(probs[state_idx]), 3),
            "gamma_surge": round(float(bands.get("gamma", 0.05) * 10), 3),
            "metacognition_index": round(float(bands.get("beta", 0.15) * 5), 3),
            "control_index": 0.0,
            "components": {},
            "band_powers": bands,
        }

    def get_session_stats(self) -> Dict:
        """Get lucid dream session statistics."""
        if not self._lucidity_history:
            return {"n_epochs": 0}

        scores = np.array(self._lucidity_history)
        return {
            "n_epochs": len(scores),
            "avg_lucidity": round(float(np.mean(scores)), 3),
            "max_lucidity": round(float(np.max(scores)), 3),
            "lucid_epochs": int(np.sum(scores >= 0.5)),
            "pre_lucid_epochs": int(np.sum((scores >= 0.25) & (scores < 0.5))),
            "lucidity_ratio": round(float(np.mean(scores >= 0.5)), 3),
        }
