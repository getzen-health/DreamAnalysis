"""Food Emotion Predictor — EEG-based food state and nutrition recommendation model.

Maps real-time EEG biomarkers to food craving states and provides nutritional guidance:

    FAA  = ln(AF8_alpha) - ln(AF7_alpha)  → approach/avoidance motivation
    high-beta (20-30 Hz)                  → sugar/carb craving index
    prefrontal theta (4-8 Hz)             → dietary self-regulation capacity
    delta (0.5-4 Hz)                      → satiety / post-meal relaxation signal

Six food states:
    craving_carbs         — high-beta + positive FAA + low self-regulation
    appetite_suppressed   — high delta + low approach motivation
    comfort_seeking       — negative FAA (withdrawal) + elevated theta
    balanced              — moderate values across all biomarkers
    stress_eating         — high-beta + negative FAA + low self-regulation
    mindful_eating        — high theta (self-regulation) + balanced FAA

Muse 2 channel order (BrainFlow board 38):
    ch0 = TP9   ch1 = AF7 (FAA left)   ch2 = AF8 (FAA right)   ch3 = TP10
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Biomarker weights for each food state ────────────────────────────────────
# Each row: [faa_weight, high_beta_weight, theta_weight, delta_weight]
# Positive weight → higher value increases state probability
# Negative weight → higher value decreases state probability
_STATE_PROFILES: Dict[str, Dict[str, float]] = {
    "craving_carbs": {
        "faa": 0.35,       # positive FAA → approach/craving
        "high_beta": 0.40,  # high-beta drives craving
        "theta": -0.15,    # low self-regulation → craving
        "delta": -0.10,    # not satiated
    },
    "appetite_suppressed": {
        "faa": -0.20,      # withdrawal motivation
        "high_beta": -0.20, # low arousal
        "theta": 0.20,     # high regulation
        "delta": 0.40,     # satiety signal prominent
    },
    "comfort_seeking": {
        "faa": -0.40,      # withdrawal/negative affect
        "high_beta": 0.15,
        "theta": 0.30,     # moderate theta (stress-coping)
        "delta": 0.15,
    },
    "balanced": {
        "faa": 0.05,       # near-neutral FAA
        "high_beta": -0.30, # low stress/craving
        "theta": 0.35,     # good self-regulation
        "delta": 0.30,     # moderate satiety
    },
    "stress_eating": {
        "faa": -0.30,      # negative valence
        "high_beta": 0.50,  # stress-driven
        "theta": -0.10,    # low regulation
        "delta": -0.10,
    },
    "mindful_eating": {
        "faa": 0.15,       # slightly positive
        "high_beta": -0.35, # calm
        "theta": 0.40,     # high self-regulation
        "delta": 0.10,
    },
}

# Food recommendations per state
_RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {
    "craving_carbs": {
        "avoid": ["refined sugar", "white bread", "candy", "soda"],
        "prefer": ["complex carbs", "oats", "brown rice", "sweet potato", "fruit"],
        "strategy": "Choose complex carbohydrates with fiber to satisfy cravings without blood sugar spike.",
        "mindfulness_tip": "Pause 5 minutes before eating — cravings often peak and fade.",
    },
    "appetite_suppressed": {
        "avoid": ["heavy meals", "high-fat foods", "alcohol"],
        "prefer": ["light snacks", "smoothies", "soup", "small portions of nutrient-dense food"],
        "strategy": "Eat small, nutrient-dense portions to maintain energy without overwhelming your system.",
        "mindfulness_tip": "Check in with hydration — thirst is often misread as appetite suppression.",
    },
    "comfort_seeking": {
        "avoid": ["ultra-processed comfort foods", "excessive sugar", "emotional binge foods"],
        "prefer": ["warm foods", "herbal tea", "dark chocolate (70%+)", "fermented foods"],
        "strategy": "Reach for foods with genuine serotonin or GABA precursors: tryptophan (turkey, nuts), magnesium (dark chocolate, spinach).",
        "mindfulness_tip": "Name the emotion before eating. Comfort eating is a signal, not just a craving.",
    },
    "balanced": {
        "avoid": ["skipping meals", "extreme diets"],
        "prefer": ["varied whole foods", "Mediterranean-style meals"],
        "strategy": "Your brain state is balanced — this is the optimal window for intuitive eating decisions.",
        "mindfulness_tip": "Eat slowly and savor. Your regulatory capacity is high right now.",
    },
    "stress_eating": {
        "avoid": ["fast food", "processed snacks", "caffeine", "alcohol"],
        "prefer": ["magnesium-rich foods", "complex carbs", "omega-3s", "chamomile tea"],
        "strategy": "Stress depletes magnesium and B-vitamins. Prioritize leafy greens, nuts, seeds, and whole grains.",
        "mindfulness_tip": "Take 3 slow breaths before eating. Stress eating bypasses hunger signals — slow down.",
    },
    "mindful_eating": {
        "avoid": ["rushed eating", "eating while distracted"],
        "prefer": ["whole foods", "high-fiber", "fermented foods", "colorful vegetables"],
        "strategy": "Your prefrontal theta is high — leverage this moment of self-regulation for your best food choices.",
        "mindfulness_tip": "This is your peak window for mindful eating. Eat with full sensory awareness.",
    },
}


class FoodEmotionPredictor:
    """Predicts food craving state from EEG biomarkers and provides dietary guidance.

    Usage:
        model = FoodEmotionPredictor()
        result = model.predict(eeg_array, fs=256.0)
        print(result["food_state"], result["recommendations"])
    """

    # ── Band frequency boundaries (Hz) ───────────────────────────────────────
    _DELTA_BAND = (0.5, 4.0)
    _THETA_BAND = (4.0, 8.0)
    _ALPHA_BAND = (8.0, 13.0)
    _HIGH_BETA_BAND = (20.0, 30.0)

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_path = model_path
        self._model: Optional[Any] = None

        # Auto-calibration buffer — collects first 30 predictions to compute stable baseline
        self._calibration_buffer: List[Dict[str, float]] = []
        self._is_calibrated: bool = False
        self.baseline_faa: Optional[float] = None
        self.baseline_high_beta: Optional[float] = None
        self.baseline_theta: Optional[float] = None
        self.baseline_delta: Optional[float] = None

        # EMA history for smoothed output
        self._food_state_history: List[str] = []
        self._faa_ema: Optional[float] = None
        self._high_beta_ema: Optional[float] = None
        self._theta_ema: Optional[float] = None
        self._delta_ema: Optional[float] = None

        if model_path:
            self._load_model(model_path)

    # ─── Model loading ────────────────────────────────────────────────────────

    def _load_model(self, path: str) -> None:
        try:
            import joblib
            self._model = joblib.load(path)
            logger.info("FoodEmotionPredictor: loaded model from %s", path)
        except Exception as exc:
            logger.warning("FoodEmotionPredictor: could not load model from %s — %s", path, exc)
            self._model = None

    # ─── Public calibration ───────────────────────────────────────────────────

    def calibrate(self, resting_eeg: np.ndarray, fs: float = 256.0) -> Dict[str, Any]:
        """Compute resting-state baselines for all four biomarkers.

        Args:
            resting_eeg: EEG array — shape (n_channels, n_samples) or (n_samples,)
            fs: Sampling frequency in Hz.

        Returns:
            Dict confirming calibration success and baseline values.
        """
        bp = self._extract_biomarkers(resting_eeg, fs)
        self.baseline_faa = bp["faa"]
        self.baseline_high_beta = bp["high_beta_power"]
        self.baseline_theta = bp["theta_power"]
        self.baseline_delta = bp["delta_power"]
        self._is_calibrated = True
        self._calibration_buffer.clear()
        logger.info("FoodEmotionPredictor: calibrated — FAA=%.3f high_beta=%.4f theta=%.4f delta=%.4f",
                    self.baseline_faa, self.baseline_high_beta, self.baseline_theta, self.baseline_delta)
        return {
            "calibrated": True,
            "baseline_faa": float(self.baseline_faa),
            "baseline_high_beta": float(self.baseline_high_beta),
            "baseline_theta": float(self.baseline_theta),
            "baseline_delta": float(self.baseline_delta),
        }

    # ─── Main predict ─────────────────────────────────────────────────────────

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict[str, Any]:
        """Predict food craving state from EEG.

        Args:
            eeg: EEG signals — shape (n_channels, n_samples) or (n_samples,)
            fs: Sampling frequency in Hz.

        Returns:
            Dict with food_state, food_state_index, confidence, recommendations,
            components, band_powers, faa, and model_type.
        """
        biomarkers = self._extract_biomarkers(eeg, fs)

        # ── Auto-calibrate from first 30 readings ────────────────────────────
        if not self._is_calibrated:
            self._calibration_buffer.append({
                "faa": biomarkers["faa"],
                "high_beta": biomarkers["high_beta_power"],
                "theta": biomarkers["theta_power"],
                "delta": biomarkers["delta_power"],
            })
            if len(self._calibration_buffer) >= 30:
                self.baseline_faa = float(np.median([b["faa"] for b in self._calibration_buffer]))
                self.baseline_high_beta = float(np.median([b["high_beta"] for b in self._calibration_buffer]))
                self.baseline_theta = float(np.median([b["theta"] for b in self._calibration_buffer]))
                self.baseline_delta = float(np.median([b["delta"] for b in self._calibration_buffer]))
                self._is_calibrated = True
                logger.info("FoodEmotionPredictor: auto-calibrated from 30 readings")

        # ── Normalise biomarkers relative to baseline ─────────────────────────
        faa_norm = self._normalise(biomarkers["faa"], self.baseline_faa, scale=2.0)
        high_beta_norm = self._normalise(biomarkers["high_beta_power"], self.baseline_high_beta, scale=1.5)
        theta_norm = self._normalise(biomarkers["theta_power"], self.baseline_theta, scale=1.5)
        delta_norm = self._normalise(biomarkers["delta_power"], self.baseline_delta, scale=1.5)

        # ── EMA smoothing ─────────────────────────────────────────────────────
        alpha_ema = 0.25  # EMA weight for new reading
        self._faa_ema = faa_norm if self._faa_ema is None else alpha_ema * faa_norm + (1 - alpha_ema) * self._faa_ema
        self._high_beta_ema = high_beta_norm if self._high_beta_ema is None else alpha_ema * high_beta_norm + (1 - alpha_ema) * self._high_beta_ema
        self._theta_ema = theta_norm if self._theta_ema is None else alpha_ema * theta_norm + (1 - alpha_ema) * self._theta_ema
        self._delta_ema = delta_norm if self._delta_ema is None else alpha_ema * delta_norm + (1 - alpha_ema) * self._delta_ema

        # ── Score each food state ─────────────────────────────────────────────
        scores: Dict[str, float] = {}
        for state, profile in _STATE_PROFILES.items():
            score = (
                profile["faa"] * float(self._faa_ema)
                + profile["high_beta"] * float(self._high_beta_ema)
                + profile["theta"] * float(self._theta_ema)
                + profile["delta"] * float(self._delta_ema)
            )
            scores[state] = float(score)

        # Softmax for probabilities
        score_arr = np.array(list(scores.values()))
        exp_scores = np.exp(score_arr - score_arr.max())
        probs = exp_scores / exp_scores.sum()
        state_probs = dict(zip(scores.keys(), probs.tolist()))

        best_state: str = max(state_probs, key=lambda k: state_probs[k])
        confidence = float(state_probs[best_state])

        # food_state_index: normalised 0-1 score (higher = stronger signal)
        food_state_index = float(np.clip(confidence, 0.0, 1.0))

        # Keep short history for display
        self._food_state_history.append(best_state)
        if len(self._food_state_history) > 10:
            self._food_state_history.pop(0)

        return {
            "food_state": best_state,
            "food_state_index": food_state_index,
            "confidence": confidence,
            "state_probabilities": state_probs,
            "recommendations": _RECOMMENDATIONS[best_state],
            "components": {
                "faa": float(self._faa_ema),
                "high_beta": float(self._high_beta_ema),
                "prefrontal_theta": float(self._theta_ema),
                "delta": float(self._delta_ema),
            },
            "band_powers": {
                "delta": float(biomarkers["delta_power"]),
                "theta": float(biomarkers["theta_power"]),
                "alpha": float(biomarkers["alpha_power"]),
                "high_beta": float(biomarkers["high_beta_power"]),
            },
            "faa": float(biomarkers["faa"]),
            "is_calibrated": self._is_calibrated,
            "calibration_progress": 1.0 if self._is_calibrated else min(len(self._calibration_buffer), 30) / 30.0,
            "model_type": "feature-based",
        }

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _extract_biomarkers(self, eeg: np.ndarray, fs: float) -> Dict[str, float]:
        """Extract FAA, high-beta, theta, delta, and alpha from raw EEG.

        Handles single-channel (n_samples,) and multichannel (n_channels, n_samples).
        For FAA, uses ch1=AF7 and ch2=AF8 when a multichannel array is provided.
        """
        if eeg.ndim == 1:
            signal = eeg
            channels = None
        else:
            signal = eeg[0]   # AF7 (left frontal) as primary single-channel signal
            channels = eeg

        # Band powers on primary signal (AF7)
        delta_p = self._band_power(signal, fs, *self._DELTA_BAND)
        theta_p = self._band_power(signal, fs, *self._THETA_BAND)
        alpha_p = self._band_power(signal, fs, *self._ALPHA_BAND)
        high_beta_p = self._band_power(signal, fs, *self._HIGH_BETA_BAND)

        # FAA: use AF7 (ch1) and AF8 (ch2) when multichannel available
        faa = 0.0
        if channels is not None and channels.shape[0] >= 3:
            af7_alpha = self._band_power(channels[1], fs, *self._ALPHA_BAND)
            af8_alpha = self._band_power(channels[2], fs, *self._ALPHA_BAND)
            # FAA = ln(right_alpha) - ln(left_alpha); positive → approach motivation
            faa = float(np.log(af8_alpha + 1e-10) - np.log(af7_alpha + 1e-10))
        elif channels is not None and channels.shape[0] >= 2:
            # Two channels available — treat ch0 as left, ch1 as right
            af7_alpha = self._band_power(channels[0], fs, *self._ALPHA_BAND)
            af8_alpha = self._band_power(channels[1], fs, *self._ALPHA_BAND)
            faa = float(np.log(af8_alpha + 1e-10) - np.log(af7_alpha + 1e-10))
        else:
            # Single channel — no FAA; use alpha/beta heuristic as proxy
            beta_p = self._band_power(signal, fs, 12.0, 30.0)
            faa = float(np.clip(np.tanh((alpha_p / (beta_p + 1e-10) - 1.0) * 2.0), -1.0, 1.0))

        return {
            "faa": faa,
            "high_beta_power": float(high_beta_p),
            "theta_power": float(theta_p),
            "delta_power": float(delta_p),
            "alpha_power": float(alpha_p),
        }

    @staticmethod
    def _band_power(signal: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
        """Estimate band power using Welch PSD (Scipy)."""
        try:
            from scipy.signal import welch
            n_samples = len(signal)
            nperseg = min(256, n_samples)
            freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            if not np.any(idx):
                return 1e-10
            return float(np.trapz(psd[idx], freqs[idx]))
        except Exception:
            # Fallback: simple variance proxy
            return float(np.var(signal) + 1e-10)

    @staticmethod
    def _normalise(value: float, baseline: Optional[float], scale: float = 1.5) -> float:
        """Normalise a value relative to its baseline using tanh compression.

        Without a baseline, returns tanh(value * scale) for bounded output.
        """
        if baseline is None or baseline == 0.0:
            return float(np.clip(np.tanh(value * scale), -1.0, 1.0))
        relative = (value - baseline) / (abs(baseline) + 1e-10)
        return float(np.clip(np.tanh(relative * scale), -1.0, 1.0))
