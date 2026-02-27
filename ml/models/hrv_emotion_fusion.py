"""HRV-EEG Multimodal Emotion Fusion Model.

Combines EEG-derived emotion features (from EmotionClassifier.predict or
extract_features()) with heart-rate variability metrics to produce a more
accurate combined stress and emotion estimate.

HRV Research Basis:
- SDNN (standard deviation of normal-to-normal intervals) is the most robust
  HRV metric for general autonomic nervous system balance.  SDNN < 30 ms
  indicates elevated sympathetic tone (stress / reduced parasympathetic).
- RMSSD (root mean square successive differences) reflects parasympathetic
  activity (vagal tone).  Lower values → higher stress.
- Resting heart rate > 80 bpm is associated with elevated sympathetic
  activation and reduced HRV (Task Force, European Heart Journal, 1996).

Fusion Weights (empirically motivated by literature):
- Stress:  EEG 70 % + HRV 30 %
  EEG beta/alpha and FAA are strong real-time stress indicators.
  HRV captures longer-timescale autonomic state.
- Valence: EEG 60 % + HRV 40 %
  HRV-derived valence proxy is weaker than FAA but adds meaningful
  parasympathetic signal for positive vs. negative affect.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)


# ── HRV threshold constants (Task Force 1996 + meta-analyses) ─────────────────

# SDNN
_SDNN_LOW_STRESS_THRESHOLD  = 50.0   # ms — parasympathetically rich, low stress
_SDNN_HIGH_STRESS_THRESHOLD = 30.0   # ms — autonomic imbalance / high stress

# Resting HR
_HR_LOW_STRESS_THRESHOLD  = 65.0   # bpm — athletic/relaxed baseline
_HR_HIGH_STRESS_THRESHOLD = 80.0   # bpm — elevated sympathetic tone

# RMSSD (optional — used when available for a sharper parasympathetic signal)
_RMSSD_LOW_STRESS_THRESHOLD  = 42.0  # ms
_RMSSD_HIGH_STRESS_THRESHOLD = 20.0  # ms


class HRVEmotionFusion:
    """Fuse EEG emotion features with HRV biometrics.

    Usage::

        fusion = HRVEmotionFusion()

        eeg_result = emotion_model.predict(eeg_array, fs=256)
        hrv_features = {
            "hrv_sdnn": 45.0,
            "resting_heart_rate": 62.0,
            "current_heart_rate": 70.0,
            "hrv_rmssd": 38.0,  # optional
        }
        combined = fusion.predict(eeg_result, hrv_features)
    """

    # Fusion weight constants
    _EEG_STRESS_WEIGHT:  float = 0.70
    _HRV_STRESS_WEIGHT:  float = 0.30
    _EEG_VALENCE_WEIGHT: float = 0.60
    _HRV_VALENCE_WEIGHT: float = 0.40

    def predict(
        self,
        eeg_features: Dict[str, Any],
        hrv_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Blend EEG and HRV signals into a combined emotion/stress result.

        Args:
            eeg_features: Output dict from EmotionClassifier.predict() or
                extract_features().  Expected keys include ``stress_index``,
                ``valence``, ``arousal``, ``focus_index``.
            hrv_features: HRV measurement dict.  Expected keys:
                ``hrv_sdnn`` (ms), ``resting_heart_rate`` (bpm),
                ``current_heart_rate`` (bpm).  Optional key: ``hrv_rmssd``.

        Returns:
            Dict with ``stress_index``, ``valence``, ``arousal``,
            ``focus_index``, ``hrv_stress_component``, ``hrv_contribution``,
            and ``data_quality``.
        """
        eeg_stress  = float(eeg_features.get("stress_index", 0.5))
        eeg_valence = float(eeg_features.get("valence", 0.0))
        eeg_arousal = float(eeg_features.get("arousal", 0.5))
        eeg_focus   = float(eeg_features.get("focus_index", 0.5))

        hrv_stress, hrv_valence, hrv_available, data_quality = (
            self._compute_hrv_components(hrv_features)
        )

        if hrv_available:
            combined_stress  = self._EEG_STRESS_WEIGHT  * eeg_stress  + self._HRV_STRESS_WEIGHT  * hrv_stress
            combined_valence = self._EEG_VALENCE_WEIGHT * eeg_valence + self._HRV_VALENCE_WEIGHT * hrv_valence
            hrv_contribution = self._HRV_STRESS_WEIGHT
        else:
            # No usable HRV data — fall back to EEG-only
            combined_stress  = eeg_stress
            combined_valence = eeg_valence
            hrv_stress       = 0.0
            hrv_valence      = 0.0
            hrv_contribution = 0.0

        combined_stress  = float(max(0.0, min(1.0, combined_stress)))
        combined_valence = float(max(-1.0, min(1.0, combined_valence)))

        return {
            "stress_index":        round(combined_stress, 4),
            "valence":             round(combined_valence, 4),
            "arousal":             round(eeg_arousal, 4),
            "focus_index":         round(eeg_focus, 4),
            "hrv_stress_component": round(hrv_stress, 4),
            "hrv_valence_proxy":   round(hrv_valence, 4),
            "hrv_contribution":    round(hrv_contribution, 4),
            "data_quality":        data_quality,
            "fusion_weights": {
                "eeg_stress":  self._EEG_STRESS_WEIGHT,
                "hrv_stress":  self._HRV_STRESS_WEIGHT  if hrv_available else 0.0,
                "eeg_valence": self._EEG_VALENCE_WEIGHT,
                "hrv_valence": self._HRV_VALENCE_WEIGHT if hrv_available else 0.0,
            },
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _compute_hrv_components(
        self,
        hrv: Dict[str, Any],
    ) -> Tuple[float, float, bool, Dict[str, Any]]:
        """Derive HRV stress (0–1) and valence proxy (–1 to 1) from raw HRV dict.

        Returns:
            Tuple of (hrv_stress, hrv_valence_proxy, hrv_available, data_quality).
        """
        sdnn          = hrv.get("hrv_sdnn")
        resting_hr    = hrv.get("resting_heart_rate")
        current_hr    = hrv.get("current_heart_rate")
        rmssd         = hrv.get("hrv_rmssd")

        sdnn_valid       = sdnn is not None and sdnn > 0
        resting_hr_valid = resting_hr is not None and resting_hr > 0
        current_hr_valid = current_hr is not None and current_hr > 0
        rmssd_valid      = rmssd is not None and rmssd > 0

        n_valid = sum([sdnn_valid, resting_hr_valid, current_hr_valid])
        if n_valid == 0:
            return 0.0, 0.0, False, self._build_quality(
                sdnn_valid, resting_hr_valid, current_hr_valid, rmssd_valid
            )

        stress_votes:  list[float] = []
        valence_votes: list[float] = []

        # ── SDNN component ───────────────────────────────────────────────────
        if sdnn_valid:
            sdnn_f   = float(sdnn)
            # Stress: low SDNN → high stress; normalise to 0–1
            # Piecewise linear: sdnn ≤ 30 → stress=1.0; sdnn ≥ 50 → stress=0.0
            if sdnn_f <= _SDNN_HIGH_STRESS_THRESHOLD:
                sdnn_stress = 1.0
            elif sdnn_f >= _SDNN_LOW_STRESS_THRESHOLD:
                sdnn_stress = 0.0
            else:
                sdnn_stress = 1.0 - (sdnn_f - _SDNN_HIGH_STRESS_THRESHOLD) / (
                    _SDNN_LOW_STRESS_THRESHOLD - _SDNN_HIGH_STRESS_THRESHOLD
                )
            stress_votes.append(sdnn_stress)

            # Valence proxy: high SDNN → positive valence (parasympathetic = approach)
            # Map: sdnn=20 → -1.0; sdnn=50 → 0.0; sdnn=80 → +1.0 (linear)
            sdnn_valence = (sdnn_f - 50.0) / 30.0
            valence_votes.append(max(-1.0, min(1.0, sdnn_valence)))

        # ── Resting HR component ─────────────────────────────────────────────
        if resting_hr_valid:
            hr_f = float(resting_hr)
            # Stress: hr ≥ 80 → 1.0; hr ≤ 65 → 0.0; linear between
            if hr_f >= _HR_HIGH_STRESS_THRESHOLD:
                hr_stress = 1.0
            elif hr_f <= _HR_LOW_STRESS_THRESHOLD:
                hr_stress = 0.0
            else:
                hr_stress = (hr_f - _HR_LOW_STRESS_THRESHOLD) / (
                    _HR_HIGH_STRESS_THRESHOLD - _HR_LOW_STRESS_THRESHOLD
                )
            stress_votes.append(hr_stress)

            # HR valence proxy: low resting HR → positive (relaxed / parasympathetic)
            hr_valence = -(hr_f - 72.5) / 15.0  # centred at 72.5 bpm
            valence_votes.append(max(-1.0, min(1.0, hr_valence)))

        # ── Current HR elevation (relative to resting) ───────────────────────
        if current_hr_valid and resting_hr_valid:
            excess = float(current_hr) - float(resting_hr)
            # Each 10 bpm above resting ≈ 0.2 stress increment
            hr_elevation_stress = max(0.0, min(1.0, excess / 50.0))
            stress_votes.append(hr_elevation_stress)

        # ── RMSSD (optional, higher precision parasympathetic signal) ─────────
        if rmssd_valid:
            rmssd_f = float(rmssd)
            if rmssd_f <= _RMSSD_HIGH_STRESS_THRESHOLD:
                rmssd_stress = 1.0
            elif rmssd_f >= _RMSSD_LOW_STRESS_THRESHOLD:
                rmssd_stress = 0.0
            else:
                rmssd_stress = 1.0 - (rmssd_f - _RMSSD_HIGH_STRESS_THRESHOLD) / (
                    _RMSSD_LOW_STRESS_THRESHOLD - _RMSSD_HIGH_STRESS_THRESHOLD
                )
            stress_votes.append(rmssd_stress)
            rmssd_valence = (rmssd_f - 30.0) / 20.0
            valence_votes.append(max(-1.0, min(1.0, rmssd_valence)))

        hrv_stress  = float(sum(stress_votes)  / len(stress_votes))
        hrv_valence = float(sum(valence_votes) / len(valence_votes))

        return (
            hrv_stress,
            hrv_valence,
            True,
            self._build_quality(sdnn_valid, resting_hr_valid, current_hr_valid, rmssd_valid),
        )

    @staticmethod
    def _build_quality(
        sdnn_ok: bool,
        resting_hr_ok: bool,
        current_hr_ok: bool,
        rmssd_ok: bool,
    ) -> Dict[str, Any]:
        n_fields = sum([sdnn_ok, resting_hr_ok, current_hr_ok, rmssd_ok])
        if n_fields == 0:
            level = "eeg_only"
        elif n_fields <= 2:
            level = "partial_hrv"
        else:
            level = "full_hrv"
        return {
            "level":          level,
            "sdnn_available":       sdnn_ok,
            "resting_hr_available": resting_hr_ok,
            "current_hr_available": current_hr_ok,
            "rmssd_available":      rmssd_ok,
            "n_hrv_fields":         n_fields,
        }

    # ── Class method: payload extraction ──────────────────────────────────────

    @classmethod
    def from_health_data(cls, health_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract HRV features from an Apple Health or Google Fit payload.

        Supports two payload shapes:

        Apple HealthKit (from apple_health.parse_healthkit_payload / raw export)::

            {
                "source": "apple_health",
                "data": {
                    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": <float>,
                    "HKQuantityTypeIdentifierRestingHeartRate": <float>,
                    "HKQuantityTypeIdentifierHeartRate": <float>,
                }
            }

        Google Fit / Health Connect::

            {
                "source": "google_fit",          # or "health_connect"
                "data": {
                    "com.google.heart_rate.variability": <float>,
                    "com.google.heart_rate.bpm": <float>,
                    "HRVSDNNRecord": <float>,      # Health Connect variant
                    "HeartRateRecord": <float>,    # Health Connect variant
                }
            }

        Also accepts our normalised internal format (field names are the
        keys we use internally: ``hrv_sdnn``, ``resting_heart_rate``, etc.)

        Returns:
            HRV features dict ready to pass to ``HRVEmotionFusion.predict()``.
        """
        source = health_payload.get("source", "").lower()
        data   = health_payload.get("data", health_payload)

        hrv_features: Dict[str, Optional[float]] = {
            "hrv_sdnn":            None,
            "resting_heart_rate":  None,
            "current_heart_rate":  None,
            "hrv_rmssd":           None,
        }

        def _safe_float(val: Any) -> Optional[float]:
            try:
                return float(val) if val is not None else None
            except (TypeError, ValueError):
                return None

        if source == "apple_health":
            hrv_features["hrv_sdnn"]           = _safe_float(data.get("HKQuantityTypeIdentifierHeartRateVariabilitySDNN"))
            hrv_features["resting_heart_rate"]  = _safe_float(data.get("HKQuantityTypeIdentifierRestingHeartRate"))
            hrv_features["current_heart_rate"]  = _safe_float(data.get("HKQuantityTypeIdentifierHeartRate"))

        elif source in ("google_fit", "health_connect"):
            # Google Fit data type strings
            hrv_features["hrv_sdnn"]          = _safe_float(data.get("com.google.heart_rate.variability"))
            hrv_features["current_heart_rate"] = _safe_float(data.get("com.google.heart_rate.bpm"))
            # Health Connect (Android 14+) record-style keys
            if hrv_features["hrv_sdnn"] is None:
                hrv_features["hrv_sdnn"]       = _safe_float(data.get("HRVSDNNRecord"))
            if hrv_features["current_heart_rate"] is None:
                hrv_features["current_heart_rate"] = _safe_float(data.get("HeartRateRecord"))
            hrv_features["resting_heart_rate"] = _safe_float(data.get("resting_heart_rate"))

        else:
            # Assume normalised internal format or plain dict with our field names
            hrv_features["hrv_sdnn"]           = _safe_float(data.get("hrv_sdnn"))
            hrv_features["resting_heart_rate"]  = _safe_float(data.get("resting_heart_rate"))
            hrv_features["current_heart_rate"]  = _safe_float(data.get("current_heart_rate") or data.get("heart_rate"))
            hrv_features["hrv_rmssd"]           = _safe_float(data.get("hrv_rmssd"))

        # Strip None values so the predict() call can use .get() with defaults
        return {k: v for k, v in hrv_features.items() if v is not None}
