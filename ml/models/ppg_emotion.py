"""PPG-based emotion / physiological state estimation.

Maps HR, HRV, and respiratory rate features to:
  - arousal (0-1): calm ↔ activated
  - valence (-1 to 1): negative ↔ positive
  - stress (0-1): relaxed ↔ stressed
  - autonomic state: parasympathetic / balanced / sympathetic

Feature-based heuristics — no training data required.

Physiological grounding:
  - Sympathetic activation (stress, fear, anger): high HR, low HRV, high LF/HF
  - Parasympathetic dominance (calm, relaxed): low HR, high RMSSD/pNN50, low LF/HF
  - Arousal (excited, energized): high HR, moderate-high LF
  - Anxiety/stress: very high LF/HF, high HR, low HRV, high respiratory rate
  - Relaxation/calm: low LF/HF (<0.5), high RMSSD, normal respiratory rate

References:
  - Task Force ESC/NASPE (1996): HRV standard measures
  - McCraty et al. (2009): HRV and emotional states
  - Kim & André (2008): physiological signals for emotion recognition
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

log = logging.getLogger(__name__)


class PPGEmotionModel:
    """Map PPG-derived physiological features to emotional state estimates.

    This is a rule-based model using population-average thresholds.
    It does not require a trained model file.

    Population reference ranges (healthy adults at rest):
      HR:       60-80 BPM
      SDNN:     50-100 ms
      RMSSD:    20-60 ms
      pNN50:    3-20%
      LF/HF:    0.5-2.0 (balanced), >2.0 (sympathetic), <0.5 (parasympathetic)
      Resp:     12-20 breaths/min
    """

    # Population reference ranges used for normalization
    _HR_REST: float = 70.0    # BPM
    _HR_MAX_STRESS: float = 130.0  # BPM at which stress score saturates

    _RMSSD_CALM: float = 50.0   # ms — high vagal tone
    _RMSSD_STRESS: float = 10.0  # ms — low vagal tone

    _RESP_NORMAL: float = 15.0   # breaths/min
    _RESP_ANXIOUS: float = 25.0  # breaths/min

    def predict(self, hrv_features: Dict[str, float]) -> Dict[str, float]:
        """Estimate emotional state from HRV features.

        Args:
            hrv_features: Output from ppg_processor.extract_hrv_features().

        Returns:
            dict with:
              arousal (0-1), valence (-1 to 1), stress (0-1),
              autonomic_state ("parasympathetic" | "balanced" | "sympathetic"),
              confidence (0-1, based on signal quality and data sufficiency),
              and an explanation string.
        """
        # Extract key features with safe defaults
        mean_hr = float(hrv_features.get("mean_hr", 0.0))
        rmssd = float(hrv_features.get("rmssd", 0.0))
        pnn50 = float(hrv_features.get("pnn50", 0.0))
        lf_hf = float(hrv_features.get("lf_hf_ratio", 1.0))
        resp_rate = float(hrv_features.get("respiratory_rate", 0.0))
        sqi = float(hrv_features.get("sqi", 0.0))
        n_beats = int(hrv_features.get("n_beats", 0))

        # Insufficient data — return null estimate
        if mean_hr < 20.0 or n_beats < 5:
            return self._null_result("insufficient PPG data")

        # ── Confidence ────────────────────────────────────────────────────────
        # Scales with signal quality and beat count (saturates at 60 beats ~ 1 min)
        quality_conf = float(np.clip(sqi, 0.0, 1.0))
        count_conf = float(np.clip(n_beats / 60.0, 0.0, 1.0))
        confidence = round(0.6 * quality_conf + 0.4 * count_conf, 3)

        # ── Arousal ───────────────────────────────────────────────────────────
        # High HR + low HRV + high LF → high arousal
        hr_arousal = float(np.clip((mean_hr - self._HR_REST) / (self._HR_MAX_STRESS - self._HR_REST), 0.0, 1.0))

        rmssd_arousal = 1.0 - float(np.clip(rmssd / self._RMSSD_CALM, 0.0, 1.0))  # high rmssd = low arousal

        lf_hf_arousal = float(np.clip((lf_hf - 0.5) / 3.5, 0.0, 1.0))  # 0.5→0, 4.0→1

        if resp_rate > 0:
            resp_arousal = float(np.clip((resp_rate - self._RESP_NORMAL) / (self._RESP_ANXIOUS - self._RESP_NORMAL), 0.0, 1.0))
        else:
            resp_arousal = 0.3  # neutral if respiratory rate unknown

        # Weighted combination (HR most reliable from PPG)
        arousal = float(np.clip(
            0.35 * hr_arousal + 0.30 * rmssd_arousal + 0.25 * lf_hf_arousal + 0.10 * resp_arousal,
            0.0, 1.0,
        ))

        # ── Stress ────────────────────────────────────────────────────────────
        # Stress is arousal + negative valence context: high LF/HF + low RMSSD + high HR
        # (differs from arousal by weighting LF/HF more heavily)
        hr_stress = float(np.clip((mean_hr - 60.0) / 60.0, 0.0, 1.0))
        rmssd_stress = 1.0 - float(np.clip(rmssd / 80.0, 0.0, 1.0))
        lf_hf_stress = float(np.clip((lf_hf - 1.0) / 4.0, 0.0, 1.0))  # balanced=1.0, stressed=5.0
        stress = float(np.clip(
            0.35 * hr_stress + 0.35 * rmssd_stress + 0.30 * lf_hf_stress,
            0.0, 1.0,
        ))

        # ── Valence ───────────────────────────────────────────────────────────
        # PPG/HRV is a better arousal predictor than valence predictor.
        # Heuristic: very high stress → negative valence; very low arousal → slightly positive.
        # High pNN50 (strong vagal tone) correlates with positive affect.
        # LF/HF < 0.5 (parasympathetic dominance) → positive valence / calm.
        pnn50_valence = float(np.clip(pnn50 * 4.0, 0.0, 1.0))   # pNN50=0.25 → valence=+1
        stress_valence = -float(np.clip(stress * 1.5, 0.0, 1.0))  # stress → negative
        # Blend: 60% stress-based (more reliable), 40% pNN50-based
        valence = float(np.clip(0.60 * stress_valence + 0.40 * pnn50_valence, -1.0, 1.0))

        # ── Autonomic state ───────────────────────────────────────────────────
        if lf_hf < 0.5 and rmssd > 40.0:
            autonomic_state = "parasympathetic"
        elif lf_hf > 2.5 or (mean_hr > 90 and rmssd < 20.0):
            autonomic_state = "sympathetic"
        else:
            autonomic_state = "balanced"

        # ── Explanation string ────────────────────────────────────────────────
        explanation = self._build_explanation(
            mean_hr=mean_hr,
            rmssd=rmssd,
            lf_hf=lf_hf,
            resp_rate=resp_rate,
            autonomic_state=autonomic_state,
        )

        return {
            "arousal": round(arousal, 3),
            "valence": round(valence, 3),
            "stress": round(stress, 3),
            "autonomic_state": autonomic_state,
            "confidence": confidence,
            "explanation": explanation,
        }

    def _build_explanation(
        self,
        mean_hr: float,
        rmssd: float,
        lf_hf: float,
        resp_rate: float,
        autonomic_state: str,
    ) -> str:
        parts = []
        if mean_hr > 90:
            parts.append(f"elevated HR ({mean_hr:.0f} BPM)")
        elif mean_hr > 0:
            parts.append(f"normal HR ({mean_hr:.0f} BPM)")

        if rmssd < 20.0 and rmssd > 0:
            parts.append("low HRV (reduced vagal tone)")
        elif rmssd > 50.0:
            parts.append("high HRV (strong vagal tone)")

        if lf_hf > 2.5:
            parts.append(f"high LF/HF ({lf_hf:.1f}, sympathetic dominance)")
        elif lf_hf < 0.5 and lf_hf > 0:
            parts.append(f"low LF/HF ({lf_hf:.2f}, parasympathetic dominance)")

        if resp_rate > 20:
            parts.append(f"elevated resp rate ({resp_rate:.0f} br/min)")

        state_label = {
            "parasympathetic": "calm/relaxed autonomic state",
            "sympathetic": "activated/stressed autonomic state",
            "balanced": "balanced autonomic tone",
        }.get(autonomic_state, autonomic_state)
        parts.append(state_label)

        return "; ".join(parts) if parts else "normal physiological state"

    @staticmethod
    def _null_result(reason: str) -> Dict[str, float]:
        return {
            "arousal": 0.0,
            "valence": 0.0,
            "stress": 0.0,
            "autonomic_state": "unknown",
            "confidence": 0.0,
            "explanation": reason,
        }
