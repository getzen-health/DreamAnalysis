"""
Attention-based multimodal fusion for emotion recognition.
Fuses EEG, voice, and health (HRV) modality predictions using
learned attention weights based on signal quality and modality agreement.

Instead of fixed 70/30 EEG/Voice weights, this computes dynamic weights
per prediction based on:
1. Per-modality confidence (model softmax entropy)
2. Signal quality (SQI for EEG, SNR for voice)
3. Cross-modality agreement (do EEG and voice agree on valence direction?)
"""
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class ModalityInput:
    """Prediction from one modality."""
    probabilities: Dict[str, float]  # 6-class emotion probs
    valence: float                    # -1 to 1
    arousal: float                    # 0 to 1
    confidence: float                 # 0-1 (model confidence)
    signal_quality: float             # 0-1 (signal quality metric)
    modality: str                     # "eeg", "voice", "health"


@dataclass
class FusionResult:
    probabilities: Dict[str, float]   # fused 6-class probs
    valence: float
    arousal: float
    dominant_emotion: str
    confidence: float
    weights_used: Dict[str, float]    # which modality contributed how much
    agreement_score: float            # 0-1, how much modalities agree


class AttentionFusion:
    """Dynamic attention-based multimodal fusion.

    Computes per-modality attention weights from three signals:
      - signal_quality: hardware/sensor quality metric (SQI, SNR)
      - confidence: model output confidence (softmax certainty)
      - agreement: cross-modality valence direction agreement

    Weights are softmax-normalized so they sum to 1.0.
    """

    EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "neutral"]

    def __init__(
        self,
        quality_weight: float = 0.4,
        confidence_weight: float = 0.4,
        agreement_weight: float = 0.2,
    ):
        if quality_weight < 0 or confidence_weight < 0 or agreement_weight < 0:
            raise ValueError("Component weights must be non-negative.")
        self.quality_weight = quality_weight
        self.confidence_weight = confidence_weight
        self.agreement_weight = agreement_weight

    # ── public API ───────────────────────────────────────────────────────

    def fuse(self, inputs: List[ModalityInput]) -> FusionResult:
        """Fuse predictions from multiple modalities using attention weights.

        Args:
            inputs: One or more ModalityInput instances. At least one is required.

        Returns:
            FusionResult with fused probabilities, valence, arousal,
            dominant emotion, confidence, weight breakdown, and agreement.

        Raises:
            ValueError: If *inputs* is empty.
        """
        if not inputs:
            raise ValueError("At least one modality input is required.")

        # Single modality: pass through unchanged
        if len(inputs) == 1:
            inp = inputs[0]
            dominant = max(inp.probabilities, key=inp.probabilities.get)
            return FusionResult(
                probabilities=dict(inp.probabilities),
                valence=inp.valence,
                arousal=inp.arousal,
                dominant_emotion=dominant,
                confidence=inp.confidence,
                weights_used={inp.modality: 1.0},
                agreement_score=1.0,
            )

        # Multi-modality path
        attention = self._compute_attention_weights(inputs)
        agreement = self._compute_agreement(inputs)

        # Weighted fusion of probabilities
        fused_probs: Dict[str, float] = {e: 0.0 for e in self.EMOTIONS}
        for inp in inputs:
            w = attention[inp.modality]
            for e in self.EMOTIONS:
                fused_probs[e] += w * inp.probabilities.get(e, 0.0)

        # Normalize fused probs to sum to 1 (guards against floating-point drift)
        prob_sum = sum(fused_probs.values())
        if prob_sum > 0:
            fused_probs = {e: v / prob_sum for e, v in fused_probs.items()}

        # Weighted fusion of valence and arousal
        fused_valence = sum(attention[inp.modality] * inp.valence for inp in inputs)
        fused_arousal = sum(attention[inp.modality] * inp.arousal for inp in inputs)

        # Clamp to valid ranges
        fused_valence = float(np.clip(fused_valence, -1.0, 1.0))
        fused_arousal = float(np.clip(fused_arousal, 0.0, 1.0))

        # Weighted confidence
        fused_confidence = sum(
            attention[inp.modality] * inp.confidence for inp in inputs
        )
        fused_confidence = float(np.clip(fused_confidence, 0.0, 1.0))

        dominant = max(fused_probs, key=fused_probs.get)

        return FusionResult(
            probabilities={e: round(v, 6) for e, v in fused_probs.items()},
            valence=round(fused_valence, 6),
            arousal=round(fused_arousal, 6),
            dominant_emotion=dominant,
            confidence=round(fused_confidence, 6),
            weights_used={k: round(v, 6) for k, v in attention.items()},
            agreement_score=round(agreement, 6),
        )

    # ── internals ────────────────────────────────────────────────────────

    def _compute_attention_weights(
        self, inputs: List[ModalityInput]
    ) -> Dict[str, float]:
        """Compute per-modality attention weights via softmax over raw scores.

        For each modality *i*:
            raw_i = quality_weight * signal_quality_i
                  + confidence_weight * confidence_i
                  + agreement_weight * agreement_i

        where *agreement_i* is the per-modality agreement bonus: 1.0 when that
        modality's valence direction matches the majority, 0.0 otherwise.

        The raw scores are then softmax-normalized to produce weights that
        sum to 1.0.
        """
        agreement_global = self._compute_agreement(inputs)

        # Per-modality agreement bonus: does this modality agree with the
        # majority valence direction?
        majority_sign = self._majority_valence_sign(inputs)
        per_modality_agreement: Dict[str, float] = {}
        for inp in inputs:
            if majority_sign == 0:
                # All near-zero: treat as full agreement
                per_modality_agreement[inp.modality] = 1.0
            else:
                agrees = (np.sign(inp.valence) == majority_sign) or abs(inp.valence) < 0.05
                per_modality_agreement[inp.modality] = 1.0 if agrees else 0.0

        # Raw attention scores
        raw_scores: Dict[str, float] = {}
        for inp in inputs:
            score = (
                self.quality_weight * inp.signal_quality
                + self.confidence_weight * inp.confidence
                + self.agreement_weight * per_modality_agreement[inp.modality]
            )
            raw_scores[inp.modality] = score

        # Softmax normalization
        modalities = list(raw_scores.keys())
        scores_array = np.array([raw_scores[m] for m in modalities], dtype=np.float64)
        # Subtract max for numerical stability
        scores_array = scores_array - np.max(scores_array)
        exp_scores = np.exp(scores_array)
        softmax_weights = exp_scores / np.sum(exp_scores)

        return {m: float(softmax_weights[i]) for i, m in enumerate(modalities)}

    def _compute_agreement(self, inputs: List[ModalityInput]) -> float:
        """How much do modalities agree on valence direction?

        Returns a score in [0, 1]:
          - 1.0: all modalities agree on the same valence sign (or all near zero)
          - 0.0-0.5: mixed signs, score decreases with more disagreement

        Uses valence sign consensus: fraction of modalities that agree with
        the majority sign, scaled to [0, 1].
        """
        if len(inputs) <= 1:
            return 1.0

        valences = [inp.valence for inp in inputs]

        # Classify each valence as positive, negative, or near-zero
        signs = []
        for v in valences:
            if abs(v) < 0.05:
                signs.append(0)
            elif v > 0:
                signs.append(1)
            else:
                signs.append(-1)

        # If all near-zero, full agreement
        non_zero = [s for s in signs if s != 0]
        if not non_zero:
            return 1.0

        # Count agreement with majority non-zero sign
        pos_count = sum(1 for s in non_zero if s > 0)
        neg_count = sum(1 for s in non_zero if s < 0)
        majority_count = max(pos_count, neg_count)

        # Agreement = fraction of non-zero signs that match majority
        # Near-zero modalities are treated as agreeing
        n_near_zero = len(signs) - len(non_zero)
        n_agree = majority_count + n_near_zero
        agreement = n_agree / len(inputs)

        return float(agreement)

    @staticmethod
    def _majority_valence_sign(inputs: List[ModalityInput]) -> int:
        """Return +1, -1, or 0 for the majority valence direction."""
        pos = sum(1 for inp in inputs if inp.valence > 0.05)
        neg = sum(1 for inp in inputs if inp.valence < -0.05)
        if pos > neg:
            return 1
        elif neg > pos:
            return -1
        return 0
