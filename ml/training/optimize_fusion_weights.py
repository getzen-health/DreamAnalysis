"""Fusion weight optimizer for multimodal HRV + voice + activity emotion detection.

Discovers per-emotion optimal fusion weights from:
- Research-validated hypotheses (voice, hrv, activity, sleep priors)
- Grid search over constrained weight space (sum ≈ 1.0)

Saves: ml/models/saved/fusion_weights.json

Usage:
    cd ml
    python3 training/optimize_fusion_weights.py

Output:
    models/saved/fusion_weights.json  — per-emotion weight dicts
    stdout — table of selected weights + cosine similarity scores
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Weight grid
# Each dimension is sampled independently; only combos that sum to ~1.0
# (within TOLERANCE) are considered valid.
# ---------------------------------------------------------------------------

VOICE_GRID:    List[float] = [0.30, 0.40, 0.50, 0.60, 0.70]
HRV_GRID:      List[float] = [0.10, 0.20, 0.30, 0.40]
ACTIVITY_GRID: List[float] = [0.05, 0.10, 0.15, 0.20]
SLEEP_GRID:    List[float] = [0.05, 0.10, 0.15, 0.20]
TOLERANCE:      float      = 0.01   # |sum - 1.0| ≤ TOLERANCE

# ---------------------------------------------------------------------------
# Research-validated priors per emotion
#
# Basis:
#   voice:    prosody (F0, jitter, shimmer, energy) is the most direct
#             real-time emotion signal; strongest for happy/sad/stress.
#   hrv:      autonomic arousal via SDNN/RMSSD — best for stress/anxiety
#             (high-beta ↔ low-HRV coupling); weak for positive affect.
#   activity: physical state (steps, movement) modulates energy/mood;
#             secondary signal for stress and happy, minimal for sad.
#   sleep:    prior-night quality has a delayed effect; strongest for
#             sadness and cognitive focus; weak for acute anxiety.
#
# References:
#   - Kreibig 2010 (ANS in emotion): HRV drops sharply for stress/anxiety.
#   - El Ayadi et al. 2011 (voice emotion survey): voice leads all modalities.
#   - Steptoe & Kivimäki 2012 (sleep & mood): prior-night sleep → next-day
#     affect, strongest for depressive/sad states.
#   - Sano & Picard 2013 (wearable stress): activity + HRV fusion > either alone.
# ---------------------------------------------------------------------------

_PRIORS: Dict[str, Dict[str, float]] = {
    "stress":  {"voice": 0.40, "hrv": 0.35, "activity": 0.10, "sleep": 0.15},
    "happy":   {"voice": 0.60, "hrv": 0.15, "activity": 0.15, "sleep": 0.10},
    "sad":     {"voice": 0.55, "hrv": 0.20, "activity": 0.05, "sleep": 0.20},
    "focus":   {"voice": 0.30, "hrv": 0.25, "activity": 0.10, "sleep": 0.35},
    "anxiety": {"voice": 0.45, "hrv": 0.30, "activity": 0.10, "sleep": 0.15},
    "neutral": {"voice": 0.40, "hrv": 0.25, "activity": 0.15, "sleep": 0.20},
    "default": {"voice": 0.50, "hrv": 0.25, "activity": 0.10, "sleep": 0.15},
}

# Canonical key order — must match between all dicts for cosine similarity.
_KEYS: List[str] = ["voice", "hrv", "activity", "sleep"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dot(a: Dict[str, float], b: Dict[str, float]) -> float:
    return sum(a[k] * b[k] for k in _KEYS)


def _norm(a: Dict[str, float]) -> float:
    return math.sqrt(sum(a[k] ** 2 for k in _KEYS))


def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity between two weight dicts (keys in _KEYS)."""
    denom = _norm(a) * _norm(b)
    if denom == 0.0:
        return 0.0
    return _dot(a, b) / denom


def _generate_valid_combos() -> List[Dict[str, float]]:
    """Return all (voice, hrv, activity, sleep) combos whose sum is within TOLERANCE of 1.0."""
    combos: List[Dict[str, float]] = []
    for v in VOICE_GRID:
        for h in HRV_GRID:
            for a in ACTIVITY_GRID:
                for s in SLEEP_GRID:
                    total = v + h + a + s
                    if abs(total - 1.0) <= TOLERANCE:
                        combos.append(
                            {"voice": v, "hrv": h, "activity": a, "sleep": s}
                        )
    return combos


def _best_combo(
    combos: List[Dict[str, float]],
    prior: Dict[str, float],
) -> Tuple[Dict[str, float], float]:
    """Return the combo with the highest cosine similarity to *prior*."""
    best_combo = combos[0]
    best_score = _cosine_similarity(combos[0], prior)
    for combo in combos[1:]:
        score = _cosine_similarity(combo, prior)
        if score > best_score:
            best_score = score
            best_combo = combo
    return best_combo, best_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def optimize() -> Dict[str, Dict[str, float]]:
    """Run grid search and return the best weights per emotion."""
    combos = _generate_valid_combos()
    print(f"Valid weight combinations: {len(combos)}")
    print()

    results: Dict[str, Dict[str, float]] = {}
    header = f"{'emotion':>10}  {'voice':>6}  {'hrv':>6}  {'activity':>8}  {'sleep':>6}  {'cosine':>8}"
    print(header)
    print("-" * len(header))

    for emotion, prior in _PRIORS.items():
        best, score = _best_combo(combos, prior)
        results[emotion] = best
        print(
            f"{emotion:>10}  {best['voice']:>6.2f}  {best['hrv']:>6.2f}"
            f"  {best['activity']:>8.2f}  {best['sleep']:>6.2f}  {score:>8.4f}"
        )

    return results


def save(weights: Dict[str, Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(weights, fh, indent=2)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    # Resolve output path relative to this script's location:
    # ml/training/optimize_fusion_weights.py → ml/models/saved/fusion_weights.json
    script_dir = Path(__file__).resolve().parent          # ml/training/
    ml_dir     = script_dir.parent                        # ml/
    output     = ml_dir / "models" / "saved" / "fusion_weights.json"

    weights = optimize()
    save(weights, output)
