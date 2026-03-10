"""Ground-truth collector for the no-EEG benchmark.

Logs one labeled check-in at a time to benchmarks/ground_truth.jsonl.
Each entry records:
  - Your self-reported emotion label + valence/arousal/stress
  - The model's predictions for each modality (voice, health, combined)

Usage
-----
# Interactive (prompts you for labels):
python benchmarks/collect_ground_truth.py --interactive

# Batch-log from a JSON file of pre-labeled samples:
python benchmarks/collect_ground_truth.py --batch samples.json

# Log a single sample non-interactively:
python benchmarks/collect_ground_truth.py \
    --label-emotion happy --label-valence 0.7 --label-arousal 0.6 \
    --voice-file /path/to/clip.wav --user-id dev

Ground-truth file: ml/benchmarks/ground_truth.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

_HERE = Path(__file__).parent
_GT_FILE = _HERE / "ground_truth.jsonl"

EMOTION_CHOICES = ["happy", "sad", "angry", "fear", "neutral", "calm"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ask_choice(prompt: str, choices: list) -> str:
    while True:
        val = input(f"{prompt} [{'/'.join(choices)}]: ").strip().lower()
        if val in choices:
            return val
        print(f"  Please enter one of: {choices}")


def _ask_float(prompt: str, lo: float, hi: float) -> float:
    while True:
        raw = input(f"{prompt} [{lo}–{hi}]: ").strip()
        try:
            v = float(raw)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"  Enter a number between {lo} and {hi}")


def _ask_bool(prompt: str) -> bool:
    while True:
        raw = input(f"{prompt} [y/n]: ").strip().lower()
        if raw in ("y", "yes", "true", "1"):
            return True
        if raw in ("n", "no", "false", "0"):
            return False


def _call_voice_model(voice_file: Optional[str], user_id: str) -> Dict[str, Any]:
    """Call the local voice emotion model if possible; return empty dict on failure."""
    if not voice_file:
        return {}
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models.voice_emotion_model import VoiceEmotionModel  # type: ignore
        import numpy as np

        model = VoiceEmotionModel()
        result = model.predict(voice_file)
        return {
            "voice_emotion": result.get("emotion", "neutral"),
            "voice_valence": result.get("valence", 0.0),
            "voice_arousal": result.get("arousal", 0.5),
        }
    except Exception as exc:
        print(f"  [warn] Voice model unavailable: {exc}")
        return {}


def _log_sample(sample: Dict[str, Any], gt_file: Path = _GT_FILE) -> None:
    gt_file.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_file, "a") as f:
        f.write(json.dumps(sample) + "\n")
    print(f"  Logged to {gt_file}")


# ---------------------------------------------------------------------------
# Interactive collection
# ---------------------------------------------------------------------------


def collect_interactive(user_id: str = "dev", voice_file: Optional[str] = None) -> None:
    print("\n=== No-EEG Ground-Truth Collection ===")
    print(f"User: {user_id}  |  Samples in {_GT_FILE}")
    print()

    label_emotion = _ask_choice("Your current emotion", EMOTION_CHOICES)
    label_valence = _ask_float("Your valence (-1 = very negative, +1 = very positive)", -1.0, 1.0)
    label_arousal = _ask_float("Your arousal (0 = calm, 1 = highly energetic)", 0.0, 1.0)
    label_stress  = _ask_bool("Are you stressed right now?")

    sample: Dict[str, Any] = {
        "timestamp":     time.time(),
        "user_id":       user_id,
        "label_emotion": label_emotion,
        "label_valence": label_valence,
        "label_arousal": label_arousal,
        "label_stress":  label_stress,
    }

    voice_preds = _call_voice_model(voice_file, user_id)
    sample.update(voice_preds)

    _log_sample(sample)
    print(f"\n  Logged: emotion={label_emotion}, valence={label_valence:.2f}, arousal={label_arousal:.2f}, stress={label_stress}")

    # Show running count
    if _GT_FILE.exists():
        count = sum(1 for _ in open(_GT_FILE))
        print(f"  Total samples collected: {count}")


def collect_batch(batch_file: str) -> None:
    """Log all samples from a pre-labeled JSON array file."""
    with open(batch_file) as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        samples = [samples]

    for s in samples:
        if "timestamp" not in s:
            s["timestamp"] = time.time()
        _log_sample(s)

    print(f"Logged {len(samples)} samples from {batch_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect ground-truth labels for no-EEG benchmark")
    parser.add_argument("--user-id", default="dev", help="User identifier")
    parser.add_argument("--interactive", action="store_true", help="Interactive label collection")
    parser.add_argument("--batch", metavar="FILE", help="Batch-log from a JSON file")
    parser.add_argument("--voice-file", metavar="WAV", help="Path to voice clip to analyze")
    parser.add_argument("--label-emotion", choices=EMOTION_CHOICES, help="Emotion label (non-interactive)")
    parser.add_argument("--label-valence", type=float, help="Valence label (non-interactive)")
    parser.add_argument("--label-arousal", type=float, help="Arousal label (non-interactive)")
    parser.add_argument("--label-stress", type=lambda x: x.lower() in ("true", "1", "yes"),
                        default=False, help="Stress label (non-interactive)")
    args = parser.parse_args()

    if args.batch:
        collect_batch(args.batch)
    elif args.interactive or args.label_emotion is None:
        collect_interactive(user_id=args.user_id, voice_file=args.voice_file)
    else:
        # Non-interactive single sample
        sample: Dict[str, Any] = {
            "timestamp":     time.time(),
            "user_id":       args.user_id,
            "label_emotion": args.label_emotion,
            "label_valence": args.label_valence or 0.0,
            "label_arousal": args.label_arousal or 0.5,
            "label_stress":  args.label_stress,
        }
        voice_preds = _call_voice_model(args.voice_file, args.user_id)
        sample.update(voice_preds)
        _log_sample(sample)
