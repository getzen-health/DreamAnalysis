"""No-EEG accuracy benchmark API endpoints.

Endpoints
---------
POST /no-eeg-benchmark/log
    Log one labeled ground-truth sample (for ongoing 30-day collection).

GET  /no-eeg-benchmark/run
    Run the modality-ablation benchmark against all logged samples.
    Returns accuracy table across: voice-only, health-only, voice+health, etc.

GET  /no-eeg-benchmark/status
    Return number of samples collected and estimated next-run readiness.

GET  /no-eeg-benchmark/results
    Return the most recently saved benchmark results JSON.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/no-eeg-benchmark", tags=["No-EEG Benchmark"])

_BENCHMARKS_DIR = Path(__file__).parent.parent.parent / "benchmarks"
_GT_FILE        = _BENCHMARKS_DIR / "ground_truth.jsonl"
_RESULTS_FILE   = _BENCHMARKS_DIR / "no_eeg_benchmark_results.json"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class GroundTruthSample(BaseModel):
    """One labeled check-in for the no-EEG benchmark ground-truth corpus."""
    user_id: str = Field(default="dev")
    timestamp: Optional[float] = Field(default=None, description="Unix timestamp (defaults to now)")

    # Ground-truth labels
    label_emotion: str = Field(
        description="Self-reported emotion: happy/sad/angry/fear/neutral/calm"
    )
    label_valence: float = Field(description="Self-reported valence (-1 to +1)")
    label_arousal: float = Field(description="Self-reported arousal (0 to 1)")
    label_stress:  bool  = Field(default=False, description="Self-reported: stressed?")

    # Model predictions (optional — fill in whatever was available at check-in time)
    voice_emotion: Optional[str]   = None
    voice_valence: Optional[float] = None
    voice_arousal: Optional[float] = None
    health_valence: Optional[float] = None
    health_arousal: Optional[float] = None
    health_stress:  Optional[bool]  = None
    combined_emotion: Optional[str]   = None
    combined_valence: Optional[float] = None
    combined_arousal: Optional[float] = None
    combined_stress:  Optional[bool]  = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/log")
async def log_ground_truth_sample(sample: GroundTruthSample):
    """Append one labeled sample to the ground-truth corpus.

    Call this after each voice check-in, alongside the model's predictions.
    After 30+ samples you'll have a statistically meaningful benchmark.
    After 200+ samples the benchmark becomes publication-quality.
    """
    if sample.label_emotion not in ("happy", "sad", "angry", "fear", "neutral", "calm"):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail=f"label_emotion must be one of: happy/sad/angry/fear/neutral/calm. Got: {sample.label_emotion!r}",
        )

    record = sample.model_dump(exclude_none=False)
    if record["timestamp"] is None:
        record["timestamp"] = time.time()

    # Append to JSONL
    _GT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_GT_FILE, "a") as f:
        f.write(json.dumps({k: v for k, v in record.items() if v is not None}) + "\n")

    # Count
    n_samples = sum(1 for _ in open(_GT_FILE))

    return {
        "status": "logged",
        "n_samples_total": n_samples,
        "benchmark_ready": n_samples >= 30,
        "publication_ready": n_samples >= 200,
        "progress": {
            "samples": n_samples,
            "target_min": 30,
            "target_publication": 200,
            "pct_to_min": round(min(1.0, n_samples / 30) * 100, 1),
        },
    }


@router.get("/run")
async def run_benchmark():
    """Run the no-EEG modality ablation benchmark.

    Evaluates all accumulated ground-truth samples across:
    - Voice only
    - Health only
    - Voice + Health
    - Voice + Health + Supplements (if combined_supplement_* fields present)

    Metrics: 6-class accuracy, weighted F1, valence MAE, arousal MAE, stress F1,
    mood direction accuracy.

    Returns the full results table and saves to benchmarks/no_eeg_benchmark_results.json.
    """
    try:
        from benchmarks.no_eeg_benchmark import run_benchmark as _run, _DEFAULT_GT_FILE, _DEFAULT_RESULTS_FILE
        results = _run(
            gt_file=_GT_FILE,
            save_results=True,
            results_file=_RESULTS_FILE,
        )
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}

    if "error" in results:
        return {
            "status": "no_data",
            "message": "No labeled samples collected yet. Call POST /no-eeg-benchmark/log first.",
            "n_samples": 0,
        }

    return {"status": "ok", "results": results}


@router.get("/status")
async def get_benchmark_status():
    """Return sample count and collection progress."""
    n_samples = 0
    if _GT_FILE.exists():
        n_samples = sum(1 for line in open(_GT_FILE) if line.strip())

    return {
        "n_samples": n_samples,
        "gt_file": str(_GT_FILE),
        "benchmark_ready": n_samples >= 30,
        "publication_ready": n_samples >= 200,
        "progress": {
            "samples": n_samples,
            "target_min": 30,
            "target_publication": 200,
            "pct_to_min": round(min(1.0, n_samples / 30) * 100, 1),
            "pct_to_publication": round(min(1.0, n_samples / 200) * 100, 1),
        },
        "collection_protocol": (
            "Run POST /no-eeg-benchmark/log after each voice check-in (3x/day). "
            "Target: 30 samples minimum (10 days), 200 for publication."
        ),
    }


@router.get("/results")
async def get_saved_results():
    """Return the most recently saved benchmark results JSON."""
    if not _RESULTS_FILE.exists():
        return {
            "status": "no_results",
            "message": "No benchmark results yet. Call GET /no-eeg-benchmark/run after collecting samples.",
        }
    with open(_RESULTS_FILE) as f:
        data = json.load(f)
    return {"status": "ok", "results": data}
