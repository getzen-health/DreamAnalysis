"""Two-brain hyperscanning for social cognition with consumer EEG (#130).

Computes inter-brain synchrony (IBS) metrics between two simultaneous EEG
streams. Based on Koike et al. (2016) and Dumas et al. (2010) hyperscanning
paradigms adapted for Muse 2 consumer headsets.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/hyperscanning", tags=["hyperscanning"])

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class HyperscanInput(BaseModel):
    brain_a_signals: List[List[float]]   # (n_ch, n_samples) — person A
    brain_b_signals: List[List[float]]   # (n_ch, n_samples) — person B
    fs: float = 256.0
    pair_id: str = "pair_default"        # identifies the dyad


class HyperscanResult(BaseModel):
    pair_id: str
    alpha_ibs: float                     # inter-brain synchrony in alpha band (0-1)
    theta_ibs: float
    beta_ibs: float
    overall_synchrony: float             # weighted average
    social_engagement_index: float       # 0-1
    coordination_label: str             # low / moderate / high / very_high
    n_channel_pairs: int
    processed_at: float


class PairStats(BaseModel):
    pair_id: str
    n_epochs: int
    mean_overall_synchrony: float
    peak_synchrony: float
    coordination_distribution: dict


# ---------------------------------------------------------------------------
# In-memory
# ---------------------------------------------------------------------------

_pair_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=300))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _band_power(signal: np.ndarray, fs: float, flo: float, fhi: float) -> float:
    from scipy.signal import welch
    nperseg = min(len(signal), int(fs * 2))
    f, psd = welch(signal, fs=fs, nperseg=nperseg)
    idx = np.logical_and(f >= flo, f <= fhi)
    return float(np.mean(psd[idx])) if idx.any() else 1e-9


def _plv(sig_a: np.ndarray, sig_b: np.ndarray, fs: float,
         flo: float, fhi: float) -> float:
    """Phase-locking value between two signals in a frequency band."""
    from scipy.signal import butter, filtfilt, hilbert
    b, a = butter(4, [flo, fhi], btype="band", fs=fs)
    fa = filtfilt(b, a, sig_a)
    fb = filtfilt(b, a, sig_b)
    phase_diff = np.angle(hilbert(fa)) - np.angle(hilbert(fb))
    return float(abs(np.mean(np.exp(1j * phase_diff))))


def _compute_ibs(brain_a: np.ndarray, brain_b: np.ndarray, fs: float) -> dict:
    n_a = brain_a.shape[0]
    n_b = brain_b.shape[0]
    n_ch = min(n_a, n_b, 4)

    alpha_plvs, theta_plvs, beta_plvs = [], [], []
    for ch in range(n_ch):
        try:
            alpha_plvs.append(_plv(brain_a[ch], brain_b[ch], fs, 8, 12))
            theta_plvs.append(_plv(brain_a[ch], brain_b[ch], fs, 4, 8))
            beta_plvs.append(_plv(brain_a[ch], brain_b[ch], fs, 12, 30))
        except Exception:
            pass

    alpha_ibs = float(np.mean(alpha_plvs)) if alpha_plvs else 0.0
    theta_ibs = float(np.mean(theta_plvs)) if theta_plvs else 0.0
    beta_ibs  = float(np.mean(beta_plvs))  if beta_plvs  else 0.0

    # Weighted: alpha most relevant for social alignment (Dumas 2010)
    overall = float(np.clip(0.5 * alpha_ibs + 0.3 * theta_ibs + 0.2 * beta_ibs, 0, 1))

    # Social engagement index combines alpha IBS with individual engagement
    sei = float(np.clip(0.7 * overall + 0.3 * alpha_ibs, 0, 1))

    if overall < 0.25:
        label = "low"
    elif overall < 0.45:
        label = "moderate"
    elif overall < 0.65:
        label = "high"
    else:
        label = "very_high"

    return {
        "alpha_ibs": alpha_ibs,
        "theta_ibs": theta_ibs,
        "beta_ibs": beta_ibs,
        "overall_synchrony": overall,
        "social_engagement_index": sei,
        "coordination_label": label,
        "n_channel_pairs": n_ch,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/sync", response_model=HyperscanResult)
async def compute_hyperscanning_sync(req: HyperscanInput):
    """Compute inter-brain synchrony between two simultaneous EEG streams."""
    brain_a = np.array(req.brain_a_signals, dtype=float)
    brain_b = np.array(req.brain_b_signals, dtype=float)
    if brain_a.ndim == 1:
        brain_a = brain_a[np.newaxis, :]
    if brain_b.ndim == 1:
        brain_b = brain_b[np.newaxis, :]

    min_len = min(brain_a.shape[1], brain_b.shape[1])
    if min_len < int(req.fs * 0.5):
        raise HTTPException(400, "Need at least 0.5 s of data per brain")

    brain_a = brain_a[:, :min_len]
    brain_b = brain_b[:, :min_len]

    metrics = _compute_ibs(brain_a, brain_b, req.fs)
    result = HyperscanResult(
        pair_id=req.pair_id,
        processed_at=time.time(),
        **metrics,
    )
    _pair_history[req.pair_id].append(result.dict())
    return result


@router.get("/stats/{pair_id}", response_model=PairStats)
async def get_hyperscanning_stats(pair_id: str):
    """Return aggregate inter-brain synchrony statistics for a dyad."""
    h = list(_pair_history[pair_id])
    if not h:
        return PairStats(
            pair_id=pair_id, n_epochs=0,
            mean_overall_synchrony=0.0, peak_synchrony=0.0,
            coordination_distribution={},
        )
    syncs = [r["overall_synchrony"] for r in h]
    labels = [r["coordination_label"] for r in h]
    dist   = {lbl: labels.count(lbl) for lbl in set(labels)}
    return PairStats(
        pair_id=pair_id,
        n_epochs=len(h),
        mean_overall_synchrony=float(np.mean(syncs)),
        peak_synchrony=float(np.max(syncs)),
        coordination_distribution=dist,
    )


@router.post("/reset/{pair_id}")
async def reset_hyperscanning(pair_id: str):
    """Clear stored hyperscanning history for a dyad."""
    _pair_history[pair_id].clear()
    return {"pair_id": pair_id, "status": "reset"}
