"""Brain connectivity and anomaly baseline endpoints."""

import numpy as np
from fastapi import APIRouter, HTTPException

from ._shared import (
    _anomaly_detector,
    EEGInput, AnomalyBaselineRequest,
    compute_granger_causality, compute_dtf, compute_graph_metrics,
)

router = APIRouter()


@router.post("/analyze-connectivity")
async def analyze_connectivity(data: EEGInput):
    """Compute brain network connectivity and graph metrics."""
    try:
        signals = np.array(data.signals)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        fs = data.fs
        n_channels = signals.shape[0]

        if n_channels < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 channels for connectivity analysis")

        corr = np.corrcoef(signals)
        np.fill_diagonal(corr, 0)
        connectivity_matrix = np.abs(corr)

        graph_metrics = compute_graph_metrics(connectivity_matrix)
        gc = compute_granger_causality(signals, fs)
        dtf = compute_dtf(signals, fs)

        return {
            "connectivity_matrix": connectivity_matrix.tolist(),
            "graph_metrics": graph_metrics,
            "directed_flow": {
                "granger": gc,
                "dtf_matrix": dtf["dtf_matrix"],
                "dominant_direction": dtf["dominant_direction"],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/anomaly/set-baseline")
async def set_anomaly_baseline(request: AnomalyBaselineRequest):
    """Train anomaly detector on user's normal EEG features."""
    return _anomaly_detector.fit_baseline(request.features_list)
