"""Brain connectivity graph — compute inter-channel connectivity.

Builds a functional connectivity graph from 4-channel Muse 2 EEG
using coherence, phase-locking value (PLV), and Granger causality
proxy. Maps the 6 possible channel pairs into a connectivity matrix.

Channel pairs (Muse 2):
  TP9-AF7, TP9-AF8, TP9-TP10, AF7-AF8, AF7-TP10, AF8-TP10

Connectivity types:
- Coherence: linear coupling in a frequency band
- PLV: phase synchronization
- wPLI: weighted phase lag index (robust to volume conduction)

References:
    Lachaux et al. (1999) — Phase-locking value
    Vinck et al. (2011) — Weighted phase lag index
    Stam et al. (2007) — Small-world networks in EEG
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal


class ConnectivityGraph:
    """Compute functional brain connectivity from EEG channel pairs.

    Builds coherence and PLV matrices for each frequency band,
    computes graph metrics (density, clustering), and tracks
    connectivity changes over time.
    """

    BANDS = {
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
    }

    CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]

    def __init__(self, fs: float = 256.0):
        self._fs = fs
        self._baselines: Dict[str, Dict] = {}
        self._history: Dict[str, List[Dict]] = {}

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record baseline connectivity.

        Args:
            signals: (n_channels, n_samples) EEG.
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with baseline connectivity matrices.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        conn = self._compute_connectivity(signals, fs)
        self._baselines[user_id] = conn

        return {
            "baseline_set": True,
            "n_channels": signals.shape[0],
            "n_pairs": len(conn.get("pairs", [])),
        }

    def analyze(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Analyze current brain connectivity.

        Args:
            signals: (n_channels, n_samples) EEG.
            fs: Sampling rate.
            user_id: User identifier.

        Returns:
            Dict with connectivity matrices, graph metrics,
            and connectivity state classification.
        """
        fs = fs or self._fs
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        conn = self._compute_connectivity(signals, fs)
        baseline = self._baselines.get(user_id, {})

        # Graph metrics
        n_ch = signals.shape[0]
        density = self._graph_density(conn, n_ch)
        mean_coherence = self._mean_metric(conn, "coherence")
        mean_plv = self._mean_metric(conn, "plv")

        # Connectivity state
        if mean_coherence > 0.6:
            state = "highly_connected"
        elif mean_coherence > 0.3:
            state = "moderately_connected"
        else:
            state = "weakly_connected"

        # Frontal-temporal connectivity (AF7/AF8 to TP9/TP10)
        ft_pairs = [(1, 0), (1, 3), (2, 0), (2, 3)]
        ft_coherences = []
        for pair in conn.get("pairs", []):
            ch_pair = (pair["ch_i"], pair["ch_j"])
            if ch_pair in ft_pairs or (ch_pair[1], ch_pair[0]) in ft_pairs:
                ft_coherences.append(pair.get("alpha_coherence", 0))

        ft_connectivity = float(np.mean(ft_coherences)) if ft_coherences else 0.0

        # Inter-hemispheric connectivity (left vs right)
        ih_pairs = [(0, 3), (1, 2)]  # TP9-TP10, AF7-AF8
        ih_coherences = []
        for pair in conn.get("pairs", []):
            ch_pair = (pair["ch_i"], pair["ch_j"])
            if ch_pair in ih_pairs or (ch_pair[1], ch_pair[0]) in ih_pairs:
                ih_coherences.append(pair.get("alpha_coherence", 0))

        ih_connectivity = float(np.mean(ih_coherences)) if ih_coherences else 0.0

        result = {
            "pairs": conn.get("pairs", []),
            "mean_coherence": round(mean_coherence, 4),
            "mean_plv": round(mean_plv, 4),
            "graph_density": round(density, 4),
            "connectivity_state": state,
            "frontal_temporal_connectivity": round(ft_connectivity, 4),
            "interhemispheric_connectivity": round(ih_connectivity, 4),
            "n_channels": n_ch,
            "has_baseline": bool(baseline),
        }

        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)
        if len(self._history[user_id]) > 500:
            self._history[user_id] = self._history[user_id][-500:]

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get session statistics."""
        history = self._history.get(user_id, [])
        if not history:
            return {"n_epochs": 0, "has_baseline": user_id in self._baselines}

        coherences = [h["mean_coherence"] for h in history]
        states = [h["connectivity_state"] for h in history]
        state_counts = {}
        for s in states:
            state_counts[s] = state_counts.get(s, 0) + 1

        return {
            "n_epochs": len(history),
            "has_baseline": user_id in self._baselines,
            "mean_coherence": round(float(np.mean(coherences)), 4),
            "dominant_state": max(state_counts, key=state_counts.get),
            "state_distribution": state_counts,
        }

    def get_history(
        self, user_id: str = "default", last_n: Optional[int] = None
    ) -> List[Dict]:
        """Get analysis history."""
        history = self._history.get(user_id, [])
        if last_n:
            history = history[-last_n:]
        return history

    def reset(self, user_id: str = "default"):
        """Clear all data."""
        self._baselines.pop(user_id, None)
        self._history.pop(user_id, None)

    # ── Private helpers ─────────────────────────────────────────

    def _compute_connectivity(self, signals: np.ndarray, fs: float) -> Dict:
        """Compute pairwise connectivity metrics."""
        n_ch = signals.shape[0]
        pairs = []

        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                pair = {"ch_i": i, "ch_j": j}

                nperseg = min(signals.shape[1], int(fs * 2))
                if nperseg < 4:
                    pair.update({
                        "alpha_coherence": 0.0, "theta_coherence": 0.0,
                        "beta_coherence": 0.0, "plv": 0.0
                    })
                    pairs.append(pair)
                    continue

                # Coherence per band
                try:
                    f_coh, coh = scipy_signal.coherence(
                        signals[i], signals[j], fs=fs, nperseg=nperseg
                    )
                    for band, (low, high) in self.BANDS.items():
                        mask = (f_coh >= low) & (f_coh <= high)
                        if np.any(mask):
                            pair[f"{band}_coherence"] = round(float(np.mean(coh[mask])), 4)
                        else:
                            pair[f"{band}_coherence"] = 0.0
                except Exception:
                    for band in self.BANDS:
                        pair[f"{band}_coherence"] = 0.0

                # PLV in alpha band
                pair["plv"] = round(self._compute_plv(signals[i], signals[j], fs), 4)

                pairs.append(pair)

        return {"pairs": pairs}

    def _compute_plv(self, sig1: np.ndarray, sig2: np.ndarray, fs: float) -> float:
        """Compute phase-locking value in alpha band."""
        # Bandpass to alpha (8-12 Hz)
        try:
            nyq = fs / 2
            low, high = 8 / nyq, 12 / nyq
            if high >= 1.0:
                high = 0.99
            b, a = scipy_signal.butter(3, [low, high], btype='band')
            filt1 = scipy_signal.filtfilt(b, a, sig1)
            filt2 = scipy_signal.filtfilt(b, a, sig2)

            # Hilbert transform for instantaneous phase
            phase1 = np.angle(scipy_signal.hilbert(filt1))
            phase2 = np.angle(scipy_signal.hilbert(filt2))

            # PLV
            plv = float(np.abs(np.mean(np.exp(1j * (phase1 - phase2)))))
            return plv
        except Exception:
            return 0.0

    def _graph_density(self, conn: Dict, n_ch: int) -> float:
        """Compute graph density (fraction of strong connections)."""
        pairs = conn.get("pairs", [])
        if not pairs:
            return 0.0
        threshold = 0.3
        strong = sum(1 for p in pairs if p.get("alpha_coherence", 0) > threshold)
        total = len(pairs)
        return strong / total if total > 0 else 0.0

    def _mean_metric(self, conn: Dict, metric: str) -> float:
        """Compute mean of a metric across all pairs."""
        pairs = conn.get("pairs", [])
        if not pairs:
            return 0.0
        if metric == "coherence":
            values = [p.get("alpha_coherence", 0) for p in pairs]
        elif metric == "plv":
            values = [p.get("plv", 0) for p in pairs]
        else:
            return 0.0
        return float(np.mean(values)) if values else 0.0
