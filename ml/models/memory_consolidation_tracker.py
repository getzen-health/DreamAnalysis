"""Sleep Memory Consolidation Tracker via Spindle-SO Coupling.

Tracks memory consolidation quality during sleep by measuring:
1. Sleep spindle density (11-16 Hz bursts per minute) — from existing detector
2. Slow oscillation (SO) detection (0.5-1.5 Hz, 1-4 Hz for Muse 2's limited bandwidth)
3. SO-spindle coupling (phase-amplitude coupling proxy)
4. Optional: TMR (targeted memory reactivation) up-state detection

Scientific basis:
- npj Science of Learning (2025): personalized TMR with SO-spindle sync enhances memory
- bioRxiv (2025): forehead EEG (AF7/AF8) can reliably detect SO phase for closed-loop TMR
- 15-25% increase in slow-wave activity with phase-locked cueing
- Spindle density in N2/N3 predicts next-day memory performance (r=0.4-0.6)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from typing import Dict, List, Optional


def detect_slow_oscillations(
    signal: np.ndarray, fs: float = 256.0
) -> Dict:
    """Detect slow oscillation (SO) events in EEG signal.

    For Muse 2 with limited delta bandwidth, targets 0.5-2.0 Hz.
    Returns SO events, up-state timestamps, and power.
    """
    # Bandpass 0.5–2.0 Hz for slow oscillations
    nyq = fs / 2
    low, high = 0.5 / nyq, min(2.0 / nyq, 0.99)
    b, a = butter(2, [low, high], btype="band")
    so_signal = filtfilt(b, a, signal)

    # SO power
    so_power = float(np.mean(so_signal ** 2))

    # Detect up-states: positive zero-crossings of the filtered signal
    # Up-state = transition from negative to positive half-cycle
    zero_crossings = np.where(np.diff(np.sign(so_signal)) > 0)[0]

    # Filter: amplitude must exceed threshold for valid SO
    threshold = 30.0  # µV, relaxed for prefrontal sites (lower amplitude)
    valid_upcross = []
    for zc in zero_crossings:
        window_start = max(0, zc - int(0.5 * fs))
        window_end = min(len(so_signal), zc + int(0.5 * fs))
        if np.ptp(so_signal[window_start:window_end]) > threshold:
            valid_upcross.append(zc)

    so_rate = len(valid_upcross) / (len(signal) / fs / 60.0 + 1e-6)  # per minute

    return {
        "so_power": round(so_power, 6),
        "so_count": len(valid_upcross),
        "so_rate_per_min": round(so_rate, 2),
        "up_state_indices": valid_upcross[:50],  # cap for API response
        "so_signal": so_signal,  # for coupling computation
    }


def compute_spindle_so_coupling(
    signal: np.ndarray,
    fs: float = 256.0,
    so_result: Optional[Dict] = None,
) -> float:
    """Compute SO-spindle coupling strength (phase-amplitude coupling proxy).

    Returns coupling score 0-1. Higher = stronger spindle activity
    locked to SO up-states.
    """
    if so_result is None:
        so_result = detect_slow_oscillations(signal, fs)

    up_states = so_result.get("up_state_indices", [])
    if len(up_states) < 3:
        return 0.0

    # Spindle band power (11-16 Hz)
    nyq = fs / 2
    b, a = butter(4, [11 / nyq, 16 / nyq], btype="band")
    spindle_signal = filtfilt(b, a, signal)
    spindle_env = np.abs(hilbert(spindle_signal))

    # Sample spindle envelope at SO up-states and nearby times
    window = int(0.5 * fs)  # 500ms window around up-state
    up_envelope = []
    for idx in up_states[:30]:
        start = max(0, idx - window // 2)
        end = min(len(spindle_env), idx + window // 2)
        up_envelope.append(float(np.mean(spindle_env[start:end])))

    # Compare to baseline envelope
    baseline_env = float(np.mean(spindle_env))
    if baseline_env < 1e-10:
        return 0.0

    mean_up_env = float(np.mean(up_envelope))
    coupling = float(np.clip((mean_up_env - baseline_env) / (baseline_env + 1e-10), 0, 1))
    return coupling


class MemoryConsolidationTracker:
    """Track memory consolidation quality during sleep via spindle-SO coupling.

    Score per epoch (30 seconds) and aggregate across a sleep session.
    """

    def __init__(self):
        self._epoch_scores: List[Dict] = []

    def score_epoch(
        self,
        signals: np.ndarray,
        fs: float = 256.0,
        sleep_stage: str = "N2",
    ) -> Dict:
        """Score a single sleep epoch for memory consolidation quality.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) array
            fs: sampling rate
            sleep_stage: current sleep stage (N2/N3 most relevant)

        Returns:
            dict with consolidation_quality (0-1), spindle_density,
            coupling_strength, so_rate, and sleep_stage.
        """
        from processing.eeg_processor import detect_sleep_spindles, preprocess

        # Use frontal channel (AF7 or ch1 for Muse 2)
        if signals.ndim == 2 and signals.shape[0] >= 2:
            frontal = signals[1]  # AF7
        elif signals.ndim == 2:
            frontal = signals[0]
        else:
            frontal = signals

        processed = preprocess(frontal, fs)
        epoch_duration_min = len(processed) / fs / 60.0

        # Spindle detection (existing function)
        try:
            spindle_result = detect_sleep_spindles(processed, fs)
            spindle_count = spindle_result.get("spindle_count", 0)
        except Exception:
            spindle_count = 0

        spindle_density = spindle_count / (epoch_duration_min + 1e-6)  # per minute

        # Slow oscillation detection
        so_result = detect_slow_oscillations(processed, fs)

        # SO-spindle coupling
        coupling = compute_spindle_so_coupling(processed, fs, so_result)

        # Sleep stage weight: N3 > N2 >> N1 > REM > Wake
        stage_upper = sleep_stage.upper()
        stage_weight = {
            "N3": 1.0, "N2": 0.8, "N1": 0.3, "REM": 0.4, "WAKE": 0.1
        }.get(stage_upper, 0.5)

        # Consolidation quality score
        # Normalize spindle density: >3/min is excellent for N2/N3
        spindle_score = float(np.clip(spindle_density / 4.0, 0, 1))
        coupling_score = float(np.clip(coupling * 2, 0, 1))
        so_score = float(np.clip(so_result["so_rate_per_min"] / 3.0, 0, 1))

        quality = stage_weight * (
            0.40 * spindle_score
            + 0.35 * coupling_score
            + 0.25 * so_score
        )
        quality = float(np.clip(quality, 0.0, 1.0))

        epoch_score = {
            "consolidation_quality": round(quality, 4),
            "spindle_density_per_min": round(spindle_density, 2),
            "spindle_count": int(spindle_count),
            "coupling_strength": round(coupling, 3),
            "so_rate_per_min": round(so_result["so_rate_per_min"], 2),
            "so_count": int(so_result["so_count"]),
            "sleep_stage": sleep_stage,
            "stage_weight": round(stage_weight, 2),
            "model_type": "spindle_so_coupling",
        }
        self._epoch_scores.append(epoch_score)
        return epoch_score

    def score_session(self, epoch_scores: Optional[List[Dict]] = None) -> Dict:
        """Aggregate memory consolidation across a full sleep session.

        Uses weighted average (N3 > N2 >> other stages).
        """
        scores = epoch_scores or self._epoch_scores
        if not scores:
            return {
                "session_quality": 0.0,
                "mean_spindle_density": 0.0,
                "mean_coupling": 0.0,
                "n_epochs": 0,
                "message": "No epochs scored yet",
            }

        weights = [s.get("stage_weight", 0.5) for s in scores]
        qualities = [s.get("consolidation_quality", 0.0) for s in scores]
        spindle_densities = [s.get("spindle_density_per_min", 0.0) for s in scores]
        couplings = [s.get("coupling_strength", 0.0) for s in scores]

        total_weight = sum(weights) + 1e-10
        session_quality = sum(w * q for w, q in zip(weights, qualities)) / total_weight

        if session_quality >= 0.65:
            quality_label = "excellent"
            message = "Strong memory consolidation activity detected."
        elif session_quality >= 0.45:
            quality_label = "good"
            message = "Moderate memory consolidation activity."
        elif session_quality >= 0.25:
            quality_label = "fair"
            message = "Below-average memory consolidation. More slow-wave sleep may help."
        else:
            quality_label = "poor"
            message = "Low memory consolidation activity. Consider sleep hygiene improvements."

        return {
            "session_quality": round(float(session_quality), 4),
            "quality_label": quality_label,
            "mean_spindle_density": round(float(np.mean(spindle_densities)), 2),
            "mean_coupling": round(float(np.mean(couplings)), 3),
            "n_epochs": len(scores),
            "n2_n3_epochs": sum(1 for s in scores if s.get("sleep_stage") in ("N2", "N3")),
            "message": message,
        }

    def get_tmr_trigger(
        self, signals: np.ndarray, fs: float = 256.0, sleep_stage: str = "N2"
    ) -> Dict:
        """Check if current SO phase is appropriate for TMR audio cue.

        Returns True if we are near an SO up-state (good time to play sound).
        For real-time TMR implementation.
        """
        if signals.ndim == 2 and signals.shape[0] >= 2:
            frontal = signals[1]
        elif signals.ndim == 2:
            frontal = signals[0]
        else:
            frontal = signals

        from processing.eeg_processor import preprocess
        processed = preprocess(frontal, fs)
        so = detect_slow_oscillations(processed, fs)

        up_states = so.get("up_state_indices", [])
        last_sample = len(processed) - 1

        # Check if we just crossed an up-state (within last 200ms)
        recent_threshold = int(0.2 * fs)
        in_upstate = any(abs(last_sample - u) < recent_threshold for u in up_states)

        return {
            "trigger_cue": bool(in_upstate and sleep_stage in ("N2", "N3")),
            "sleep_stage": sleep_stage,
            "so_count": so["so_count"],
            "near_upstate": in_upstate,
        }


_tracker_instances: dict = {}


def get_memory_tracker(user_id: str = "default") -> MemoryConsolidationTracker:
    if user_id not in _tracker_instances:
        _tracker_instances[user_id] = MemoryConsolidationTracker()
    return _tracker_instances[user_id]
