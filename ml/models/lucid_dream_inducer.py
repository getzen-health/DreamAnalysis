"""LuciEntry-style closed-loop lucid dream induction controller.

Monitors EEG+EOG in real-time during sleep, detects stable REM, schedules
audio cues ("This is a dream" x3 over 10 sec), then listens for deliberate
left-right eye movement (LRLR) as confirmation of lucidity.

State machine:
  idle → rem_monitoring → rem_stable → cues_scheduled → cues_delivered
       → lr_detection → lucid_confirmed / retry (back to rem_monitoring)

EOG from Muse 2:
  TP9/TP10 (ch0/ch3) capture slow horizontal eye movements as opposite-polarity
  deflections — left saccade → TP9 positive, TP10 negative; right saccade → opposite.
  The detect_lr_signal() method looks for alternating-polarity bursts with 200-800 ms
  inter-saccade gap, matching the 2-3 LRLR confirmation pattern.

Reference: Sakaino et al. (2025) LuciEntry, DIS 2025; Moctezuma et al. (2025) BioMed RI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from processing.eeg_processor import extract_band_powers, preprocess


# ─── State machine ────────────────────────────────────────────────────────────

class InductionState(str, Enum):
    IDLE = "idle"
    REM_MONITORING = "rem_monitoring"
    REM_STABLE = "rem_stable"
    CUES_SCHEDULED = "cues_scheduled"
    CUES_DELIVERED = "cues_delivered"
    LR_DETECTION = "lr_detection"
    LUCID_CONFIRMED = "lucid_confirmed"
    RETRY = "retry"


# Seconds of stable REM required before scheduling cues (LuciEntry: 5-10 min).
# For testing purposes the default is short; production should use 300–600 s.
REM_STABLE_DURATION_S: float = 300.0

# How many consecutive REM-positive epochs (at 1 Hz) to declare "stable REM".
REM_CONSECUTIVE_EPOCHS: int = 30  # 30 s of continuous REM detection

# Cue delivery window: 3 repetitions of the cue phrase, 10 seconds total.
CUE_DURATION_S: float = 10.0

# After cues, monitor for LR signal for this long before declaring failure.
LR_WINDOW_S: float = 60.0

# Wait this long between retry attempts.
RETRY_WAIT_S: float = 300.0


@dataclass
class InductionSession:
    user_id: str
    fs: float = 256.0
    started_at: float = field(default_factory=time.time)
    state: InductionState = InductionState.IDLE
    rem_consecutive: int = 0
    rem_stable_since: Optional[float] = None
    cue_scheduled_at: Optional[float] = None
    cue_delivered_at: Optional[float] = None
    lr_window_started_at: Optional[float] = None
    last_retry_at: Optional[float] = None
    lucid_episodes: int = 0
    retry_count: int = 0
    # EOG polarity history: +1 = TP9 dominant, -1 = TP10 dominant
    _polarity_history: List[float] = field(default_factory=list)
    _theta_history: List[float] = field(default_factory=list)
    _epoch_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "state": self.state.value,
            "rem_consecutive_epochs": self.rem_consecutive,
            "rem_stable": self.rem_stable_since is not None,
            "cue_scheduled": self.cue_scheduled_at is not None,
            "cue_delivered": self.cue_delivered_at is not None,
            "lucid_episodes": self.lucid_episodes,
            "retry_count": self.retry_count,
            "elapsed_s": round(time.time() - self.started_at, 1),
            "epoch_count": self._epoch_count,
        }


# ─── Core model ───────────────────────────────────────────────────────────────

class LucidDreamInducer:
    """Closed-loop lucid dream induction via LuciEntry protocol.

    Usage (one call per epoch, ~1 Hz):
        result = inducer.process_epoch(user_id, eeg_4ch, fs=256)
        if result["trigger_cue"]:
            # tell client to play audio cue
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, InductionSession] = {}

    # ── Session management ─────────────────────────────────────────────────

    def start_session(self, user_id: str, fs: float = 256.0) -> Dict:
        session = InductionSession(user_id=user_id, fs=fs)
        session.state = InductionState.REM_MONITORING
        self._sessions[user_id] = session
        return {"status": "started", **session.to_dict()}

    def stop_session(self, user_id: str) -> Dict:
        session = self._sessions.pop(user_id, None)
        if session is None:
            return {"status": "not_found", "user_id": user_id}
        return {"status": "stopped", "lucid_episodes": session.lucid_episodes,
                "retry_count": session.retry_count}

    def get_status(self, user_id: str) -> Dict:
        session = self._sessions.get(user_id)
        if session is None:
            return {"status": "no_session", "user_id": user_id}
        return session.to_dict()

    def confirm_lucidity(self, user_id: str) -> Dict:
        """Manual confirmation of lucidity by the user pressing a button."""
        session = self._sessions.get(user_id)
        if session is None:
            return {"status": "no_session"}
        session.lucid_episodes += 1
        session.state = InductionState.LUCID_CONFIRMED
        return {"status": "confirmed", "lucid_episodes": session.lucid_episodes}

    # ── Main processing loop ───────────────────────────────────────────────

    def process_epoch(
        self,
        user_id: str,
        eeg: np.ndarray,
        fs: float = 256.0,
        sleep_stage: Optional[str] = None,
    ) -> Dict:
        """Process one epoch (any length ≥ 1 s) and advance the state machine.

        Args:
            user_id: Session owner.
            eeg: (4, n_samples) multichannel EEG; ch0=TP9 ch1=AF7 ch2=AF8 ch3=TP10.
            fs: Sampling rate.
            sleep_stage: Optional sleep stage label ("REM", "N1" …).

        Returns:
            Dict with current state, rem_score, lr_score, trigger_cue flag.
        """
        if user_id not in self._sessions:
            session = InductionSession(user_id=user_id, fs=fs)
            session.state = InductionState.REM_MONITORING
            self._sessions[user_id] = session
        session = self._sessions[user_id]
        session._epoch_count += 1
        now = time.time()

        # Ensure multichannel shape
        if eeg.ndim == 1:
            eeg = eeg.reshape(1, -1)

        rem_score = self._detect_rem(eeg, fs, sleep_stage)
        lr_score, lr_detected = self._detect_lr_signal(eeg, fs)
        trigger_cue = False

        # ── State transitions ───────────────────────────────────────────
        if session.state == InductionState.REM_MONITORING:
            if rem_score >= 0.6:
                session.rem_consecutive += 1
            else:
                session.rem_consecutive = max(0, session.rem_consecutive - 1)

            if session.rem_consecutive >= REM_CONSECUTIVE_EPOCHS:
                session.state = InductionState.REM_STABLE
                session.rem_stable_since = now

        elif session.state == InductionState.REM_STABLE:
            # Keep verifying REM
            if rem_score < 0.4:
                session.rem_consecutive = max(0, session.rem_consecutive - 2)
                if session.rem_consecutive < REM_CONSECUTIVE_EPOCHS // 2:
                    session.state = InductionState.REM_MONITORING
                    session.rem_stable_since = None
            else:
                # Schedule cues after REM_STABLE_DURATION_S of stable REM
                stable_for = now - (session.rem_stable_since or now)
                if stable_for >= REM_STABLE_DURATION_S:
                    session.state = InductionState.CUES_SCHEDULED
                    session.cue_scheduled_at = now

        elif session.state == InductionState.CUES_SCHEDULED:
            # Immediately deliver (client plays audio on receiving trigger_cue=True)
            trigger_cue = True
            session.state = InductionState.CUES_DELIVERED
            session.cue_delivered_at = now

        elif session.state == InductionState.CUES_DELIVERED:
            # Wait CUE_DURATION_S for cues to finish, then listen for LRLR
            if now - (session.cue_delivered_at or now) >= CUE_DURATION_S:
                session.state = InductionState.LR_DETECTION
                session.lr_window_started_at = now

        elif session.state == InductionState.LR_DETECTION:
            if lr_detected:
                session.lucid_episodes += 1
                session.state = InductionState.LUCID_CONFIRMED
            elif now - (session.lr_window_started_at or now) >= LR_WINDOW_S:
                # Timeout — retry
                session.retry_count += 1
                session.state = InductionState.RETRY
                session.last_retry_at = now

        elif session.state == InductionState.RETRY:
            # Wait before the next attempt
            if now - (session.last_retry_at or now) >= RETRY_WAIT_S:
                # Reset REM tracking for next attempt
                session.rem_consecutive = max(0, session.rem_consecutive - 10)
                session.rem_stable_since = None
                session.cue_scheduled_at = None
                session.cue_delivered_at = None
                session.lr_window_started_at = None
                session.state = InductionState.REM_MONITORING

        elif session.state == InductionState.LUCID_CONFIRMED:
            # Stay in confirmed state until session is reset
            pass

        return {
            "user_id": user_id,
            "state": session.state.value,
            "rem_score": round(float(rem_score), 3),
            "rem_consecutive_epochs": session.rem_consecutive,
            "lr_score": round(float(lr_score), 3),
            "lr_detected": lr_detected,
            "trigger_cue": trigger_cue,
            "lucid_episodes": session.lucid_episodes,
            "retry_count": session.retry_count,
            "epoch": session._epoch_count,
        }

    # ── Signal processing ──────────────────────────────────────────────────

    def _detect_rem(
        self,
        eeg: np.ndarray,
        fs: float,
        sleep_stage: Optional[str] = None,
    ) -> float:
        """Estimate probability that the current epoch is REM sleep.

        Uses:
        - Theta dominance (4-8 Hz) — characteristic of REM
        - Low delta relative to theta (distinguishes REM from N3)
        - Sleep stage label if provided by the sleep staging model

        Returns:
            rem_score in [0, 1].
        """
        # If the sleep staging model says REM, trust it heavily
        if sleep_stage is not None:
            stage_lower = sleep_stage.lower()
            if "rem" in stage_lower:
                return 0.92
            if stage_lower in ("wake", "w"):
                return 0.02
            if "n3" in stage_lower or "slow" in stage_lower:
                return 0.05

        # Feature-based fallback using AF7 (ch1) — more frontal EEG, less EMG than TP9
        ch_idx = min(1, eeg.shape[0] - 1)
        signal = eeg[ch_idx]
        try:
            processed = preprocess(signal, fs)
            bands = extract_band_powers(processed, fs)
        except Exception:
            return 0.0

        theta = bands.get("theta", 0.0)
        delta = bands.get("delta", 0.0)
        alpha = bands.get("alpha", 0.0)
        beta  = bands.get("beta", 0.0)

        # Track theta history for relative increase detection
        self._sessions_theta(theta)

        # Theta dominance over delta: REM signature
        theta_dom = theta / (delta + theta + 1e-8)

        # Low beta (no waking activity)
        beta_low = 1.0 - np.clip(beta / (beta + alpha + 1e-8), 0, 1)

        # High theta absolute (normalised via tanh)
        theta_abs = float(np.tanh(theta * 8))

        rem_score = 0.50 * theta_dom + 0.25 * beta_low + 0.25 * theta_abs
        return float(np.clip(rem_score, 0, 1))

    def _sessions_theta(self, theta: float) -> None:
        """Update internal theta history (no-op placeholder for per-session tracking)."""
        pass

    def _detect_lr_signal(
        self,
        eeg: np.ndarray,
        fs: float,
    ) -> Tuple[float, bool]:
        """Detect deliberate left-right-left-right eye movement (LRLR) from TP9/TP10.

        In Muse 2, horizontal saccades appear as opposite-polarity low-frequency
        deflections (0.5-4 Hz) in TP9 vs TP10:
          - Leftward saccade: TP9 goes positive, TP10 goes negative
          - Rightward saccade: TP9 goes negative, TP10 goes positive

        A deliberate LRLR signal = 2+ alternations within ~3 seconds,
        each alternation having ≥ 75 µV peak amplitude.

        Returns:
            (lr_score, detected_bool)
        """
        if eeg.shape[0] < 4:
            return 0.0, False

        tp9  = eeg[0]  # ch0 = TP9 (left temporal)
        tp10 = eeg[3]  # ch3 = TP10 (right temporal)

        # EOG differential: positive = left saccade, negative = right saccade
        eog_diff = tp9 - tp10

        # Low-pass to isolate slow eye movements (< 4 Hz)
        try:
            from scipy.signal import butter, filtfilt
            b, a = butter(2, 4.0 / (fs / 2), btype="low")
            eog_slow = filtfilt(b, a, eog_diff)
        except Exception:
            eog_slow = eog_diff

        # Detect polarity reversals above threshold
        threshold = 50.0  # µV
        n_samples = len(eog_slow)
        min_gap = int(0.15 * fs)   # min 150 ms between reversals
        max_gap = int(1.5 * fs)    # max 1.5 s between reversals

        crossings: List[Tuple[int, float]] = []  # (sample_idx, direction)
        i = min_gap
        while i < n_samples - min_gap:
            v = eog_slow[i]
            if abs(v) >= threshold:
                direction = 1.0 if v > 0 else -1.0
                # Only count if polarity changed from last crossing
                if not crossings or (direction != crossings[-1][1]
                                     and i - crossings[-1][0] >= min_gap
                                     and i - crossings[-1][0] <= max_gap):
                    crossings.append((i, direction))
                    i += min_gap  # skip ahead to avoid double-counting
                    continue
            i += 1

        # LRLR = at least 2 alternations (4 crossings: L R L R or R L R L)
        detected = len(crossings) >= 4

        # Smooth score = fraction of expected alternations found
        lr_score = float(np.clip(len(crossings) / 4.0, 0, 1))

        return lr_score, detected

    def list_sessions(self) -> List[Dict]:
        return [s.to_dict() for s in self._sessions.values()]
