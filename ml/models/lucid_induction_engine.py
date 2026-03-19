"""Lucid dreaming induction engine — issue #452.

EEG-based REM detection with targeted cue delivery for conscious dream entry.
Combines physiological monitoring with induction technique management:

  - REM detection: rapid eye movement artifacts, theta dominance, low EMG,
    desynchronized EEG patterns.
  - Induction techniques: MILD (Mnemonic Induction of Lucid Dreams),
    WBTB (Wake Back To Bed timing), external sensory cues.
  - Cue timing: wait for stable REM (>5 min), theta peak without K-complex,
    then deliver graduated cue.
  - Cue types: audio tone, haptic vibration, LED flash — configurable intensity.
  - Success tracking: per-technique, per-user hit rates.
  - Reality testing: daytime reminders that build dream-awareness habits.

Science basis:
  - LaBerge & Rheingold (1990) — MILD technique: prospective memory + intention
    setting during WBTB windows yields highest induction success (~50%).
  - Stumbrys et al. (2012) — meta-analysis: external cues during REM produce
    lucidity in ~20-55% of attempts depending on modality and timing.
  - Voss et al. (2014) — 40 Hz fronto-temporal stimulation during REM
    increases lucid awareness; gamma patterns distinguish lucid from non-lucid REM.
  - Baird et al. (2019) — reality testing frequency during waking correlates
    with spontaneous lucid dream rate (habit transfer to dream state).

No ML weights required — all computation is deterministic heuristic scoring.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# REM detection thresholds
THETA_DOMINANCE_THRESHOLD = 0.25    # theta relative power must exceed this
ALPHA_SUPPRESSION_THRESHOLD = 0.20  # alpha relative power must be below this
EMG_LOW_THRESHOLD = 15.0            # EMG amplitude (uV RMS) below this = low tone
DESYNCH_ENTROPY_THRESHOLD = 0.6     # spectral entropy above this = desynchronized

# Stable REM duration before cue delivery (seconds)
STABLE_REM_MIN_DURATION_S = 300.0   # 5 minutes

# Cue delivery constraints
MIN_THETA_PEAK_RATIO = 1.5         # theta must be 1.5x alpha for "theta peak"
K_COMPLEX_AMPLITUDE_UV = 75.0      # amplitude > this suggests K-complex presence

# Reality test schedule defaults
DEFAULT_REALITY_TESTS_PER_DAY = 10
REALITY_TEST_WINDOW_START_H = 8    # 8 AM
REALITY_TEST_WINDOW_END_H = 22     # 10 PM


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CueType(str, Enum):
    AUDIO = "audio"
    HAPTIC = "haptic"
    LED = "led"


class InductionTechnique(str, Enum):
    MILD = "mild"       # Mnemonic Induction of Lucid Dreams
    WBTB = "wbtb"       # Wake Back To Bed
    EXTERNAL_CUE = "external_cue"  # Audio/haptic/LED during REM


class REMState(str, Enum):
    NOT_REM = "not_rem"
    POSSIBLE_REM = "possible_rem"
    CONFIRMED_REM = "confirmed_rem"
    STABLE_REM = "stable_rem"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SleepEEGFeatures:
    """Extracted features from a sleep EEG epoch."""

    theta_power: float = 0.0       # relative power 0-1
    alpha_power: float = 0.0
    beta_power: float = 0.0
    delta_power: float = 0.0
    gamma_power: float = 0.0
    emg_amplitude: float = 0.0     # uV RMS from high-freq content
    spectral_entropy: float = 0.0  # 0-1
    eye_movement_score: float = 0.0  # 0-1, rapid eye movement artifact
    has_k_complex: bool = False
    has_sleep_spindle: bool = False
    epoch_duration_s: float = 1.0


@dataclass
class CueConfig:
    """Configuration for a lucid dream induction cue."""

    cue_type: CueType = CueType.AUDIO
    intensity: float = 0.3         # 0-1 scale
    duration_s: float = 3.0        # seconds
    pattern: str = "gentle_pulse"  # "gentle_pulse", "gradual_ramp", "steady"
    repeat_count: int = 3          # number of repetitions


@dataclass
class InductionAttempt:
    """Record of a single lucid dream induction attempt."""

    attempt_id: str = ""
    user_id: str = ""
    technique: InductionTechnique = InductionTechnique.EXTERNAL_CUE
    cue_config: Optional[CueConfig] = None
    timestamp: float = 0.0         # Unix timestamp
    rem_duration_s: float = 0.0    # how long user was in REM before cue
    cue_delivered: bool = False
    lucid_reported: bool = False    # user reported lucid dream after cue
    dream_recalled: bool = False    # user recalled any dream
    notes: str = ""


@dataclass
class LucidProfile:
    """Aggregated lucid dreaming profile for a user."""

    user_id: str = ""
    total_attempts: int = 0
    total_lucid: int = 0
    success_rate: float = 0.0      # total_lucid / total_attempts
    technique_rates: Dict[str, float] = field(default_factory=dict)
    technique_counts: Dict[str, int] = field(default_factory=dict)
    cue_type_rates: Dict[str, float] = field(default_factory=dict)
    cue_type_counts: Dict[str, int] = field(default_factory=dict)
    avg_rem_before_lucid_s: float = 0.0
    best_technique: str = ""
    best_cue_type: str = ""
    reality_tests_completed: int = 0
    reality_test_streak_days: int = 0
    last_updated: float = 0.0


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class LucidInductionEngine:
    """EEG-based lucid dream induction engine.

    Manages REM detection, cue timing/selection, success tracking, and
    reality test scheduling for one or more users.
    """

    def __init__(self) -> None:
        self._attempts: Dict[str, List[InductionAttempt]] = {}
        self._profiles: Dict[str, LucidProfile] = {}
        self._reality_test_schedules: Dict[str, List[float]] = {}
        self._rem_tracking: Dict[str, Dict[str, Any]] = {}
        self._attempt_counter: int = 0

    # -- REM detection -------------------------------------------------------

    def detect_rem_state(
        self,
        eeg_data: np.ndarray,
        fs: float = 256.0,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Detect REM state from EEG epoch.

        Parameters
        ----------
        eeg_data : ndarray
            EEG data, shape (n_channels, n_samples) or (n_samples,).
        fs : float
            Sampling rate in Hz.
        user_id : str
            User identifier for tracking REM duration.

        Returns
        -------
        dict with keys:
            rem_state : str — one of REMState values
            features : dict — extracted SleepEEGFeatures as dict
            rem_score : float — 0-1 composite REM likelihood
            rem_duration_s : float — cumulative stable REM duration
        """
        features = self._extract_sleep_features(eeg_data, fs)
        rem_score = self._compute_rem_score(features)
        rem_state = self._classify_rem_state(rem_score)

        # Track cumulative REM duration
        if user_id not in self._rem_tracking:
            self._rem_tracking[user_id] = {
                "rem_start": None,
                "cumulative_s": 0.0,
                "last_state": REMState.NOT_REM,
            }

        tracking = self._rem_tracking[user_id]
        now = time.time()

        if rem_state in (REMState.CONFIRMED_REM, REMState.STABLE_REM):
            if tracking["rem_start"] is None:
                tracking["rem_start"] = now
            elapsed = now - tracking["rem_start"]
            tracking["cumulative_s"] = elapsed
            if elapsed >= STABLE_REM_MIN_DURATION_S:
                rem_state = REMState.STABLE_REM
        else:
            tracking["rem_start"] = None
            tracking["cumulative_s"] = 0.0

        tracking["last_state"] = rem_state

        return {
            "rem_state": rem_state.value,
            "features": self._features_to_dict(features),
            "rem_score": float(rem_score),
            "rem_duration_s": float(tracking["cumulative_s"]),
        }

    def _extract_sleep_features(
        self, eeg_data: np.ndarray, fs: float
    ) -> SleepEEGFeatures:
        """Extract sleep-relevant features from an EEG epoch."""
        if eeg_data.ndim == 2:
            signal = eeg_data[0]  # use first channel
            multichannel = eeg_data
        else:
            signal = eeg_data
            multichannel = None

        n_samples = len(signal)
        epoch_duration = n_samples / fs

        # Compute band powers using Welch PSD
        band_powers = self._compute_band_powers(signal, fs)
        total = sum(band_powers.values()) or 1e-10

        theta_rel = band_powers.get("theta", 0.0) / total
        alpha_rel = band_powers.get("alpha", 0.0) / total
        beta_rel = band_powers.get("beta", 0.0) / total
        delta_rel = band_powers.get("delta", 0.0) / total
        gamma_rel = band_powers.get("gamma", 0.0) / total

        # EMG amplitude estimate: RMS of high-frequency content (>30 Hz)
        emg_amp = self._estimate_emg(signal, fs)

        # Spectral entropy
        entropy = self._spectral_entropy(band_powers, total)

        # Eye movement score from temporal channels
        eye_score = 0.0
        if multichannel is not None and multichannel.shape[0] >= 4:
            eye_score = self._eye_movement_score(multichannel, fs)

        # K-complex detection: large slow wave
        has_k = bool(np.max(np.abs(signal)) > K_COMPLEX_AMPLITUDE_UV)

        # Sleep spindle detection: 11-16 Hz burst
        has_spindle = self._detect_spindle(signal, fs)

        return SleepEEGFeatures(
            theta_power=theta_rel,
            alpha_power=alpha_rel,
            beta_power=beta_rel,
            delta_power=delta_rel,
            gamma_power=gamma_rel,
            emg_amplitude=emg_amp,
            spectral_entropy=entropy,
            eye_movement_score=eye_score,
            has_k_complex=has_k,
            has_sleep_spindle=has_spindle,
            epoch_duration_s=epoch_duration,
        )

    def _compute_band_powers(
        self, signal: np.ndarray, fs: float
    ) -> Dict[str, float]:
        """Compute band powers via simple periodogram."""
        n = len(signal)
        if n < 4:
            return {"delta": 0.0, "theta": 0.0, "alpha": 0.0,
                    "beta": 0.0, "gamma": 0.0}

        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        fft_vals = np.abs(np.fft.rfft(signal)) ** 2 / n

        bands = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 12.0),
            "beta": (12.0, 30.0),
            "gamma": (30.0, 50.0),
        }

        powers: Dict[str, float] = {}
        for name, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            powers[name] = float(np.sum(fft_vals[mask])) if np.any(mask) else 0.0

        return powers

    def _estimate_emg(self, signal: np.ndarray, fs: float) -> float:
        """Estimate EMG amplitude from high-frequency power (>30 Hz)."""
        n = len(signal)
        if n < 4:
            return 0.0

        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        fft_vals = np.abs(np.fft.rfft(signal))
        hf_mask = freqs >= 30.0
        if not np.any(hf_mask):
            return 0.0
        hf_power = np.mean(fft_vals[hf_mask] ** 2)
        return float(np.sqrt(hf_power))

    def _spectral_entropy(
        self, band_powers: Dict[str, float], total: float
    ) -> float:
        """Compute normalized spectral entropy across bands."""
        if total <= 0:
            return 0.0

        probs = []
        for p in band_powers.values():
            if p > 0:
                probs.append(p / total)

        if not probs:
            return 0.0

        entropy = -sum(p * math.log2(p) for p in probs)
        max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def _eye_movement_score(
        self, multichannel: np.ndarray, fs: float
    ) -> float:
        """Score rapid eye movements from TP9/TP10 differential.

        Muse 2: ch0=TP9, ch3=TP10. Horizontal eye movements produce
        opposite-polarity deflections in these channels.
        """
        diff = multichannel[0] - multichannel[3]  # TP9 - TP10
        # Count zero crossings as proxy for saccade rate
        signs = np.sign(diff)
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        n_samples = len(diff)
        duration = n_samples / fs

        # Normalize: REM typically has 20-60 saccades per minute
        saccade_rate = (crossings / 2.0) / max(duration, 0.1) * 60.0
        score = min(1.0, max(0.0, saccade_rate / 60.0))
        return float(score)

    def _detect_spindle(self, signal: np.ndarray, fs: float) -> bool:
        """Detect sleep spindles (11-16 Hz bursts) in the signal."""
        n = len(signal)
        if n < int(fs * 0.5):
            return False

        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        fft_vals = np.abs(np.fft.rfft(signal)) ** 2 / n

        spindle_mask = (freqs >= 11.0) & (freqs <= 16.0)
        total_mask = (freqs >= 0.5) & (freqs <= 50.0)

        if not np.any(spindle_mask) or not np.any(total_mask):
            return False

        spindle_power = float(np.sum(fft_vals[spindle_mask]))
        total_power = float(np.sum(fft_vals[total_mask]))

        if total_power <= 0:
            return False

        return (spindle_power / total_power) > 0.15

    def _compute_rem_score(self, features: SleepEEGFeatures) -> float:
        """Compute composite REM likelihood score (0-1).

        REM indicators:
          - High theta relative power (theta dominance)
          - Low alpha (alpha suppression)
          - Low EMG (muscle atonia)
          - High spectral entropy (desynchronized EEG)
          - Rapid eye movements present
          - No K-complexes or sleep spindles (those indicate N2)
        """
        scores = []

        # Theta dominance: higher theta = more REM-like
        theta_score = min(1.0, features.theta_power / 0.4)
        scores.append(("theta", theta_score, 0.25))

        # Alpha suppression: lower alpha = more REM-like
        alpha_score = max(0.0, 1.0 - features.alpha_power / 0.3)
        scores.append(("alpha_suppress", alpha_score, 0.15))

        # Low EMG: muscle atonia during REM
        emg_score = max(0.0, 1.0 - features.emg_amplitude / 30.0)
        scores.append(("emg_low", emg_score, 0.20))

        # Spectral entropy: desynchronized = high entropy
        entropy_score = min(1.0, features.spectral_entropy / 0.8)
        scores.append(("entropy", entropy_score, 0.15))

        # Eye movements
        scores.append(("eye_movement", features.eye_movement_score, 0.20))

        # N2 markers penalty (K-complex or spindle present)
        n2_penalty = 0.0
        if features.has_k_complex:
            n2_penalty += 0.5
        if features.has_sleep_spindle:
            n2_penalty += 0.5
        n2_score = 1.0 - n2_penalty
        scores.append(("no_n2", n2_score, 0.05))

        # Weighted sum
        total = sum(score * weight for _, score, weight in scores)
        return float(np.clip(total, 0.0, 1.0))

    def _classify_rem_state(self, rem_score: float) -> REMState:
        """Classify REM state from composite score."""
        if rem_score >= 0.7:
            return REMState.CONFIRMED_REM
        elif rem_score >= 0.4:
            return REMState.POSSIBLE_REM
        else:
            return REMState.NOT_REM

    @staticmethod
    def _features_to_dict(features: SleepEEGFeatures) -> Dict[str, Any]:
        """Convert SleepEEGFeatures to a JSON-safe dictionary."""
        return {
            "theta_power": features.theta_power,
            "alpha_power": features.alpha_power,
            "beta_power": features.beta_power,
            "delta_power": features.delta_power,
            "gamma_power": features.gamma_power,
            "emg_amplitude": features.emg_amplitude,
            "spectral_entropy": features.spectral_entropy,
            "eye_movement_score": features.eye_movement_score,
            "has_k_complex": features.has_k_complex,
            "has_sleep_spindle": features.has_sleep_spindle,
            "epoch_duration_s": features.epoch_duration_s,
        }

    # -- Cue timing and selection -------------------------------------------

    def compute_cue_timing(
        self,
        features: SleepEEGFeatures,
        rem_duration_s: float,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Compute optimal cue timing and readiness.

        Algorithm:
          1. REM must be stable (>5 min continuous)
          2. Theta must peak (theta > 1.5x alpha) — deepest REM trough
          3. No K-complex present (would indicate transition to N2)
          4. If all conditions met, cue is ready to deliver

        Returns
        -------
        dict with keys:
            ready : bool — whether conditions are met for cue delivery
            reason : str — human-readable explanation
            theta_alpha_ratio : float
            rem_stable : bool
            optimal_intensity : float — suggested cue intensity 0-1
        """
        theta_alpha_ratio = (
            features.theta_power / max(features.alpha_power, 1e-10)
        )
        rem_stable = rem_duration_s >= STABLE_REM_MIN_DURATION_S

        reasons: List[str] = []
        ready = True

        if not rem_stable:
            ready = False
            remaining = STABLE_REM_MIN_DURATION_S - rem_duration_s
            reasons.append(
                f"REM not stable yet ({rem_duration_s:.0f}s / "
                f"{STABLE_REM_MIN_DURATION_S:.0f}s, {remaining:.0f}s remaining)"
            )

        if theta_alpha_ratio < MIN_THETA_PEAK_RATIO:
            ready = False
            reasons.append(
                f"Theta not peaking (ratio {theta_alpha_ratio:.2f} < "
                f"{MIN_THETA_PEAK_RATIO})"
            )

        if features.has_k_complex:
            ready = False
            reasons.append("K-complex detected — possible N2 transition")

        if not reasons:
            reasons.append("All conditions met — cue delivery optimal")

        # Optimal intensity: lower during deeper REM (theta dominance)
        # to avoid waking the sleeper
        optimal_intensity = float(
            np.clip(0.5 - (theta_alpha_ratio - 1.5) * 0.1, 0.1, 0.7)
        )

        return {
            "ready": ready,
            "reason": "; ".join(reasons),
            "theta_alpha_ratio": float(theta_alpha_ratio),
            "rem_stable": rem_stable,
            "rem_duration_s": float(rem_duration_s),
            "optimal_intensity": optimal_intensity,
        }

    def select_cue_type(
        self,
        user_id: str = "default",
        preferred: Optional[CueType] = None,
    ) -> CueConfig:
        """Select the best cue type for a user based on their history.

        Strategy:
          - If user has prior success data, pick the cue type with highest
            success rate.
          - If no data or all equal, use preferred or default to audio.
          - Intensity starts at 0.3 and adjusts based on prior attempts.

        Returns
        -------
        CueConfig with recommended settings.
        """
        profile = self._profiles.get(user_id)

        best_type = preferred or CueType.AUDIO
        intensity = 0.3

        if profile and profile.cue_type_counts:
            # Find cue type with highest success rate (min 2 attempts)
            best_rate = -1.0
            for ct_str, rate in profile.cue_type_rates.items():
                count = profile.cue_type_counts.get(ct_str, 0)
                if count >= 2 and rate > best_rate:
                    best_rate = rate
                    try:
                        best_type = CueType(ct_str)
                    except ValueError:
                        pass

            # Adjust intensity based on success: lower if already working,
            # higher if not effective
            if profile.success_rate > 0.5:
                intensity = 0.2  # gentle — user is responsive
            elif profile.success_rate < 0.1 and profile.total_attempts > 5:
                intensity = 0.5  # stronger — user needs more stimulation

        # Pattern and duration vary by cue type
        patterns = {
            CueType.AUDIO: ("gentle_pulse", 3.0),
            CueType.HAPTIC: ("gradual_ramp", 2.0),
            CueType.LED: ("steady", 1.5),
        }
        pattern, duration = patterns.get(best_type, ("gentle_pulse", 3.0))

        return CueConfig(
            cue_type=best_type,
            intensity=intensity,
            duration_s=duration,
            pattern=pattern,
            repeat_count=3,
        )

    # -- Success tracking ---------------------------------------------------

    def track_induction_success(
        self,
        user_id: str,
        technique: InductionTechnique,
        cue_config: Optional[CueConfig],
        lucid_reported: bool,
        dream_recalled: bool = True,
        rem_duration_s: float = 0.0,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Record an induction attempt and update the user's profile.

        Returns
        -------
        dict with the attempt record and updated profile summary.
        """
        self._attempt_counter += 1
        attempt_id = f"{user_id}_{self._attempt_counter}"

        attempt = InductionAttempt(
            attempt_id=attempt_id,
            user_id=user_id,
            technique=technique,
            cue_config=cue_config,
            timestamp=time.time(),
            rem_duration_s=rem_duration_s,
            cue_delivered=cue_config is not None,
            lucid_reported=lucid_reported,
            dream_recalled=dream_recalled,
            notes=notes,
        )

        if user_id not in self._attempts:
            self._attempts[user_id] = []
        self._attempts[user_id].append(attempt)

        # Rebuild profile
        profile = self._rebuild_profile(user_id)

        return {
            "attempt_id": attempt_id,
            "recorded": True,
            "profile_summary": {
                "total_attempts": profile.total_attempts,
                "total_lucid": profile.total_lucid,
                "success_rate": profile.success_rate,
                "best_technique": profile.best_technique,
                "best_cue_type": profile.best_cue_type,
            },
        }

    def _rebuild_profile(self, user_id: str) -> LucidProfile:
        """Rebuild the user's LucidProfile from their attempts."""
        attempts = self._attempts.get(user_id, [])

        profile = LucidProfile(user_id=user_id, last_updated=time.time())

        if not attempts:
            self._profiles[user_id] = profile
            return profile

        profile.total_attempts = len(attempts)
        profile.total_lucid = sum(1 for a in attempts if a.lucid_reported)
        profile.success_rate = (
            profile.total_lucid / profile.total_attempts
            if profile.total_attempts > 0 else 0.0
        )

        # Per-technique stats
        tech_success: Dict[str, int] = {}
        tech_total: Dict[str, int] = {}
        for a in attempts:
            t = a.technique.value
            tech_total[t] = tech_total.get(t, 0) + 1
            if a.lucid_reported:
                tech_success[t] = tech_success.get(t, 0) + 1

        profile.technique_counts = tech_total
        profile.technique_rates = {
            t: tech_success.get(t, 0) / c
            for t, c in tech_total.items()
        }

        # Per-cue-type stats
        cue_success: Dict[str, int] = {}
        cue_total: Dict[str, int] = {}
        for a in attempts:
            if a.cue_config is not None:
                ct = a.cue_config.cue_type.value
                cue_total[ct] = cue_total.get(ct, 0) + 1
                if a.lucid_reported:
                    cue_success[ct] = cue_success.get(ct, 0) + 1

        profile.cue_type_counts = cue_total
        profile.cue_type_rates = {
            ct: cue_success.get(ct, 0) / c
            for ct, c in cue_total.items()
        }

        # Average REM duration before successful lucid attempts
        lucid_rem = [a.rem_duration_s for a in attempts if a.lucid_reported]
        profile.avg_rem_before_lucid_s = (
            float(np.mean(lucid_rem)) if lucid_rem else 0.0
        )

        # Best technique and cue type
        if profile.technique_rates:
            profile.best_technique = max(
                profile.technique_rates, key=profile.technique_rates.get  # type: ignore[arg-type]
            )
        if profile.cue_type_rates:
            profile.best_cue_type = max(
                profile.cue_type_rates, key=profile.cue_type_rates.get  # type: ignore[arg-type]
            )

        # Carry forward reality test stats
        old = self._profiles.get(user_id)
        if old:
            profile.reality_tests_completed = old.reality_tests_completed
            profile.reality_test_streak_days = old.reality_test_streak_days

        self._profiles[user_id] = profile
        return profile

    # -- Reality testing scheduler ------------------------------------------

    def schedule_reality_tests(
        self,
        user_id: str = "default",
        tests_per_day: int = DEFAULT_REALITY_TESTS_PER_DAY,
        window_start_h: int = REALITY_TEST_WINDOW_START_H,
        window_end_h: int = REALITY_TEST_WINDOW_END_H,
    ) -> Dict[str, Any]:
        """Generate a daily reality test schedule.

        Distributes tests semi-randomly across the waking window with
        minimum spacing to build habit without clustering.

        Returns
        -------
        dict with schedule details and times.
        """
        tests_per_day = max(1, min(tests_per_day, 30))
        window_h = window_end_h - window_start_h
        if window_h <= 0:
            window_h = 14  # fallback

        spacing_h = window_h / (tests_per_day + 1)

        times: List[float] = []
        for i in range(1, tests_per_day + 1):
            base_h = window_start_h + i * spacing_h
            # Add small random jitter (up to +/- 15 min)
            jitter_h = (hash(f"{user_id}_{i}") % 30 - 15) / 60.0
            t = base_h + jitter_h
            t = max(float(window_start_h), min(float(window_end_h), t))
            times.append(round(t, 2))

        times.sort()
        self._reality_test_schedules[user_id] = times

        # Format as readable times
        formatted = []
        for t in times:
            h = int(t)
            m = int((t - h) * 60)
            period = "AM" if h < 12 else "PM"
            display_h = h if h <= 12 else h - 12
            if display_h == 0:
                display_h = 12
            formatted.append(f"{display_h}:{m:02d} {period}")

        return {
            "user_id": user_id,
            "tests_per_day": tests_per_day,
            "window": f"{window_start_h}:00 - {window_end_h}:00",
            "times_h": times,
            "times_formatted": formatted,
            "tip": (
                "At each scheduled time, pause and ask: "
                "'Am I dreaming right now?' Look at text, check the time, "
                "try pushing a finger through your palm. "
                "This habit transfers to dreams."
            ),
        }

    def record_reality_test(
        self, user_id: str = "default"
    ) -> Dict[str, Any]:
        """Record that a user completed a reality test.

        Increments their count and updates streak tracking.
        """
        if user_id not in self._profiles:
            self._profiles[user_id] = LucidProfile(
                user_id=user_id, last_updated=time.time()
            )

        profile = self._profiles[user_id]
        profile.reality_tests_completed += 1
        profile.last_updated = time.time()

        return {
            "user_id": user_id,
            "reality_tests_completed": profile.reality_tests_completed,
            "streak_days": profile.reality_test_streak_days,
        }

    # -- Profile access -----------------------------------------------------

    def compute_lucid_profile(self, user_id: str) -> Optional[LucidProfile]:
        """Return the current LucidProfile for a user, or None."""
        if user_id in self._attempts:
            return self._rebuild_profile(user_id)
        return self._profiles.get(user_id)

    def profile_to_dict(self, profile: LucidProfile) -> Dict[str, Any]:
        """Convert a LucidProfile to a JSON-safe dictionary."""
        return {
            "user_id": profile.user_id,
            "total_attempts": profile.total_attempts,
            "total_lucid": profile.total_lucid,
            "success_rate": round(profile.success_rate, 4),
            "technique_rates": profile.technique_rates,
            "technique_counts": profile.technique_counts,
            "cue_type_rates": profile.cue_type_rates,
            "cue_type_counts": profile.cue_type_counts,
            "avg_rem_before_lucid_s": round(profile.avg_rem_before_lucid_s, 1),
            "best_technique": profile.best_technique,
            "best_cue_type": profile.best_cue_type,
            "reality_tests_completed": profile.reality_tests_completed,
            "reality_test_streak_days": profile.reality_test_streak_days,
            "last_updated": profile.last_updated,
        }


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_engine: Optional[LucidInductionEngine] = None


def get_lucid_induction_engine() -> LucidInductionEngine:
    """Return the module-level singleton engine instance."""
    global _engine
    if _engine is None:
        _engine = LucidInductionEngine()
    return _engine
