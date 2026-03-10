"""Tests for JITAI engine: HRV trigger detection, Thompson Sampling bandit,
intensity-adaptive selection, and integration with the intervention system."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


# ── HRV Trigger Detection ───────────────────────────────────────────────────


def test_hrv_detector_baseline_tracking():
    """HRV detector should accumulate readings and compute rolling baseline."""
    from models.jitai_engine import HRVTriggerDetector

    det = HRVTriggerDetector()
    user = "test_hrv_001"

    # First reading initializes baseline
    status = det.add_reading(user, rmssd=45.0)
    assert status["count"] == 1
    assert status["baseline_ready"] is False
    assert abs(status["rmssd_mean"] - 45.0) < 0.1

    # Add more readings to stabilize
    for _ in range(4):
        status = det.add_reading(user, rmssd=45.0)

    assert status["count"] == 5
    assert status["baseline_ready"] is True
    assert abs(status["rmssd_mean"] - 45.0) < 1.0


def test_hrv_detector_trigger_on_rmssd_drop():
    """Trigger should fire when RMSSD drops > 1.5 SD below baseline."""
    from models.jitai_engine import HRVTriggerDetector

    det = HRVTriggerDetector()
    user = "test_hrv_002"

    # Build a baseline with meaningful variance around 50 ms.
    # Use wider spread so the EMA std is large enough that 45 is normal.
    readings = [50, 42, 58, 44, 56, 50, 42, 58, 44, 56,
                50, 48, 52, 46, 54, 50, 48, 52, 46, 54]
    for r in readings:
        det.add_reading(user, rmssd=float(r))

    # Reading close to mean — should not trigger
    result = det.check_trigger(user, current_rmssd=45.0)
    assert result["triggered"] is False
    assert result["baseline_ready"] is True

    # Dramatic drop well below baseline — should trigger
    result = det.check_trigger(user, current_rmssd=10.0)
    assert result["triggered"] is True
    assert result["reason"] == "rmssd_drop"
    assert result["rmssd_z_score"] < -1.5


def test_hrv_detector_no_trigger_insufficient_baseline():
    """Should not trigger if baseline is not yet stable (< MIN_READINGS)."""
    from models.jitai_engine import HRVTriggerDetector

    det = HRVTriggerDetector()
    user = "test_hrv_003"

    det.add_reading(user, rmssd=50.0)
    det.add_reading(user, rmssd=50.0)

    result = det.check_trigger(user, current_rmssd=10.0)
    assert result["triggered"] is False
    assert result["reason"] == "insufficient_baseline"
    assert result["baseline_ready"] is False


def test_hrv_detector_unknown_user():
    """Check trigger for unknown user should not crash."""
    from models.jitai_engine import HRVTriggerDetector

    det = HRVTriggerDetector()
    result = det.check_trigger("nonexistent", current_rmssd=30.0)
    assert result["triggered"] is False
    assert result["baseline_ready"] is False


def test_hrv_detector_get_baseline():
    """get_baseline should return None for unknown user, stats for known."""
    from models.jitai_engine import HRVTriggerDetector

    det = HRVTriggerDetector()
    assert det.get_baseline("nobody") is None

    det.add_reading("somebody", rmssd=40.0)
    bl = det.get_baseline("somebody")
    assert bl is not None
    assert bl["count"] == 1


# ── Bandit Cold Start ────────────────────────────────────────────────────────


def test_bandit_cold_start_low_intensity():
    """Cold start with low intensity should prefer cognitive_reappraisal."""
    from models.jitai_engine import InterventionBandit

    bandit = InterventionBandit()
    available = [
        "breathing", "cyclic_sighing", "cognitive_reappraisal",
        "slow_breathing", "body_scan",
    ]
    selected = bandit.select("cold_user", available, intensity=0.2)
    assert selected == "cognitive_reappraisal"


def test_bandit_cold_start_medium_intensity():
    """Cold start with medium intensity should prefer slow_breathing."""
    from models.jitai_engine import InterventionBandit

    bandit = InterventionBandit()
    available = [
        "breathing", "cyclic_sighing", "cognitive_reappraisal",
        "slow_breathing", "body_scan",
    ]
    selected = bandit.select("cold_user", available, intensity=0.5)
    assert selected == "slow_breathing"


def test_bandit_cold_start_high_intensity():
    """Cold start with high intensity should prefer cyclic_sighing."""
    from models.jitai_engine import InterventionBandit

    bandit = InterventionBandit()
    available = [
        "breathing", "cyclic_sighing", "cognitive_reappraisal",
        "slow_breathing", "body_scan", "grounding_54321",
    ]
    selected = bandit.select("cold_user", available, intensity=0.85)
    assert selected == "cyclic_sighing"


def test_bandit_cold_start_high_intensity_no_sighing():
    """If cyclic_sighing not available, high intensity should pick grounding."""
    from models.jitai_engine import InterventionBandit

    bandit = InterventionBandit()
    available = ["breathing", "grounding_54321", "slow_breathing"]
    selected = bandit.select("cold_user", available, intensity=0.9)
    assert selected == "grounding_54321"


def test_bandit_cold_start_fallback():
    """If no preferred intervention is available, pick first available."""
    from models.jitai_engine import InterventionBandit

    bandit = InterventionBandit()
    available = ["music_focus", "food"]
    selected = bandit.select("cold_user", available, intensity=0.3)
    # cognitive_reappraisal is preferred but not available → fallback
    assert selected in available


def test_bandit_cold_start_empty_available():
    """Empty available list should fall back to 'breathing'."""
    from models.jitai_engine import InterventionBandit

    bandit = InterventionBandit()
    selected = bandit.select("cold_user", [], intensity=0.5)
    assert selected == "breathing"


# ── Bandit After Outcomes ────────────────────────────────────────────────────


def test_bandit_after_threshold_uses_thompson():
    """After 10+ outcomes, bandit should use Thompson Sampling (not defaults)."""
    from models.jitai_engine import InterventionBandit, COLD_START_THRESHOLD

    bandit = InterventionBandit()
    user = "trained_user"
    available = ["breathing", "slow_breathing", "cyclic_sighing"]

    # Feed 10 outcomes: breathing always rewarded, others not
    for _ in range(COLD_START_THRESHOLD):
        bandit.update(user, "breathing", reward=0.9)
        bandit.update(user, "slow_breathing", reward=0.1)

    # After enough updates, bandit should mostly prefer breathing
    # Run 20 selections and check breathing is picked most often
    counts = {"breathing": 0, "slow_breathing": 0, "cyclic_sighing": 0}
    np.random.seed(42)
    for _ in range(20):
        pick = bandit.select(user, available, intensity=0.5)
        counts[pick] += 1

    # Breathing should dominate (high alpha vs low beta)
    assert counts["breathing"] > counts["slow_breathing"]


def test_bandit_convergence_toward_effective():
    """Bandit should converge toward interventions with higher reward."""
    from models.jitai_engine import InterventionBandit

    bandit = InterventionBandit()
    user = "converge_user"

    # Train: body_scan always effective, others mediocre
    for _ in range(20):
        bandit.update(user, "body_scan", reward=0.95)
        bandit.update(user, "walking", reward=0.3)
        bandit.update(user, "music_calm", reward=0.2)

    stats = bandit.get_stats(user)
    assert stats["is_cold_start"] is False
    # body_scan should have highest estimated reward
    assert stats["arms"]["body_scan"]["estimated_reward"] > stats["arms"]["walking"]["estimated_reward"]
    assert stats["arms"]["body_scan"]["estimated_reward"] > stats["arms"]["music_calm"]["estimated_reward"]


def test_bandit_update_clamps_reward():
    """Reward should be clamped to [0, 1]."""
    from models.jitai_engine import InterventionBandit

    bandit = InterventionBandit()
    # Should not crash with out-of-range values
    bandit.update("clamp_user", "breathing", reward=1.5)
    bandit.update("clamp_user", "breathing", reward=-0.5)

    stats = bandit.get_stats("clamp_user")
    # Alpha should be initial(1) + 1.0 (clamped) + 0.0 (clamped) = 2.0
    assert stats["arms"]["breathing"]["alpha"] == 2.0
    # Beta should be initial(1) + 0.0 + 1.0 = 2.0
    assert stats["arms"]["breathing"]["beta"] == 2.0


def test_bandit_stats_unknown_user():
    """Stats for unknown user should return empty arms and cold start."""
    from models.jitai_engine import InterventionBandit

    bandit = InterventionBandit()
    stats = bandit.get_stats("nobody")
    assert stats["total_outcomes"] == 0
    assert stats["is_cold_start"] is True
    assert stats["arms"] == {}


# ── JITAI Integration with Interventions ─────────────────────────────────────


def _get_jitai_check():
    """Return the sync JITAI check logic function."""
    from api.routes.interventions import _jitai_check_logic
    return _jitai_check_logic


def _reset_state():
    """Clear per-user state for test isolation."""
    from api.routes.interventions import _state, _intervention_history
    _state.clear()
    _intervention_history.clear()


def test_jitai_check_no_trigger():
    """JITAI check with normal readings should return no recommendation."""
    from api.routes.interventions import JITAICheckRequest
    _reset_state()

    req = JITAICheckRequest(
        user_id="jitai_001",
        stress_index=0.3,
        focus_index=0.6,
        emotion_intensity=0.2,
    )
    result = _get_jitai_check()(req)
    assert result["has_recommendation"] is False
    assert result["intervention"] is None


def test_jitai_check_stress_trigger():
    """High stress should trigger JITAI recommendation."""
    from api.routes.interventions import JITAICheckRequest
    _reset_state()

    req = JITAICheckRequest(
        user_id="jitai_002",
        stress_index=0.8,
        focus_index=0.6,
        emotion_intensity=0.8,
    )
    result = _get_jitai_check()(req)
    assert result["has_recommendation"] is True
    assert result["intervention"] is not None
    assert result["trigger_source"] == "eeg"
    assert result["intervention"]["type"] in [
        "cyclic_sighing", "grounding_54321", "slow_breathing", "breathing",
    ]


def test_jitai_check_hrv_trigger():
    """RMSSD drop should trigger JITAI recommendation via HRV path."""
    from api.routes.interventions import (
        JITAICheckRequest, _hrv_detector as det
    )
    _reset_state()

    user = "jitai_hrv_003"
    # Build HRV baseline
    for _ in range(10):
        det.add_reading(user, rmssd=50.0)

    req = JITAICheckRequest(
        user_id=user,
        stress_index=0.3,  # low stress from EEG
        focus_index=0.6,
        emotion_intensity=0.5,
        rmssd=20.0,  # dramatic drop → trigger
    )
    result = _get_jitai_check()(req)
    assert result["has_recommendation"] is True
    assert result["trigger_source"] == "hrv"


def test_jitai_check_voice_trigger():
    """High-arousal voice reading should trigger recommendation."""
    from api.routes.interventions import JITAICheckRequest
    _reset_state()

    req = JITAICheckRequest(
        user_id="jitai_voice_004",
        stress_index=0.3,
        focus_index=0.6,
        emotion_intensity=0.6,
        voice_emotion={
            "emotion": "angry",
            "valence": -0.7,
            "arousal": 0.85,
            "confidence": 0.9,
        },
    )
    result = _get_jitai_check()(req)
    assert result["has_recommendation"] is True
    assert result["trigger_source"] == "voice"


def test_jitai_check_respects_snooze():
    """JITAI should respect the existing snooze mechanism."""
    from api.routes.interventions import JITAICheckRequest, _user_state
    import time
    _reset_state()

    user = "jitai_snooze_005"
    st = _user_state(user)
    st["last_snooze_until"] = time.time() + 600  # snoozed for 10 min

    req = JITAICheckRequest(
        user_id=user,
        stress_index=0.9,
        emotion_intensity=0.9,
    )
    result = _get_jitai_check()(req)
    assert result["has_recommendation"] is False
    assert result.get("snoozed") is True


def test_jitai_check_respects_cooldown():
    """JITAI should respect the existing cooldown mechanism."""
    from api.routes.interventions import JITAICheckRequest, _user_state
    import time
    _reset_state()

    user = "jitai_cd_006"
    st = _user_state(user)
    st["last_triggered_ts"] = time.time()  # just triggered

    req = JITAICheckRequest(
        user_id=user,
        stress_index=0.9,
        emotion_intensity=0.9,
    )
    result = _get_jitai_check()(req)
    assert result["has_recommendation"] is False
    assert result.get("cooldown_active") is True


def test_jitai_outcome_updates_bandit():
    """JITAI outcome endpoint should update the bandit with a reward."""
    from api.routes.interventions import (
        JITAICheckRequest, OutcomeRequest,
        _jitai_check_logic, _bandit, _user_history,
    )
    import time
    _reset_state()

    user = "jitai_outcome_007"

    # Record a trigger manually in history
    entry = {
        "type": "slow_breathing",
        "triggered_at": time.time(),
        "stress_before": 0.8,
        "focus_before": 0.5,
        "stress_after": None,
        "focus_after": None,
        "felt_helpful": None,
        "outcome_recorded_at": None,
    }
    _user_history(user).append(entry)

    # Import the async function and call its sync logic
    from api.routes.interventions import jitai_outcome
    import asyncio

    outcome_req = OutcomeRequest(
        user_id=user,
        intervention_type="slow_breathing",
        stress_after=0.3,
        focus_after=0.7,
        felt_helpful=True,
    )
    result = asyncio.get_event_loop().run_until_complete(jitai_outcome(outcome_req))
    assert result["ok"] is True
    assert result["stress_delta"] > 0  # stress dropped
    assert result["bandit_reward"] > 0.5  # meaningful reward

    # Verify bandit was updated
    stats = _bandit.get_stats(user)
    assert stats["total_outcomes"] >= 1
    assert "slow_breathing" in stats["arms"]


def test_jitai_intensity_adaptive_selection():
    """Verify intensity maps to appropriate interventions during cold start.

    Note: when stress_index triggers, intensity = max(emotion_intensity, stress_index),
    so a high stress_index raises the effective intensity.
    """
    from api.routes.interventions import JITAICheckRequest, _hrv_detector
    _reset_state()

    # Low intensity via HRV trigger (stress_index stays low, so intensity stays low)
    user_low = "adapt_low"
    # Build HRV baseline then trigger via drop
    for _ in range(20):
        _hrv_detector.add_reading(user_low, rmssd=50.0)

    req_low = JITAICheckRequest(
        user_id=user_low,
        stress_index=0.3,  # below stress threshold — won't inflate intensity
        focus_index=0.6,
        emotion_intensity=0.2,
        rmssd=10.0,  # HRV drop triggers
    )
    result_low = _get_jitai_check()(req_low)
    assert result_low["has_recommendation"] is True
    # At low intensity, cognitive_reappraisal or body_scan preferred
    assert result_low["intervention"]["type"] in [
        "cognitive_reappraisal", "body_scan", "slow_breathing",
    ]

    # High intensity → should get cyclic_sighing or grounding
    _reset_state()
    req_high = JITAICheckRequest(
        user_id="adapt_high",
        stress_index=0.9,
        emotion_intensity=0.9,
    )
    result_high = _get_jitai_check()(req_high)
    assert result_high["has_recommendation"] is True
    assert result_high["intervention"]["type"] in [
        "cyclic_sighing", "grounding_54321", "slow_breathing", "breathing",
    ]


def test_new_interventions_in_catalogue():
    """All new JITAI interventions should appear in the INTERVENTIONS dict."""
    from api.routes.interventions import INTERVENTIONS

    new_types = [
        "cyclic_sighing", "grounding_54321", "body_scan",
        "cognitive_reappraisal", "slow_breathing",
    ]
    for t in new_types:
        assert t in INTERVENTIONS, f"{t} missing from INTERVENTIONS"
        assert INTERVENTIONS[t]["type"] == t
        assert "evidence" in INTERVENTIONS[t]
        assert "duration_seconds" in INTERVENTIONS[t]
        assert "intensity_range" in INTERVENTIONS[t]


def test_existing_interventions_unchanged():
    """Original 5 intervention types must still be present and unchanged."""
    from api.routes.interventions import INTERVENTIONS

    originals = ["breathing", "music_calm", "music_focus", "food", "walk"]
    for t in originals:
        assert t in INTERVENTIONS, f"Original intervention {t} was removed"
        assert INTERVENTIONS[t]["type"] == t


def test_existing_check_endpoint_not_modified():
    """The existing /interventions/check logic must work exactly as before."""
    from api.routes.interventions import CheckRequest, _check_intervention_logic, _state
    _state.clear()

    # Normal state → no intervention
    req = CheckRequest(user_id="legacy_check_001")
    result = _check_intervention_logic(req)
    assert result["has_recommendation"] is False

    # With voice → voice intervention
    req2 = CheckRequest(
        user_id="legacy_check_002",
        voice_emotion={
            "emotion": "angry",
            "valence": -0.7,
            "arousal": 0.85,
            "confidence": 0.9,
        },
    )
    result2 = _check_intervention_logic(req2)
    assert result2["has_recommendation"] is True
    assert result2["intervention"]["type"] == "voice_stress"
