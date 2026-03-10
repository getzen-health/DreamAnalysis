"""Comprehensive tests for the SupplementTracker model and API routes."""
import os
import sys
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.supplement_knowledge import (
    check_interactions,
    get_supplement_knowledge,
    population_vs_personal,
)
from models.supplement_tracker import (
    SupplementTracker,
    SupplementEntry,
    BrainStateSnapshot,
    VALID_SUPPLEMENT_TYPES,
    _MIN_READINGS_FOR_CORRELATION,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def tracker():
    """Fresh SupplementTracker for each test."""
    return SupplementTracker()


@pytest.fixture
def base_time():
    """A fixed base timestamp for deterministic tests."""
    return 1_700_000_000.0


# ── Log supplement and retrieve ─────────────────────────────────────


def test_log_supplement_returns_entry_id(tracker, base_time):
    entry_id = tracker.log_supplement(
        "user1", "Omega-3", "supplement", 1000, "mg", timestamp=base_time,
    )
    assert isinstance(entry_id, str)
    assert len(entry_id) == 8


def test_get_log_returns_logged_entry(tracker, base_time):
    tracker.log_supplement(
        "user1", "Omega-3", "supplement", 1000, "mg", timestamp=base_time,
    )
    log = tracker.get_log("user1")
    assert len(log) == 1
    entry = log[0]
    assert entry["name"] == "Omega-3"
    assert entry["supplement_type"] == "supplement"
    assert entry["dosage"] == 1000
    assert entry["unit"] == "mg"
    assert entry["timestamp"] == base_time


def test_log_supplement_with_notes(tracker, base_time):
    tracker.log_supplement(
        "user1", "Vitamin D", "vitamin", 5000, "IU",
        timestamp=base_time, notes="with food",
    )
    log = tracker.get_log("user1")
    assert log[0]["notes"] == "with food"


def test_log_supplement_default_timestamp(tracker):
    before = time.time()
    tracker.log_supplement("user1", "Magnesium", "supplement", 400, "mg")
    after = time.time()
    log = tracker.get_log("user1")
    assert before <= log[0]["timestamp"] <= after


def test_log_multiple_supplements(tracker, base_time):
    tracker.log_supplement("user1", "Omega-3", "supplement", 1000, "mg", base_time)
    tracker.log_supplement("user1", "Vitamin D", "vitamin", 5000, "IU", base_time + 60)
    tracker.log_supplement("user1", "Omega-3", "supplement", 1000, "mg", base_time + 120)

    log = tracker.get_log("user1")
    assert len(log) == 3


def test_get_log_filter_by_name(tracker, base_time):
    tracker.log_supplement("user1", "Omega-3", "supplement", 1000, "mg", base_time)
    tracker.log_supplement("user1", "Vitamin D", "vitamin", 5000, "IU", base_time + 60)
    tracker.log_supplement("user1", "Omega-3", "supplement", 1000, "mg", base_time + 120)

    filtered = tracker.get_log("user1", supplement_name="Omega-3")
    assert len(filtered) == 2
    assert all(e["name"] == "Omega-3" for e in filtered)


def test_get_log_filter_case_insensitive(tracker, base_time):
    tracker.log_supplement("user1", "Omega-3", "supplement", 1000, "mg", base_time)
    filtered = tracker.get_log("user1", supplement_name="omega-3")
    assert len(filtered) == 1


def test_get_log_last_n(tracker, base_time):
    for i in range(10):
        tracker.log_supplement(
            "user1", "VitC", "vitamin", 500, "mg", base_time + i,
        )
    log = tracker.get_log("user1", last_n=3)
    assert len(log) == 3
    # Should be the last 3
    assert log[0]["timestamp"] == base_time + 7
    assert log[2]["timestamp"] == base_time + 9


def test_get_log_empty(tracker):
    log = tracker.get_log("user1")
    assert log == []


# ── Brain state logging ────────────────────────────────────────────


def test_log_brain_state(tracker, base_time):
    tracker.log_brain_state("user1", base_time, {
        "valence": 0.5, "arousal": 0.6,
        "stress_index": 0.2, "focus_index": 0.7,
        "alpha_beta_ratio": 1.2, "theta_power": 0.3, "faa": 0.1,
        "source": "eeg",
    })
    # Verify internal storage
    assert len(tracker._brain_states["user1"]) == 1
    bs = tracker._brain_states["user1"][0]
    assert bs.valence == 0.5
    assert bs.arousal == 0.6
    assert bs.stress_index == 0.2
    assert bs.focus_index == 0.7
    assert bs.source == "eeg"
    assert bs.alpha_beta_ratio == 1.2
    assert bs.theta_power == 0.3
    assert bs.faa == 0.1


def test_log_brain_state_partial_data(tracker, base_time):
    """Only valence provided; other fields default to 0.0."""
    tracker.log_brain_state("user1", base_time, {"valence": -0.3})
    bs = tracker._brain_states["user1"][0]
    assert bs.valence == -0.3
    assert bs.arousal == 0.0
    assert bs.stress_index == 0.0


# ── Correlation analysis ───────────────────────────────────────────


def _populate_for_correlation(
    tracker, base_time, post_valence=0.5, post_stress=0.1,
    ctrl_valence=0.0, ctrl_stress=0.3, n_post=6, n_ctrl=10,
):
    """Helper: log a supplement and brain states for correlation testing.

    Creates brain states in the 4-hour window after supplement intake
    (post-supplement) and brain states outside that window (control).
    """
    # Log supplement at base_time
    tracker.log_supplement(
        "user1", "TestSupp", "supplement", 500, "mg", base_time,
    )

    # Post-supplement brain states (within 4h = 14400 sec)
    for i in range(n_post):
        tracker.log_brain_state("user1", base_time + 600 + i * 300, {
            "valence": post_valence,
            "arousal": 0.5,
            "stress_index": post_stress,
            "focus_index": 0.6,
            "alpha_beta_ratio": 1.5,
            "theta_power": 0.25,
            "faa": 0.15,
        })

    # Control brain states (before supplement or well after window)
    for i in range(n_ctrl):
        tracker.log_brain_state("user1", base_time - 86400 + i * 300, {
            "valence": ctrl_valence,
            "arousal": 0.5,
            "stress_index": ctrl_stress,
            "focus_index": 0.5,
            "alpha_beta_ratio": 1.0,
            "theta_power": 0.3,
            "faa": 0.0,
        })


def test_analyze_correlations_positive_verdict(tracker, base_time):
    _populate_for_correlation(
        tracker, base_time,
        post_valence=0.4, post_stress=0.1,
        ctrl_valence=0.0, ctrl_stress=0.3,
    )
    result = tracker.analyze_correlations("user1", "TestSupp")
    assert result["verdict"] == "positive"
    assert result["avg_valence_shift"] > 0.05
    assert result["avg_stress_shift"] < -0.03
    assert result["sample_count_post"] == 6
    assert result["sample_count_control"] == 10


def test_analyze_correlations_negative_verdict_from_valence(tracker, base_time):
    _populate_for_correlation(
        tracker, base_time,
        post_valence=-0.3, post_stress=0.2,
        ctrl_valence=0.0, ctrl_stress=0.2,
    )
    result = tracker.analyze_correlations("user1", "TestSupp")
    assert result["verdict"] == "negative"


def test_analyze_correlations_negative_verdict_from_stress(tracker, base_time):
    _populate_for_correlation(
        tracker, base_time,
        post_valence=0.0, post_stress=0.4,
        ctrl_valence=0.0, ctrl_stress=0.2,
    )
    result = tracker.analyze_correlations("user1", "TestSupp")
    assert result["verdict"] == "negative"


def test_analyze_correlations_neutral_verdict(tracker, base_time):
    _populate_for_correlation(
        tracker, base_time,
        post_valence=0.02, post_stress=0.2,
        ctrl_valence=0.0, ctrl_stress=0.2,
    )
    result = tracker.analyze_correlations("user1", "TestSupp")
    assert result["verdict"] == "neutral"


def test_analyze_correlations_insufficient_data_no_entries(tracker):
    result = tracker.analyze_correlations("user1", "NonExistent")
    assert result["verdict"] == "insufficient_data"
    assert result["reason"] == "no_supplement_entries"


def test_analyze_correlations_insufficient_data_no_brain_states(tracker, base_time):
    tracker.log_supplement("user1", "TestSupp", "supplement", 500, "mg", base_time)
    result = tracker.analyze_correlations("user1", "TestSupp")
    assert result["verdict"] == "insufficient_data"
    assert result["reason"] == "no_brain_states"


def test_analyze_correlations_insufficient_data_too_few(tracker, base_time):
    tracker.log_supplement("user1", "TestSupp", "supplement", 500, "mg", base_time)
    # Only 2 post-supplement readings (below threshold of 5)
    for i in range(2):
        tracker.log_brain_state("user1", base_time + 600 + i * 300, {
            "valence": 0.5, "arousal": 0.5,
            "stress_index": 0.1, "focus_index": 0.6,
        })
    result = tracker.analyze_correlations("user1", "TestSupp")
    assert result["verdict"] == "insufficient_data"
    assert result["reason"] == "too_few_post_supplement_readings"
    assert result["sample_count_post"] == 2


def test_analyze_correlations_custom_window(tracker, base_time):
    tracker.log_supplement("user1", "TestSupp", "supplement", 500, "mg", base_time)

    # Brain state 2 hours after (within 4h default, but outside 1h window)
    for i in range(6):
        tracker.log_brain_state("user1", base_time + 7200 + i * 60, {
            "valence": 0.5, "arousal": 0.5,
            "stress_index": 0.1, "focus_index": 0.6,
        })

    # With 1-hour window, these should all be control (outside window)
    result = tracker.analyze_correlations("user1", "TestSupp", window_hours=1.0)
    assert result["verdict"] == "insufficient_data"
    assert result["sample_count_post"] == 0


def test_analyze_correlations_eeg_insights(tracker, base_time):
    _populate_for_correlation(tracker, base_time)
    result = tracker.analyze_correlations("user1", "TestSupp")
    assert "eeg_insights" in result
    insights = result["eeg_insights"]
    assert "alpha_beta_ratio_shift" in insights
    assert "theta_power_shift" in insights
    assert "faa_shift" in insights


def test_analyze_correlations_includes_voice_data_source_summary(tracker, base_time):
    tracker.log_supplement("user1", "TestSupp", "supplement", 500, "mg", base_time)

    for i in range(6):
        tracker.log_brain_state("user1", base_time + 600 + i * 300, {
            "valence": 0.4,
            "arousal": 0.6,
            "stress_index": 0.2,
            "focus_index": 0.6,
            "speech_rate": 4.8,
            "source": "voice",
        })

    for i in range(10):
        tracker.log_brain_state("user1", base_time - 86400 + i * 300, {
            "valence": 0.0,
            "arousal": 0.5,
            "stress_index": 0.3,
            "focus_index": 0.5,
            "speech_rate": 4.0,
            "source": "voice",
        })

    result = tracker.analyze_correlations("user1", "TestSupp")
    assert result["post_source_counts"]["voice"] == 6
    assert result["post_source_counts"]["eeg"] == 0
    assert result["data_source_summary"] == "Based on 6 voice check-ins (no EEG data)"
    assert "voice_insights" in result
    assert "speech_rate_shift" in result["voice_insights"]


def test_analyze_correlations_returns_means(tracker, base_time):
    _populate_for_correlation(tracker, base_time)
    result = tracker.analyze_correlations("user1", "TestSupp")
    assert "post_means" in result
    assert "control_means" in result
    assert "valence" in result["post_means"]
    assert "stress_index" in result["control_means"]


# ── Supplement report ───────────────────────────────────────────────


def test_get_supplement_report_empty(tracker):
    report = tracker.get_supplement_report("user1")
    assert report["total_supplements"] == 0
    assert report["supplements"] == []


def test_get_supplement_report_multiple_supplements(tracker, base_time):
    # Supplement 1: positive
    tracker.log_supplement("user1", "Omega-3", "supplement", 1000, "mg", base_time)
    for i in range(6):
        tracker.log_brain_state("user1", base_time + 600 + i * 300, {
            "valence": 0.5, "arousal": 0.5,
            "stress_index": 0.1, "focus_index": 0.7,
        })

    # Supplement 2: logged but insufficient data
    tracker.log_supplement("user1", "Vitamin D", "vitamin", 5000, "IU", base_time + 86400)

    # Control readings (well before supplements)
    for i in range(10):
        tracker.log_brain_state("user1", base_time - 172800 + i * 300, {
            "valence": 0.0, "arousal": 0.5,
            "stress_index": 0.3, "focus_index": 0.5,
        })

    report = tracker.get_supplement_report("user1")
    assert report["total_supplements"] == 2
    assert len(report["supplements"]) == 2

    names = [s["name"] for s in report["supplements"]]
    assert "Omega-3" in names
    assert "Vitamin D" in names

    omega = next(s for s in report["supplements"] if s["name"] == "Omega-3")
    assert omega["entry_count"] == 1
    assert "correlation" in omega


def test_get_supplement_report_counts_voice_and_eeg_states(tracker, base_time):
    tracker.log_supplement("user1", "Omega-3", "supplement", 1000, "mg", base_time)
    tracker.log_brain_state("user1", base_time + 60, {"valence": 0.2, "source": "voice"})
    tracker.log_brain_state("user1", base_time + 120, {"valence": 0.3, "source": "eeg"})

    report = tracker.get_supplement_report("user1")
    assert report["voice_brain_states"] == 1
    assert report["eeg_brain_states"] == 1


# ── Active supplements ──────────────────────────────────────────────


def test_get_active_supplements(tracker):
    now = time.time()
    tracker.log_supplement("user1", "Omega-3", "supplement", 1000, "mg", now - 3600)
    tracker.log_supplement("user1", "Vitamin D", "vitamin", 5000, "IU", now - 100000)

    active = tracker.get_active_supplements("user1", hours=24)
    assert len(active) == 1
    assert active[0]["name"] == "Omega-3"


def test_get_active_supplements_empty(tracker):
    active = tracker.get_active_supplements("user1")
    assert active == []


def test_get_active_supplements_custom_hours(tracker):
    now = time.time()
    tracker.log_supplement("user1", "Omega-3", "supplement", 1000, "mg", now - 7200)

    # 1-hour window: should not include
    active = tracker.get_active_supplements("user1", hours=1)
    assert len(active) == 0

    # 3-hour window: should include
    active = tracker.get_active_supplements("user1", hours=3)
    assert len(active) == 1


# ── User isolation ──────────────────────────────────────────────────


def test_user_isolation(tracker, base_time):
    tracker.log_supplement("user1", "Omega-3", "supplement", 1000, "mg", base_time)
    tracker.log_supplement("user2", "Vitamin D", "vitamin", 5000, "IU", base_time)

    log1 = tracker.get_log("user1")
    log2 = tracker.get_log("user2")
    assert len(log1) == 1
    assert len(log2) == 1
    assert log1[0]["name"] == "Omega-3"
    assert log2[0]["name"] == "Vitamin D"


def test_brain_state_user_isolation(tracker, base_time):
    tracker.log_brain_state("user1", base_time, {"valence": 0.5})
    tracker.log_brain_state("user2", base_time, {"valence": -0.5})

    assert len(tracker._brain_states["user1"]) == 1
    assert len(tracker._brain_states["user2"]) == 1
    assert tracker._brain_states["user1"][0].valence == 0.5
    assert tracker._brain_states["user2"][0].valence == -0.5


# ── Reset ───────────────────────────────────────────────────────────


def test_reset_clears_all_data(tracker, base_time):
    tracker.log_supplement("user1", "Omega-3", "supplement", 1000, "mg", base_time)
    tracker.log_brain_state("user1", base_time, {"valence": 0.5})

    tracker.reset("user1")

    assert tracker.get_log("user1") == []
    report = tracker.get_supplement_report("user1")
    assert report["total_supplements"] == 0
    assert report["total_brain_states"] == 0


def test_reset_does_not_affect_other_users(tracker, base_time):
    tracker.log_supplement("user1", "Omega-3", "supplement", 1000, "mg", base_time)
    tracker.log_supplement("user2", "Vitamin D", "vitamin", 5000, "IU", base_time)

    tracker.reset("user1")

    assert tracker.get_log("user1") == []
    assert len(tracker.get_log("user2")) == 1


def test_reset_nonexistent_user(tracker):
    """Reset on a user with no data should not raise."""
    tracker.reset("nobody")  # Should not raise


# ── Storage caps ────────────────────────────────────────────────────


def test_supplement_cap(base_time):
    tracker = SupplementTracker(max_supplements=10, max_brain_states=100)
    for i in range(15):
        tracker.log_supplement(
            "user1", f"Supp{i}", "supplement", 100, "mg", base_time + i,
        )
    log = tracker.get_log("user1", last_n=100)
    assert len(log) == 10
    # Oldest entries should have been trimmed
    assert log[0]["name"] == "Supp5"


def test_brain_state_cap(base_time):
    tracker = SupplementTracker(max_supplements=100, max_brain_states=10)
    for i in range(15):
        tracker.log_brain_state("user1", base_time + i, {"valence": i * 0.01})
    assert len(tracker._brain_states["user1"]) == 10


# ── Edge cases ──────────────────────────────────────────────────────


def test_supplement_name_stripped(tracker, base_time):
    tracker.log_supplement(
        "user1", "  Omega-3  ", "supplement", 1000, "mg", base_time,
    )
    log = tracker.get_log("user1")
    assert log[0]["name"] == "Omega-3"


def test_filter_nonexistent_supplement(tracker, base_time):
    tracker.log_supplement("user1", "Omega-3", "supplement", 1000, "mg", base_time)
    filtered = tracker.get_log("user1", supplement_name="NonExistent")
    assert filtered == []


def test_duplicate_supplement_names(tracker, base_time):
    """Multiple entries with the same name should all be stored."""
    for i in range(3):
        tracker.log_supplement(
            "user1", "Omega-3", "supplement", 1000, "mg", base_time + i * 3600,
        )
    log = tracker.get_log("user1", supplement_name="Omega-3")
    assert len(log) == 3


def test_correlation_with_no_control_readings(tracker, base_time):
    """All brain states fall within the supplement window -> empty control."""
    tracker.log_supplement("user1", "TestSupp", "supplement", 500, "mg", base_time)
    # All within 4h window
    for i in range(6):
        tracker.log_brain_state("user1", base_time + 600 + i * 300, {
            "valence": 0.5, "arousal": 0.5,
            "stress_index": 0.1, "focus_index": 0.6,
        })
    result = tracker.analyze_correlations("user1", "TestSupp")
    # With no control readings, control_means should all be 0
    assert result["sample_count_control"] == 0
    # Should still compute, comparing against 0
    assert result["verdict"] in ("positive", "negative", "neutral")


def test_valid_supplement_types():
    assert "vitamin" in VALID_SUPPLEMENT_TYPES
    assert "supplement" in VALID_SUPPLEMENT_TYPES
    assert "medication" in VALID_SUPPLEMENT_TYPES
    assert "food_supplement" in VALID_SUPPLEMENT_TYPES


def test_dataclass_to_dict():
    entry = SupplementEntry(
        entry_id="abc12345",
        name="TestSupp",
        supplement_type="vitamin",
        dosage=500,
        unit="mg",
        timestamp=1000.0,
        notes="test",
    )
    d = entry.to_dict()
    assert d["entry_id"] == "abc12345"
    assert d["name"] == "TestSupp"

    snapshot = BrainStateSnapshot(
        timestamp=1000.0,
        valence=0.5,
        arousal=0.6,
        stress_index=0.2,
        focus_index=0.7,
    )
    d = snapshot.to_dict()
    assert d["valence"] == 0.5
    assert d["alpha_beta_ratio"] == 0.0  # default


# ── Route integration (import check) ───────────────────────────────


def test_route_module_imports():
    """Verify the route module can be imported without errors."""
    from api.routes.supplement_tracker import router, get_tracker
    assert router is not None
    assert get_tracker() is not None


def test_get_supplement_knowledge_resolves_alias():
    knowledge = get_supplement_knowledge("fish-oil")
    assert knowledge is not None
    assert knowledge["canonical_name"] == "omega-3"
    assert knowledge["display_name"] == "Omega-3 (EPA>DHA)"


def test_check_interactions_resolves_aliases():
    result = check_interactions(["fish-oil", "theanine", "caffeine"])
    assert result["canonical_names"] == ["omega-3", "l-theanine", "caffeine"]
    assert result["interaction_count"] == 1
    assert result["interactions"][0]["supplements"] == ["caffeine", "l-theanine"]


def test_population_vs_personal_handles_stress_as_lower_is_better():
    comparison = population_vs_personal(
        {
            "avg_valence_shift": 0.02,
            "avg_stress_shift": -0.10,
            "avg_focus_shift": 0.02,
        },
        {
            "display_name": "Magnesium",
            "expected_effects": {
                "valence": 0.03,
                "stress_index": -0.07,
                "focus_index": 0.01,
            },
        },
    )

    stress_metric = next(m for m in comparison["metrics"] if m["metric"] == "stress_index")
    assert stress_metric["comparison"] == "above_average"


def test_compare_route_returns_stress_directionally_correct_label(base_time):
    from api.routes.supplement_tracker import get_tracker, router

    tracker = get_tracker()
    tracker.reset("compare_user")
    tracker.log_supplement("compare_user", "Magnesium", "supplement", 300, "mg", base_time)

    for i in range(6):
        tracker.log_brain_state("compare_user", base_time + 300 + i * 300, {
            "valence": 0.01,
            "arousal": 0.4,
            "stress_index": 0.05,
            "focus_index": 0.02,
        })
    for i in range(10):
        tracker.log_brain_state("compare_user", base_time - 86400 + i * 300, {
            "valence": 0.0,
            "arousal": 0.4,
            "stress_index": 0.20,
            "focus_index": 0.01,
        })

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    response = client.get("/supplements/compare/compare_user/magnesium")
    assert response.status_code == 200
    body = response.json()
    stress_metric = next(
        metric
        for metric in body["population_comparison"]["metrics"]
        if metric["metric"] == "stress_index"
    )
    assert stress_metric["comparison"] == "above_average"


def test_interactions_route_resolves_aliases():
    from api.routes.supplement_tracker import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    response = client.get("/supplements/interactions", params={"names": "fish-oil,theanine,caffeine"})
    assert response.status_code == 200
    body = response.json()
    assert body["canonical_names"] == ["omega-3", "l-theanine", "caffeine"]
    assert body["interaction_count"] == 1
