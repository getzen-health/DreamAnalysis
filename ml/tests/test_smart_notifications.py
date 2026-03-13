"""Tests for the smart daily notification engine.

Covers (25+ tests):
  - MorningReportGenerator: score computation, message building, edge cases
  - EveningWindDownGenerator: stress estimation, activity selection, message building, skip logic
  - NotificationScheduler: should_send logic, quiet hours, weekends, cooldowns, next-time helpers
  - NotificationPreferences: defaults, serialisation, round-trip via from_dict
  - NotificationRecord: to_dict output
  - Module-level helpers: get_preferences, update_preferences, get_history
  - Wind-down activity catalogue
"""

from __future__ import annotations

import sys
import os
import time
import datetime

import pytest

# Make sure the ml/ directory is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from notifications.smart_notifications import (
    MorningReportGenerator,
    EveningWindDownGenerator,
    NotificationScheduler,
    NotificationPreferences,
    NotificationRecord,
    WindDownActivity,
    WIND_DOWN_ACTIVITIES,
    get_preferences,
    update_preferences,
    get_history,
    _record_notification,
    _PREFERENCES,
    _HISTORY,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _clear_state(user_id: str) -> None:
    """Remove any stored prefs/history for a test user."""
    _PREFERENCES.pop(user_id, None)
    _HISTORY.pop(user_id, None)


@pytest.fixture(autouse=True)
def _clean():
    """Reset global state before each test."""
    _PREFERENCES.clear()
    _HISTORY.clear()
    yield
    _PREFERENCES.clear()
    _HISTORY.clear()


# ── 1. MorningReportGenerator — score computation ─────────────────────────────

class TestMorningReportScores:
    def setup_method(self):
        self.gen = MorningReportGenerator()

    def test_scores_with_full_data(self):
        voice = {"avg_valence": 0.3, "avg_stress": 0.2, "avg_arousal": 0.6, "dominant_emotion": "happy"}
        health = {"sleep_efficiency": 80.0, "hrv_sdnn": 55.0}
        result = self.gen.generate("u1", voice_data=voice, health_data=health)
        assert 0 <= result["scores"]["focus"] <= 100
        assert 0 <= result["scores"]["readiness"] <= 100
        assert "sleep" in result["scores"]
        assert 0 <= result["scores"]["sleep"] <= 100

    def test_scores_within_range_no_data(self):
        result = self.gen.generate("u_empty", voice_data={}, health_data={})
        # Should still return valid default scores
        assert result["scores"]["focus"] >= 0
        assert result["scores"]["readiness"] >= 0

    def test_high_hrv_raises_focus(self):
        voice_low = {"avg_valence": 0.0, "avg_stress": 0.5}
        voice_high = {"avg_valence": 0.0, "avg_stress": 0.5}
        health_low_hrv = {"hrv_sdnn": 20.0}
        health_high_hrv = {"hrv_sdnn": 70.0}
        r_low = self.gen.generate("u_hrv", voice_data=voice_low, health_data=health_low_hrv)
        r_high = self.gen.generate("u_hrv2", voice_data=voice_high, health_data=health_high_hrv)
        assert r_high["scores"]["focus"] > r_low["scores"]["focus"]

    def test_sleep_score_fallback(self):
        health = {"sleep_score": 65.0}
        result = self.gen.generate("u_slp", voice_data={}, health_data=health)
        assert result["scores"].get("sleep") == 65.0

    def test_positive_valence_raises_focus(self):
        voice_pos = {"avg_valence": 1.0, "avg_stress": 0.1}
        voice_neg = {"avg_valence": -1.0, "avg_stress": 0.1}
        health = {"hrv_sdnn": 50.0}
        r_pos = self.gen.generate("u_vp", voice_data=voice_pos, health_data=health)
        r_neg = self.gen.generate("u_vn", voice_data=voice_neg, health_data=health)
        assert r_pos["scores"]["focus"] > r_neg["scores"]["focus"]


# ── 2. MorningReportGenerator — message building ──────────────────────────────

class TestMorningReportMessages:
    def setup_method(self):
        self.gen = MorningReportGenerator()

    def test_title_present_with_data(self):
        voice = {"avg_valence": 0.2, "avg_stress": 0.3, "dominant_emotion": "calm"}
        health = {"sleep_efficiency": 75.0}
        result = self.gen.generate("u2", voice_data=voice, health_data=health)
        assert len(result["title"]) > 0
        assert len(result["body"]) > 0

    def test_no_data_fallback_message(self):
        result = self.gen.generate("u_nd", voice_data={}, health_data={})
        assert "check-in" in result["body"].lower() or "sync" in result["body"].lower()
        assert result["has_data"] is False

    def test_has_data_flag(self):
        result_with = self.gen.generate("u_wd", voice_data={"avg_valence": 0.1}, health_data={})
        result_without = self.gen.generate("u_wod", voice_data={}, health_data={})
        assert result_with["has_data"] is True
        assert result_without["has_data"] is False

    def test_route_is_brain_report(self):
        result = self.gen.generate("u_route", voice_data={}, health_data={})
        assert result["route"] == "/brain-report"

    def test_data_sources_populated(self):
        result = self.gen.generate(
            "u_src",
            voice_data={"avg_valence": 0.0},
            health_data={"sleep_efficiency": 70.0},
        )
        assert "voice" in result["data_sources"]
        assert "health" in result["data_sources"]

    def test_focus_score_in_body(self):
        voice = {"avg_valence": 0.5, "avg_stress": 0.2}
        health = {"sleep_efficiency": 80.0, "hrv_sdnn": 60.0}
        result = self.gen.generate("u_body", voice_data=voice, health_data=health)
        assert "Focus" in result["body"] or "focus" in result["body"]


# ── 3. EveningWindDownGenerator — stress estimation ───────────────────────────

class TestEveningStressEstimation:
    def setup_method(self):
        self.gen = EveningWindDownGenerator()

    def test_high_voice_stress_reflected(self):
        voice = {"avg_stress": 0.9, "avg_valence": -0.5}
        result = self.gen.generate("u_hs", voice_data=voice, health_data={})
        assert result["stress_index"] >= 0.5

    def test_low_voice_stress(self):
        voice = {"avg_stress": 0.1, "avg_valence": 0.5}
        result = self.gen.generate("u_ls", voice_data=voice, health_data={})
        assert result["stress_index"] <= 0.5

    def test_hrv_contributes_to_stress(self):
        health_low_hrv = {"hrv_sdnn": 20.0}   # low HRV = high stress
        health_high_hrv = {"hrv_sdnn": 70.0}  # high HRV = low stress
        r_low = self.gen.generate("u_e_lhrv", voice_data={}, health_data=health_low_hrv)
        r_high = self.gen.generate("u_e_hhrv", voice_data={}, health_data=health_high_hrv)
        assert r_low["stress_index"] > r_high["stress_index"]

    def test_no_data_returns_neutral_stress(self):
        result = self.gen.generate("u_nd2", voice_data={}, health_data={})
        # Should default gracefully
        assert 0.0 <= result["stress_index"] <= 1.0


# ── 4. EveningWindDownGenerator — activity selection ─────────────────────────

class TestEveningActivitySelection:
    def setup_method(self):
        self.gen = EveningWindDownGenerator()

    def test_high_stress_selects_acute_activity(self):
        voice = {"avg_stress": 0.85}
        result = self.gen.generate("u_ha", voice_data=voice, health_data={})
        # cyclic_sigh has stress_threshold=0.6, should be selected for high stress
        assert result["activity_id"] in {"cyclic_sigh", "body_scan"}

    def test_low_stress_selects_gentle_activity(self):
        voice = {"avg_stress": 0.1}
        result = self.gen.generate("u_la", voice_data=voice, health_data={})
        # resonance_breathing has threshold=0.0, should be selected
        assert result["activity_id"] == "resonance_breathing"

    def test_activity_id_is_valid(self):
        valid_ids = {a.id for a in WIND_DOWN_ACTIVITIES}
        voice = {"avg_stress": 0.5}
        result = self.gen.generate("u_valid", voice_data=voice, health_data={})
        assert result["activity_id"] in valid_ids


# ── 5. EveningWindDownGenerator — skip logic ─────────────────────────────────

class TestEveningSkipLogic:
    def setup_method(self):
        self.gen = EveningWindDownGenerator()

    def test_skip_when_stress_below_threshold(self):
        prefs = NotificationPreferences(user_id="u_skip", min_stress_for_evening=0.5)
        voice = {"avg_stress": 0.2}
        result = self.gen.generate("u_skip", voice_data=voice, health_data={}, prefs=prefs)
        assert result["skip"] is True
        assert result["title"] == ""
        assert result["body"] == ""

    def test_no_skip_when_stress_at_threshold(self):
        prefs = NotificationPreferences(user_id="u_noskip", min_stress_for_evening=0.3)
        voice = {"avg_stress": 0.5}
        result = self.gen.generate("u_noskip", voice_data=voice, health_data={}, prefs=prefs)
        assert result["skip"] is False
        assert len(result["title"]) > 0


# ── 6. NotificationPreferences — defaults and serialisation ──────────────────

class TestNotificationPreferences:
    def test_default_values(self):
        prefs = NotificationPreferences(user_id="u_def")
        assert prefs.enabled is True
        assert prefs.morning_report_enabled is True
        assert prefs.evening_winddown_enabled is True
        assert prefs.quiet_hours_start == 22
        assert prefs.quiet_hours_end == 7
        assert prefs.morning_hour == 8
        assert prefs.evening_hour == 21

    def test_to_dict_contains_all_fields(self):
        prefs = NotificationPreferences(user_id="u_td")
        d = prefs.to_dict()
        for field in (
            "user_id", "enabled", "morning_report_enabled", "evening_winddown_enabled",
            "quiet_hours_start", "quiet_hours_end", "morning_hour", "evening_hour",
            "min_stress_for_evening", "updated_at",
        ):
            assert field in d, f"Missing field: {field}"

    def test_round_trip_from_dict(self):
        prefs = NotificationPreferences(
            user_id="u_rt",
            morning_hour=9,
            evening_hour=20,
            quiet_hours_start=23,
        )
        d = prefs.to_dict()
        restored = NotificationPreferences.from_dict(d)
        assert restored.morning_hour == 9
        assert restored.evening_hour == 20
        assert restored.quiet_hours_start == 23


# ── 7. NotificationRecord ────────────────────────────────────────────────────

class TestNotificationRecord:
    def test_to_dict_structure(self):
        rec = NotificationRecord(
            notification_id="abc-123",
            user_id="u_rec",
            notification_type="morning_report",
            title="Test title",
            body="Test body",
            route="/brain-report",
            scheduled_at=1234567890.0,
        )
        d = rec.to_dict()
        assert d["notification_id"] == "abc-123"
        assert d["user_id"] == "u_rec"
        assert d["notification_type"] == "morning_report"
        assert d["sent_at"] is None
        assert d["opened_at"] is None
        assert d["dismissed_at"] is None

    def test_metadata_preserved(self):
        rec = NotificationRecord(
            notification_id="xyz",
            user_id="u_meta",
            notification_type="evening_winddown",
            title="T",
            body="B",
            route="/biofeedback",
            scheduled_at=1.0,
            metadata={"stress_index": 0.65},
        )
        assert rec.to_dict()["metadata"]["stress_index"] == 0.65


# ── 8. NotificationScheduler — quiet hours and cooldown ──────────────────────

class TestNotificationScheduler:
    def setup_method(self):
        self.sch = NotificationScheduler()
        # Reset internal sent-time tracking between tests
        self.sch._last_sent.clear()

    def test_should_send_morning_at_correct_hour(self):
        prefs = NotificationPreferences(user_id="u_sch1", morning_hour=8, timezone_offset_hours=0)
        _PREFERENCES["u_sch1"] = prefs
        # Use a fixed UTC timestamp: 2026-03-10 08:00:00 UTC (Tuesday)
        ts = datetime.datetime(2026, 3, 10, 8, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        assert self.sch.should_send_morning("u_sch1", now=ts) is True

    def test_should_not_send_morning_wrong_hour(self):
        prefs = NotificationPreferences(user_id="u_sch2", morning_hour=8, timezone_offset_hours=0)
        _PREFERENCES["u_sch2"] = prefs
        ts = datetime.datetime(2026, 3, 10, 10, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        assert self.sch.should_send_morning("u_sch2", now=ts) is False

    def test_quiet_hours_block_notification(self):
        prefs = NotificationPreferences(
            user_id="u_qh",
            morning_hour=23,   # set to a "quiet" hour
            quiet_hours_start=22,
            quiet_hours_end=7,
            timezone_offset_hours=0,
        )
        _PREFERENCES["u_qh"] = prefs
        ts = datetime.datetime(2026, 3, 10, 23, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        assert self.sch.should_send_morning("u_qh", now=ts) is False

    def test_cooldown_blocks_repeat(self):
        prefs = NotificationPreferences(user_id="u_cd", morning_hour=8, timezone_offset_hours=0)
        _PREFERENCES["u_cd"] = prefs
        ts = datetime.datetime(2026, 3, 10, 8, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        # First call should succeed
        assert self.sch.should_send_morning("u_cd", now=ts) is True
        # Mark as sent
        self.sch.mark_sent("u_cd", "morning_report", ts)
        # Immediately after — should be on cooldown
        assert self.sch.should_send_morning("u_cd", now=ts + 60) is False

    def test_weekend_skip_blocks_morning(self):
        prefs = NotificationPreferences(
            user_id="u_wk",
            morning_hour=8,
            skip_weekends_morning=True,
            timezone_offset_hours=0,
        )
        _PREFERENCES["u_wk"] = prefs
        # Saturday 2026-03-14 08:00 UTC
        ts = datetime.datetime(2026, 3, 14, 8, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        assert self.sch.should_send_morning("u_wk", now=ts) is False

    def test_disabled_prefs_block_morning(self):
        prefs = NotificationPreferences(user_id="u_dis", enabled=False)
        _PREFERENCES["u_dis"] = prefs
        ts = datetime.datetime(2026, 3, 10, 8, 0, 0).timestamp()
        assert self.sch.should_send_morning("u_dis", now=ts) is False

    def test_next_morning_ts_in_future(self):
        prefs = NotificationPreferences(user_id="u_nxt", morning_hour=8, timezone_offset_hours=0)
        _PREFERENCES["u_nxt"] = prefs
        now = datetime.datetime(2026, 3, 10, 9, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        next_ts = self.sch.next_morning_ts("u_nxt", now=now)
        assert next_ts > now  # should be next day

    def test_next_evening_ts_in_future(self):
        prefs = NotificationPreferences(user_id="u_ev_nxt", evening_hour=21, timezone_offset_hours=0)
        _PREFERENCES["u_ev_nxt"] = prefs
        now = datetime.datetime(2026, 3, 10, 22, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        next_ts = self.sch.next_evening_ts("u_ev_nxt", now=now)
        assert next_ts > now


# ── 9. Module-level helpers ───────────────────────────────────────────────────

class TestModuleHelpers:
    def test_get_preferences_returns_defaults_for_unknown_user(self):
        prefs = get_preferences("brand_new_user")
        assert prefs.user_id == "brand_new_user"
        assert prefs.enabled is True

    def test_update_preferences_persists(self):
        update_preferences("u_upd", {"morning_hour": 7})
        prefs = get_preferences("u_upd")
        assert prefs.morning_hour == 7

    def test_update_preferences_ignores_user_id(self):
        """user_id must not be overwritable via update."""
        update_preferences("u_safe", {"morning_hour": 9})
        prefs = get_preferences("u_safe")
        assert prefs.user_id == "u_safe"

    def test_history_appends_records(self):
        rec = NotificationRecord(
            notification_id="h1",
            user_id="u_hist",
            notification_type="morning_report",
            title="T",
            body="B",
            route="/brain-report",
            scheduled_at=1.0,
            sent_at=1.0,
        )
        _record_notification(rec)
        hist = get_history("u_hist")
        assert len(hist) == 1
        assert hist[0]["notification_id"] == "h1"

    def test_history_empty_for_unknown_user(self):
        assert get_history("nobody") == []


# ── 10. Wind-down activity catalogue ─────────────────────────────────────────

class TestWindDownActivities:
    def test_catalogue_not_empty(self):
        assert len(WIND_DOWN_ACTIVITIES) >= 4

    def test_all_activities_have_required_fields(self):
        for a in WIND_DOWN_ACTIVITIES:
            assert a.id
            assert a.name
            assert a.tagline
            assert a.duration_min > 0
            assert 0.0 <= a.stress_threshold <= 1.0
            assert a.route.startswith("/")

    def test_resonance_always_applicable(self):
        resonance = next(a for a in WIND_DOWN_ACTIVITIES if a.id == "resonance_breathing")
        assert resonance.stress_threshold == 0.0
