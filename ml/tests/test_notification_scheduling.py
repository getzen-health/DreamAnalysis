"""Integration tests for ml/notifications/smart_notifications.py."""

import time
from unittest.mock import patch


# ── NotificationPreferences ────────────────────────────────────────────────────

class TestNotificationPreferencesDefaults:
    def test_defaults(self):
        from notifications.smart_notifications import NotificationPreferences
        prefs = NotificationPreferences(user_id="u1")
        assert prefs.enabled is True
        assert prefs.morning_report_enabled is True
        assert prefs.evening_winddown_enabled is True
        assert prefs.post_session_enabled is True
        assert prefs.weekly_summary_enabled is True
        assert prefs.quiet_hours_start == 22
        assert prefs.quiet_hours_end == 7
        assert prefs.morning_hour == 8
        assert prefs.evening_hour == 21
        assert prefs.timezone_offset_hours == 0
        assert prefs.skip_weekends_morning is False
        assert prefs.min_stress_for_evening == pytest.approx(0.3, abs=1e-9)

    def test_to_dict_has_all_fields(self):
        from notifications.smart_notifications import NotificationPreferences
        prefs = NotificationPreferences(user_id="u2")
        d = prefs.to_dict()
        for key in [
            "user_id", "enabled", "morning_report_enabled", "evening_winddown_enabled",
            "post_session_enabled", "weekly_summary_enabled", "quiet_hours_start",
            "quiet_hours_end", "morning_hour", "evening_hour", "timezone_offset_hours",
            "skip_weekends_morning", "min_stress_for_evening", "updated_at",
        ]:
            assert key in d

    def test_from_dict_roundtrip(self):
        from notifications.smart_notifications import NotificationPreferences
        prefs = NotificationPreferences(user_id="u3", morning_hour=9, quiet_hours_start=23)
        d = prefs.to_dict()
        restored = NotificationPreferences.from_dict(d)
        assert restored.user_id == "u3"
        assert restored.morning_hour == 9
        assert restored.quiet_hours_start == 23

    def test_updated_at_is_set_on_creation(self):
        from notifications.smart_notifications import NotificationPreferences
        before = time.time()
        prefs = NotificationPreferences(user_id="u4")
        after = time.time()
        assert before <= prefs.updated_at <= after


import pytest


# ── MorningReportGenerator ─────────────────────────────────────────────────────

class TestMorningReportGenerator:
    def test_generate_returns_required_keys(self):
        from notifications.smart_notifications import MorningReportGenerator
        gen = MorningReportGenerator()
        result = gen.generate("test_user", voice_data={}, health_data={})
        for key in ["title", "body", "route", "scores", "has_data", "data_sources"]:
            assert key in result

    def test_route_is_brain_report(self):
        from notifications.smart_notifications import MorningReportGenerator
        gen = MorningReportGenerator()
        result = gen.generate("u1", voice_data={}, health_data={})
        assert result["route"] == "/brain-report"

    def test_no_data_message(self):
        from notifications.smart_notifications import MorningReportGenerator
        gen = MorningReportGenerator()
        result = gen.generate("u1", voice_data={}, health_data={})
        assert result["has_data"] is False
        assert "voice check-in" in result["body"].lower() or "report" in result["title"].lower()

    def test_with_voice_data(self):
        from notifications.smart_notifications import MorningReportGenerator
        gen = MorningReportGenerator()
        voice = {"avg_valence": 0.5, "avg_stress": 0.3, "avg_arousal": 0.6, "dominant_emotion": "happy"}
        result = gen.generate("u1", voice_data=voice, health_data={})
        assert result["has_data"] is True
        assert "voice" in result["data_sources"]

    def test_with_health_data(self):
        from notifications.smart_notifications import MorningReportGenerator
        gen = MorningReportGenerator()
        health = {"sleep_efficiency": 85, "hrv_sdnn": 55.0}
        result = gen.generate("u1", voice_data={}, health_data=health)
        assert result["has_data"] is True
        assert "health" in result["data_sources"]

    def test_scores_has_focus_and_readiness(self):
        from notifications.smart_notifications import MorningReportGenerator
        gen = MorningReportGenerator()
        voice = {"avg_valence": 0.6, "avg_stress": 0.2, "avg_arousal": 0.7}
        health = {"sleep_score": 80, "hrv_sdnn": 60.0}
        result = gen.generate("u1", voice_data=voice, health_data=health)
        scores = result["scores"]
        assert "focus" in scores
        assert "readiness" in scores
        assert 0 <= scores["focus"] <= 100
        assert 0 <= scores["readiness"] <= 100

    def test_sleep_score_in_scores_when_health_data_has_sleep(self):
        from notifications.smart_notifications import MorningReportGenerator
        gen = MorningReportGenerator()
        health = {"sleep_efficiency": 78}
        result = gen.generate("u1", voice_data={}, health_data=health)
        assert "sleep" in result["scores"]
        assert result["scores"]["sleep"] == pytest.approx(78.0, abs=0.5)

    def test_body_contains_focus_score(self):
        from notifications.smart_notifications import MorningReportGenerator
        gen = MorningReportGenerator()
        voice = {"avg_valence": 0.5, "avg_stress": 0.3}
        result = gen.generate("u1", voice_data=voice, health_data={})
        assert "Focus" in result["body"] or "focus" in result["body"].lower()

    def test_dominant_emotion_included_in_body(self):
        from notifications.smart_notifications import MorningReportGenerator
        gen = MorningReportGenerator()
        voice = {"avg_valence": 0.5, "avg_stress": 0.3, "dominant_emotion": "curious"}
        result = gen.generate("u1", voice_data=voice, health_data={})
        assert "curious" in result["body"]

    def test_neutral_emotion_not_in_body(self):
        from notifications.smart_notifications import MorningReportGenerator
        gen = MorningReportGenerator()
        voice = {"avg_valence": 0.0, "avg_stress": 0.5, "dominant_emotion": "neutral"}
        result = gen.generate("u1", voice_data=voice, health_data={})
        # neutral should be excluded from body
        assert "neutral" not in result["body"]


# ── EveningWindDownGenerator ───────────────────────────────────────────────────

class TestEveningWindDownGenerator:
    def test_generate_returns_required_keys(self):
        from notifications.smart_notifications import (
            EveningWindDownGenerator, NotificationPreferences
        )
        gen = EveningWindDownGenerator()
        prefs = NotificationPreferences(user_id="u1", min_stress_for_evening=0.0)
        result = gen.generate("u1", voice_data={}, health_data={}, prefs=prefs)
        for key in ["title", "body", "route", "activity_id", "activity_name", "stress_index", "skip"]:
            assert key in result

    def test_skip_when_stress_below_threshold(self):
        from notifications.smart_notifications import (
            EveningWindDownGenerator, NotificationPreferences
        )
        gen = EveningWindDownGenerator()
        prefs = NotificationPreferences(user_id="u1", min_stress_for_evening=0.8)
        # Default stress when no data = 0.4, which is below 0.8
        result = gen.generate("u1", voice_data={}, health_data={}, prefs=prefs)
        assert result["skip"] is True
        assert result["skip_reason"] is not None
        assert result["title"] == ""
        assert result["body"] == ""

    def test_not_skip_when_stress_above_threshold(self):
        from notifications.smart_notifications import (
            EveningWindDownGenerator, NotificationPreferences
        )
        gen = EveningWindDownGenerator()
        prefs = NotificationPreferences(user_id="u1", min_stress_for_evening=0.3)
        voice = {"avg_stress": 0.75}
        result = gen.generate("u1", voice_data=voice, health_data={}, prefs=prefs)
        assert result["skip"] is False
        assert result["title"] != ""
        assert result["body"] != ""

    def test_high_stress_picks_cyclic_sigh_or_box(self):
        from notifications.smart_notifications import (
            EveningWindDownGenerator, NotificationPreferences
        )
        gen = EveningWindDownGenerator()
        prefs = NotificationPreferences(user_id="u1", min_stress_for_evening=0.0)
        voice = {"avg_stress": 0.9}
        result = gen.generate("u1", voice_data=voice, health_data={}, prefs=prefs)
        # High stress → activity with high stress threshold selected
        # cyclic_sigh has threshold 0.6, box_breathing has 0.4
        assert result["activity_id"] in {"cyclic_sigh", "box_breathing"}

    def test_low_stress_picks_resonance_breathing(self):
        from notifications.smart_notifications import (
            EveningWindDownGenerator, NotificationPreferences
        )
        gen = EveningWindDownGenerator()
        prefs = NotificationPreferences(user_id="u1", min_stress_for_evening=0.0)
        voice = {"avg_stress": 0.05}  # very low stress
        result = gen.generate("u1", voice_data=voice, health_data={}, prefs=prefs)
        # Only resonance_breathing (threshold=0.0) qualifies
        assert result["activity_id"] == "resonance_breathing"

    def test_stress_index_in_valid_range(self):
        from notifications.smart_notifications import (
            EveningWindDownGenerator, NotificationPreferences
        )
        gen = EveningWindDownGenerator()
        prefs = NotificationPreferences(user_id="u1", min_stress_for_evening=0.0)
        voice = {"avg_stress": 0.6}
        result = gen.generate("u1", voice_data=voice, health_data={}, prefs=prefs)
        assert 0.0 <= result["stress_index"] <= 1.0

    def test_route_is_biofeedback_or_neurofeedback(self):
        from notifications.smart_notifications import (
            EveningWindDownGenerator, NotificationPreferences
        )
        gen = EveningWindDownGenerator()
        prefs = NotificationPreferences(user_id="u1", min_stress_for_evening=0.0)
        result = gen.generate("u1", voice_data={"avg_stress": 0.5}, health_data={}, prefs=prefs)
        assert result["route"] in {"/biofeedback", "/neurofeedback"}

    def test_hrv_contributes_to_stress_estimate(self):
        from notifications.smart_notifications import (
            EveningWindDownGenerator, NotificationPreferences
        )
        gen = EveningWindDownGenerator()
        prefs = NotificationPreferences(user_id="u1", min_stress_for_evening=0.0)
        # Low HRV → high stress signal
        result_low_hrv = gen.generate("u1", voice_data={}, health_data={"hrv_sdnn": 20.0}, prefs=prefs)
        # High HRV → low stress signal
        result_high_hrv = gen.generate("u1", voice_data={}, health_data={"hrv_sdnn": 80.0}, prefs=prefs)
        assert result_low_hrv["stress_index"] > result_high_hrv["stress_index"]


# ── NotificationScheduler ──────────────────────────────────────────────────────

class TestNotificationSchedulerQuietHours:
    def _make_scheduler(self):
        from notifications.smart_notifications import NotificationScheduler
        return NotificationScheduler()

    def test_in_quiet_hours_midnight(self):
        from notifications.smart_notifications import NotificationPreferences, NotificationScheduler
        prefs = NotificationPreferences(user_id="u1", quiet_hours_start=22, quiet_hours_end=7)
        assert NotificationScheduler._in_quiet_hours(0, prefs) is True   # midnight
        assert NotificationScheduler._in_quiet_hours(3, prefs) is True   # 3am
        assert NotificationScheduler._in_quiet_hours(6, prefs) is True   # 6am

    def test_not_in_quiet_hours_daytime(self):
        from notifications.smart_notifications import NotificationPreferences, NotificationScheduler
        prefs = NotificationPreferences(user_id="u1", quiet_hours_start=22, quiet_hours_end=7)
        assert NotificationScheduler._in_quiet_hours(8, prefs) is False
        assert NotificationScheduler._in_quiet_hours(12, prefs) is False
        assert NotificationScheduler._in_quiet_hours(21, prefs) is False

    def test_in_quiet_hours_late_evening(self):
        from notifications.smart_notifications import NotificationPreferences, NotificationScheduler
        prefs = NotificationPreferences(user_id="u1", quiet_hours_start=22, quiet_hours_end=7)
        assert NotificationScheduler._in_quiet_hours(22, prefs) is True
        assert NotificationScheduler._in_quiet_hours(23, prefs) is True

    def test_should_send_morning_blocked_during_quiet_hours(self):
        from notifications.smart_notifications import (
            NotificationScheduler, NotificationPreferences, _save_prefs
        )
        uid = "test_quiet_morning"
        prefs = NotificationPreferences(user_id=uid, morning_hour=2, quiet_hours_start=22, quiet_hours_end=7)
        _save_prefs(prefs)
        scheduler = self._make_scheduler()
        import datetime
        # UTC timestamp for 2am UTC (tz_offset=0 so local == UTC)
        now_dt = datetime.datetime(2024, 6, 15, 2, 0, 0, tzinfo=datetime.timezone.utc)
        ts = now_dt.timestamp()
        assert scheduler.should_send_morning(uid, now=ts) is False

    def test_should_send_evening_blocked_during_quiet_hours(self):
        from notifications.smart_notifications import (
            NotificationScheduler, NotificationPreferences, _save_prefs
        )
        uid = "test_quiet_evening"
        prefs = NotificationPreferences(user_id=uid, evening_hour=3, quiet_hours_start=22, quiet_hours_end=7)
        _save_prefs(prefs)
        scheduler = self._make_scheduler()
        import datetime
        # UTC timestamp for 3am UTC
        now_dt = datetime.datetime(2024, 6, 15, 3, 0, 0, tzinfo=datetime.timezone.utc)
        ts = now_dt.timestamp()
        assert scheduler.should_send_evening(uid, now=ts) is False


class TestNotificationSchedulerCooldown:
    def test_20_hour_cooldown_morning(self):
        from notifications.smart_notifications import (
            NotificationScheduler, NotificationPreferences, _save_prefs
        )
        uid = "test_cooldown_morning"
        prefs = NotificationPreferences(user_id=uid, morning_hour=8, quiet_hours_start=22, quiet_hours_end=7)
        _save_prefs(prefs)
        scheduler = NotificationScheduler()

        import datetime
        # Mark sent at 8am UTC
        now_dt = datetime.datetime(2024, 6, 15, 8, 0, 0, tzinfo=datetime.timezone.utc)
        ts = now_dt.timestamp()
        scheduler.mark_sent(uid, "morning_report", ts=ts)

        # 19 hours later — still on cooldown
        ts_19h = ts + 19 * 3600
        # 8am + 19h = 3am next day, which is in quiet hours anyway, but test cooldown logic
        # Check directly via _on_cooldown
        assert scheduler._on_cooldown(uid, "morning_report", ts_19h) is True

    def test_cooldown_expires_after_20_hours(self):
        from notifications.smart_notifications import NotificationScheduler, NotificationPreferences, _save_prefs
        uid = "test_cooldown_expire"
        prefs = NotificationPreferences(user_id=uid, morning_hour=8)
        _save_prefs(prefs)
        scheduler = NotificationScheduler()

        import datetime
        ts = datetime.datetime(2024, 6, 15, 8, 0, 0).timestamp()
        scheduler.mark_sent(uid, "morning_report", ts=ts)

        # 21 hours later — cooldown should have expired
        ts_21h = ts + 21 * 3600
        assert scheduler._on_cooldown(uid, "morning_report", ts_21h) is False

    def test_no_cooldown_for_never_sent(self):
        from notifications.smart_notifications import NotificationScheduler
        scheduler = NotificationScheduler()
        # New user_id that has never sent
        assert scheduler._on_cooldown("brand_new_user_xyz", "morning_report", time.time()) is False

    def test_evening_cooldown_20_hours(self):
        from notifications.smart_notifications import NotificationScheduler
        scheduler = NotificationScheduler()
        ts = time.time()
        scheduler.mark_sent("evn_user", "evening_winddown", ts=ts)
        # 15 hours later — still cooling down
        assert scheduler._on_cooldown("evn_user", "evening_winddown", ts + 15 * 3600) is True
        # 21 hours later — free
        assert scheduler._on_cooldown("evn_user", "evening_winddown", ts + 21 * 3600) is False


class TestNotificationSchedulerShouldSend:
    def test_should_send_morning_when_hour_matches(self):
        from notifications.smart_notifications import (
            NotificationScheduler, NotificationPreferences, _save_prefs
        )
        uid = "test_should_send_morning"
        prefs = NotificationPreferences(user_id=uid, morning_hour=8, quiet_hours_start=22, quiet_hours_end=7)
        _save_prefs(prefs)
        scheduler = NotificationScheduler()
        import datetime
        # UTC 8am on Monday June 17, 2024 (with tz_offset=0, local == UTC)
        ts = datetime.datetime(2024, 6, 17, 8, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        assert scheduler.should_send_morning(uid, now=ts) is True

    def test_should_not_send_morning_wrong_hour(self):
        from notifications.smart_notifications import (
            NotificationScheduler, NotificationPreferences, _save_prefs
        )
        uid = "test_wrong_hour"
        prefs = NotificationPreferences(user_id=uid, morning_hour=8, quiet_hours_start=22, quiet_hours_end=7)
        _save_prefs(prefs)
        scheduler = NotificationScheduler()
        import datetime
        # UTC 10am — not morning_hour=8
        ts = datetime.datetime(2024, 6, 17, 10, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        assert scheduler.should_send_morning(uid, now=ts) is False

    def test_should_not_send_when_disabled(self):
        from notifications.smart_notifications import (
            NotificationScheduler, NotificationPreferences, _save_prefs
        )
        uid = "test_disabled"
        prefs = NotificationPreferences(user_id=uid, enabled=False, morning_hour=8)
        _save_prefs(prefs)
        scheduler = NotificationScheduler()
        import datetime
        ts = datetime.datetime(2024, 6, 17, 8, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        assert scheduler.should_send_morning(uid, now=ts) is False

    def test_should_not_send_morning_when_morning_report_disabled(self):
        from notifications.smart_notifications import (
            NotificationScheduler, NotificationPreferences, _save_prefs
        )
        uid = "test_morning_disabled"
        prefs = NotificationPreferences(user_id=uid, morning_report_enabled=False, morning_hour=8)
        _save_prefs(prefs)
        scheduler = NotificationScheduler()
        import datetime
        ts = datetime.datetime(2024, 6, 17, 8, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        assert scheduler.should_send_morning(uid, now=ts) is False

    def test_skip_weekends_morning(self):
        from notifications.smart_notifications import (
            NotificationScheduler, NotificationPreferences, _save_prefs
        )
        uid = "test_skip_weekend"
        prefs = NotificationPreferences(
            user_id=uid, morning_hour=8, skip_weekends_morning=True, quiet_hours_start=22, quiet_hours_end=7
        )
        _save_prefs(prefs)
        scheduler = NotificationScheduler()
        import datetime
        # June 15, 2024 UTC = Saturday — should be skipped
        ts = datetime.datetime(2024, 6, 15, 8, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        assert scheduler.should_send_morning(uid, now=ts) is False

    def test_next_morning_ts_in_future(self):
        from notifications.smart_notifications import (
            NotificationScheduler, NotificationPreferences, _save_prefs
        )
        uid = "test_next_ts"
        prefs = NotificationPreferences(user_id=uid, morning_hour=8)
        _save_prefs(prefs)
        scheduler = NotificationScheduler()
        now = time.time()
        next_ts = scheduler.next_morning_ts(uid, now=now)
        assert next_ts > now

    def test_next_evening_ts_in_future(self):
        from notifications.smart_notifications import (
            NotificationScheduler, NotificationPreferences, _save_prefs
        )
        uid = "test_next_eve_ts"
        prefs = NotificationPreferences(user_id=uid, evening_hour=21)
        _save_prefs(prefs)
        scheduler = NotificationScheduler()
        now = time.time()
        next_ts = scheduler.next_evening_ts(uid, now=now)
        assert next_ts > now


# ── Wind-down activity selection ───────────────────────────────────────────────

class TestWindDownActivitySelection:
    def test_resonance_breathing_always_available(self):
        from notifications.smart_notifications import EveningWindDownGenerator
        gen = EveningWindDownGenerator()
        # stress=0.0 → only resonance_breathing qualifies (threshold=0.0)
        activity = gen._pick_activity(0.0)
        assert activity.id == "resonance_breathing"

    def test_moderate_stress_selects_high_threshold_candidate(self):
        from notifications.smart_notifications import EveningWindDownGenerator, WIND_DOWN_ACTIVITIES
        gen = EveningWindDownGenerator()
        # stress=0.35 → all activities with threshold ≤ 0.35 qualify;
        # the one with the highest threshold wins
        activity = gen._pick_activity(0.35)
        # Find the expected winner: max threshold among activities with threshold ≤ 0.35
        candidates = [a for a in WIND_DOWN_ACTIVITIES if 0.35 >= a.stress_threshold]
        expected = max(candidates, key=lambda a: a.stress_threshold)
        assert activity.id == expected.id

    def test_cyclic_sigh_at_high_stress(self):
        from notifications.smart_notifications import EveningWindDownGenerator
        gen = EveningWindDownGenerator()
        activity = gen._pick_activity(0.9)
        # cyclic_sigh has threshold 0.6, the highest applicable threshold at 0.9
        assert activity.id == "cyclic_sigh"

    def test_activity_has_required_fields(self):
        from notifications.smart_notifications import WIND_DOWN_ACTIVITIES
        for act in WIND_DOWN_ACTIVITIES:
            assert act.id
            assert act.name
            assert act.tagline
            assert act.duration_min > 0
            assert 0.0 <= act.stress_threshold <= 1.0
            assert act.route.startswith("/")

    def test_all_activities_have_biofeedback_or_neurofeedback_route(self):
        from notifications.smart_notifications import WIND_DOWN_ACTIVITIES
        for act in WIND_DOWN_ACTIVITIES:
            assert act.route in {"/biofeedback", "/neurofeedback"}
