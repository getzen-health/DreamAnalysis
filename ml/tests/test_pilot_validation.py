"""Tests for pilot study validation and analysis (issue #200).

Covers:
  - SQI computation (clean, noisy, empty, multichannel, short signals)
  - Session completeness (complete, partial, missing phases, snake_case keys)
  - Participant summary (multiple sessions, no sessions, mood stability)
  - Pilot-wide statistics (multi-participant, block distribution, separation)
  - Readiness report (all-pass, failing checks, edge cases)
  - report_to_dict serialization
  - API route smoke tests via TestClient
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.pilot_validation import (
    DEFAULT_AMPLITUDE_THRESHOLD_UV,
    DEFAULT_FS,
    MIN_COMPLETION_RATE,
    MIN_MEAN_SQI,
    MIN_PARTICIPANTS,
    REQUIRED_PHASES,
    ReadinessReport,
    compute_participant_summary,
    compute_pilot_statistics,
    compute_session_sqi,
    generate_readiness_report,
    report_to_dict,
    validate_session_completeness,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _clean_eeg(n_samples: int = 1024, n_channels: int = 1, rng=None) -> np.ndarray:
    """Synthetic EEG well below 75 uV threshold (~20 uV RMS)."""
    if rng is None:
        rng = np.random.default_rng(42)
    if n_channels == 1:
        return rng.normal(0, 20, n_samples)
    return rng.normal(0, 20, (n_channels, n_samples))


def _noisy_eeg(n_samples: int = 1024, rng=None) -> np.ndarray:
    """Synthetic EEG with many epochs exceeding 75 uV (~200 uV RMS)."""
    if rng is None:
        rng = np.random.default_rng(99)
    return rng.normal(0, 200, n_samples)


def _complete_session(code: str = "P001", block: str = "stress") -> dict:
    """A fully complete session record (camelCase keys)."""
    return {
        "participantCode": code,
        "blockType": block,
        "preEegJson": {"signals": [[1.0, 2.0, 3.0]]},
        "postEegJson": {"signals": [[4.0, 5.0, 6.0]]},
        "eegFeaturesJson": {"alpha": 0.5, "beta": 0.3},
        "surveyJson": {"mood": 7, "stress": 3},
        "dataQualityScore": 80,
    }


def _partial_session(code: str = "P001", block: str = "food") -> dict:
    """A session missing biofeedback and survey."""
    return {
        "participantCode": code,
        "blockType": block,
        "preEegJson": {"signals": [[1.0]]},
        "postEegJson": {"signals": [[2.0]]},
        "eegFeaturesJson": None,
        "surveyJson": None,
        "dataQualityScore": 50,
    }


# ── 1. SQI tests ────────────────────────────────────────────────────────────


class TestComputeSessionSQI:
    """Tests for compute_session_sqi()."""

    def test_clean_signal_high_sqi(self):
        """Clean 20 uV signal should produce SQI close to 1.0."""
        result = compute_session_sqi(_clean_eeg())
        assert result["sqi"] >= 0.9
        assert result["total_epochs"] > 0
        assert result["good_epochs"] == result["total_epochs"]
        assert result["bad_epochs"] == 0

    def test_noisy_signal_low_sqi(self):
        """200 uV RMS signal should produce very low SQI."""
        result = compute_session_sqi(_noisy_eeg())
        assert result["sqi"] < 0.5
        assert result["bad_epochs"] > 0

    def test_empty_signal(self):
        """Empty array should return zero SQI gracefully."""
        result = compute_session_sqi([])
        assert result["sqi"] == 0.0
        assert result["total_epochs"] == 0
        assert result["good_epochs"] == 0

    def test_multichannel_sqi(self):
        """4-channel EEG should compute SQI using max-abs across channels."""
        eeg = _clean_eeg(n_samples=1024, n_channels=4)
        result = compute_session_sqi(eeg)
        assert 0.0 <= result["sqi"] <= 1.0
        assert result["total_epochs"] == 4  # 1024 / 256

    def test_short_signal_treated_as_one_epoch(self):
        """Fewer samples than one epoch should still produce a result."""
        short = np.array([10.0, 20.0, 30.0])
        result = compute_session_sqi(short, fs=256.0, epoch_length_sec=1.0)
        assert result["total_epochs"] == 1
        assert result["sqi"] == 1.0  # max(abs) = 30 < 75

    def test_custom_threshold(self):
        """Custom amplitude threshold should change which epochs pass."""
        eeg = np.ones(256) * 50.0  # all 50 uV
        strict = compute_session_sqi(eeg, amplitude_threshold_uv=40.0)
        lenient = compute_session_sqi(eeg, amplitude_threshold_uv=60.0)
        assert strict["sqi"] == 0.0  # 50 > 40
        assert lenient["sqi"] == 1.0  # 50 < 60


# ── 2. Session completeness tests ───────────────────────────────────────────


class TestValidateSessionCompleteness:
    """Tests for validate_session_completeness()."""

    def test_complete_session(self):
        result = validate_session_completeness(_complete_session())
        assert result["is_complete"] is True
        assert result["missing_phases"] == []
        assert set(result["present_phases"]) == REQUIRED_PHASES

    def test_partial_session(self):
        result = validate_session_completeness(_partial_session())
        assert result["is_complete"] is False
        assert "biofeedback" in result["missing_phases"]
        assert "survey" in result["missing_phases"]

    def test_snake_case_keys(self):
        """Should accept snake_case column names too."""
        session = {
            "pre_eeg_json": {"data": 1},
            "post_eeg_json": {"data": 2},
            "eeg_features_json": {"data": 3},
            "survey_json": {"data": 4},
        }
        result = validate_session_completeness(session)
        assert result["is_complete"] is True

    def test_empty_dict_treated_as_missing(self):
        """Empty dict values should count as missing."""
        session = {
            "preEegJson": {},
            "postEegJson": {"data": 1},
            "eegFeaturesJson": {"data": 2},
            "surveyJson": {"data": 3},
        }
        result = validate_session_completeness(session)
        assert result["is_complete"] is False
        assert "baseline" in result["missing_phases"]


# ── 3. Participant summary tests ─────────────────────────────────────────────


class TestComputeParticipantSummary:
    """Tests for compute_participant_summary()."""

    def test_all_complete(self):
        sessions = [_complete_session("P001", "stress"), _complete_session("P001", "food")]
        result = compute_participant_summary("P001", sessions)
        assert result["participant_code"] == "P001"
        assert result["total_sessions"] == 2
        assert result["complete_sessions"] == 2
        assert result["completion_rate"] == 1.0

    def test_mixed_sessions(self):
        sessions = [_complete_session("P002"), _partial_session("P002")]
        result = compute_participant_summary("P002", sessions)
        assert result["completion_rate"] == 0.5

    def test_no_sessions(self):
        result = compute_participant_summary("P099", [])
        assert result["total_sessions"] == 0
        assert result["completion_rate"] == 0.0
        assert result["avg_sqi"] is None
        assert result["mood_stability"] is None

    def test_mood_stability_computed(self):
        """Mood stability is std-dev of data_quality_score across sessions."""
        s1 = _complete_session("P003")
        s1["dataQualityScore"] = 80
        s2 = _complete_session("P003")
        s2["dataQualityScore"] = 40
        result = compute_participant_summary("P003", [s1, s2])
        assert result["mood_stability"] is not None
        assert result["mood_stability"] > 0

    def test_block_counts(self):
        sessions = [
            _complete_session("P004", "stress"),
            _complete_session("P004", "stress"),
            _complete_session("P004", "food"),
        ]
        result = compute_participant_summary("P004", sessions)
        assert result["block_counts"]["stress"] == 2
        assert result["block_counts"]["food"] == 1


# ── 4. Pilot statistics tests ───────────────────────────────────────────────


class TestComputePilotStatistics:
    """Tests for compute_pilot_statistics()."""

    def test_basic_stats(self):
        sessions = [
            _complete_session("P001", "stress"),
            _complete_session("P002", "food"),
            _complete_session("P003", "sleep"),
        ]
        result = compute_pilot_statistics(sessions)
        assert result["n_participants"] == 3
        assert result["n_sessions"] == 3
        assert result["overall_completion"] == 1.0

    def test_block_distribution(self):
        sessions = [
            _complete_session("P001", "stress"),
            _complete_session("P001", "food"),
            _complete_session("P002", "sleep"),
        ]
        result = compute_pilot_statistics(sessions)
        assert result["block_distribution"]["stress"] == 1
        assert result["block_distribution"]["food"] == 1
        assert result["block_distribution"]["sleep"] == 1

    def test_stress_vs_balanced_separation(self):
        """When stress and balanced blocks have different quality scores, separation > 0."""
        # Need multiple samples per group for meaningful pooled std.
        s_stress1 = _complete_session("P001", "stress")
        s_stress1["dataQualityScore"] = 30
        s_stress2 = _complete_session("P003", "stress")
        s_stress2["dataQualityScore"] = 35
        s_food1 = _complete_session("P002", "food")
        s_food1["dataQualityScore"] = 80
        s_food2 = _complete_session("P004", "food")
        s_food2["dataQualityScore"] = 85
        result = compute_pilot_statistics([s_stress1, s_stress2, s_food1, s_food2])
        sep = result["stress_vs_balanced"]["separation_score"]
        assert sep is not None
        assert sep > 0

    def test_empty_sessions(self):
        result = compute_pilot_statistics([])
        assert result["n_participants"] == 0
        assert result["n_sessions"] == 0
        assert result["overall_completion"] == 0.0


# ── 5. Readiness report tests ───────────────────────────────────────────────


class TestGenerateReadinessReport:
    """Tests for generate_readiness_report() and report_to_dict()."""

    def _make_full_pilot_sessions(self) -> list:
        """Create sessions from 6 participants covering all block types with good quality.

        Stress blocks get lower quality scores than food/sleep to ensure
        the stress-vs-balanced separation check passes.
        """
        sessions = []
        for i in range(1, 7):
            code = f"P{i:03d}"
            for block in ("stress", "food", "sleep"):
                s = _complete_session(code, block)
                if block == "stress":
                    s["dataQualityScore"] = 40 + i  # 41-46
                else:
                    s["dataQualityScore"] = 80 + i  # 81-86
                sessions.append(s)
        return sessions

    def test_ready_when_all_checks_pass(self):
        sessions = self._make_full_pilot_sessions()
        report = generate_readiness_report(sessions)
        assert report.ready is True
        assert "READY" in report.summary
        assert all(c.passed for c in report.checks)

    def test_not_ready_few_participants(self):
        sessions = [_complete_session("P001", "stress")]
        report = generate_readiness_report(sessions, min_participants=5)
        assert report.ready is False
        names = [c.name for c in report.checks if not c.passed]
        assert "participant_count" in names

    def test_not_ready_low_completion(self):
        """All partial sessions should fail the completion check."""
        sessions = []
        for i in range(1, 7):
            code = f"P{i:03d}"
            for block in ("stress", "food", "sleep"):
                sessions.append(_partial_session(code, block))
        report = generate_readiness_report(sessions)
        assert report.ready is False
        names = [c.name for c in report.checks if not c.passed]
        assert "completion_rate" in names

    def test_report_to_dict_structure(self):
        sessions = self._make_full_pilot_sessions()
        report = generate_readiness_report(sessions)
        d = report_to_dict(report)
        assert isinstance(d, dict)
        assert "ready" in d
        assert "summary" in d
        assert "checks" in d
        assert "statistics" in d
        assert isinstance(d["checks"], list)
        for check in d["checks"]:
            assert "name" in check
            assert "passed" in check
            assert "detail" in check

    def test_report_statistics_included(self):
        sessions = self._make_full_pilot_sessions()
        report = generate_readiness_report(sessions)
        d = report_to_dict(report)
        stats = d["statistics"]
        assert "n_participants" in stats
        assert "n_sessions" in stats


# ── 6. API route smoke tests ────────────────────────────────────────────────


class TestPilotRoutes:
    """Smoke tests for pilot validation API routes via FastAPI TestClient."""

    @pytest.fixture(autouse=True)
    def _setup_client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.routes.pilot_validation import router

        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def test_status_endpoint(self):
        resp = self.client.get("/pilot/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "pilot_validation"

    def test_validate_session_endpoint(self):
        payload = {
            "session": {
                "preEegJson": {"data": 1},
                "postEegJson": {"data": 2},
                "eegFeaturesJson": {"data": 3},
                "surveyJson": {"data": 4},
            },
        }
        resp = self.client.post("/pilot/validate-session", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["completeness"]["is_complete"] is True
        assert data["sqi"] is None  # no eeg_data sent

    def test_validate_session_with_eeg(self):
        eeg = np.random.default_rng(42).normal(0, 20, (1, 512)).tolist()
        payload = {
            "session": {"preEegJson": {"data": 1}, "postEegJson": {"data": 2},
                        "eegFeaturesJson": {"data": 3}, "surveyJson": {"data": 4}},
            "eeg_data": eeg,
            "fs": 256.0,
        }
        resp = self.client.post("/pilot/validate-session", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["sqi"] is not None
        assert 0.0 <= data["sqi"]["sqi"] <= 1.0

    def test_participant_summary_endpoint(self):
        payload = {
            "participant_code": "P001",
            "sessions": [
                {
                    "participantCode": "P001",
                    "blockType": "stress",
                    "preEegJson": {"d": 1},
                    "postEegJson": {"d": 2},
                    "eegFeaturesJson": {"d": 3},
                    "surveyJson": {"d": 4},
                }
            ],
        }
        resp = self.client.post("/pilot/participant-summary", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["participant_code"] == "P001"
        assert data["total_sessions"] == 1

    def test_readiness_endpoint(self):
        resp = self.client.get("/pilot/readiness")
        assert resp.status_code == 200
        data = resp.json()
        assert "ready" in data
        assert "checks" in data

    def test_statistics_get_endpoint(self):
        resp = self.client.get("/pilot/statistics")
        assert resp.status_code == 200
        data = resp.json()
        assert "n_participants" in data
        assert "n_sessions" in data
