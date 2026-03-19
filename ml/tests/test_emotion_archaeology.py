"""Tests for emotion archaeology — digital artifact emotional reconstruction (issue #453).

Covers: text sentiment analysis, music history mapping, calendar event mapping,
photo metadata analysis, timeline reconstruction, report generation,
serialization, emotional diversity, trajectory inference, and API routes.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from models.emotion_archaeology import (
    ArchaeologyReport,
    CalendarEmotionSignal,
    EmotionalPeriod,
    MusicEmotionSignal,
    PhotoEmotionSignal,
    TextSentiment,
    _compute_emotional_diversity,
    _infer_trajectory,
    analyze_calendar_events,
    analyze_music_history,
    analyze_photo_metadata,
    analyze_text_sentiment,
    generate_archaeology_report,
    reconstruct_emotional_timeline,
    report_to_dict,
)
from api.routes.emotion_archaeology import router

# ---------------------------------------------------------------------------
# Test client setup
# ---------------------------------------------------------------------------

app = FastAPI()
app.include_router(router)
client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Text sentiment analysis
# ---------------------------------------------------------------------------


class TestTextSentiment:
    def test_positive_text(self):
        """Positive words should produce positive net sentiment."""
        result = analyze_text_sentiment("I am so happy and grateful for this wonderful day")
        assert isinstance(result, TextSentiment)
        assert result.net_sentiment > 0
        assert result.positive_score > 0
        assert result.word_count > 0

    def test_negative_text(self):
        """Negative words should produce negative net sentiment."""
        result = analyze_text_sentiment("I feel sad and lonely, everything is broken")
        assert result.net_sentiment < 0
        assert result.negative_score > 0

    def test_neutral_text(self):
        """Text without sentiment words should be neutral."""
        result = analyze_text_sentiment("The cat sat on the mat near the window")
        assert result.net_sentiment == 0.0
        assert result.positive_score == 0.0
        assert result.negative_score == 0.0

    def test_empty_text(self):
        """Empty text should return zero scores."""
        result = analyze_text_sentiment("")
        assert result.net_sentiment == 0.0
        assert result.word_count == 0
        assert result.themes == []

    def test_theme_detection(self):
        """Themes should be detected from keywords."""
        result = analyze_text_sentiment("I am grateful and thankful for all the love in my life")
        assert "gratitude" in result.themes
        assert "love" in result.themes

    def test_text_preview_truncation(self):
        """Long text should be truncated in preview."""
        long_text = "word " * 200
        result = analyze_text_sentiment(long_text)
        assert len(result.text_preview) <= 83  # 80 + "..."

    def test_sentiment_clamped(self):
        """Net sentiment should always be in [-1, 1]."""
        result = analyze_text_sentiment("happy happy happy joy joy joy love love")
        assert -1.0 <= result.net_sentiment <= 1.0


# ---------------------------------------------------------------------------
# 2. Music history analysis
# ---------------------------------------------------------------------------


class TestMusicHistory:
    def test_genre_mapping(self):
        """Known genres should map to expected valence/energy."""
        entries = [{"genre": "dance"}, {"genre": "blues"}]
        signals = analyze_music_history(entries)
        assert len(signals) == 2
        # Dance = positive high energy
        assert signals[0].valence > 0.3
        assert signals[0].energy > 0.7
        # Blues = negative lower energy
        assert signals[1].valence < 0

    def test_custom_valence_energy(self):
        """Provided valence/energy should override genre lookup."""
        entries = [{"genre": "pop", "valence": 0.1, "energy": 0.2}]
        signals = analyze_music_history(entries)
        # valence 0.1 maps to 0.1*2 - 1 = -0.8
        assert signals[0].valence < 0
        assert signals[0].energy == 0.2

    def test_unknown_genre(self):
        """Unknown genres should get default values."""
        entries = [{"genre": "martian_synth"}]
        signals = analyze_music_history(entries)
        assert len(signals) == 1
        assert isinstance(signals[0], MusicEmotionSignal)

    def test_mood_classification(self):
        """Mood labels should reflect valence/energy quadrant."""
        # High valence, high energy -> energetic_positive
        entries = [{"genre": "dance"}]
        signals = analyze_music_history(entries)
        assert signals[0].inferred_mood == "energetic_positive"

    def test_empty_history(self):
        """Empty music history should return empty list."""
        signals = analyze_music_history([])
        assert signals == []


# ---------------------------------------------------------------------------
# 3. Calendar events analysis
# ---------------------------------------------------------------------------


class TestCalendarEvents:
    def test_known_event_types(self):
        """Known event types should map to expected emotions."""
        events = [
            {"event_type": "vacation", "name": "Beach trip"},
            {"event_type": "deadline", "name": "Project due"},
        ]
        signals = analyze_calendar_events(events)
        assert len(signals) == 2
        assert signals[0].valence > 0.5  # vacation = positive
        assert signals[0].dominant_emotion == "joy"
        assert signals[1].valence < 0  # deadline = negative
        assert signals[1].dominant_emotion == "anxiety"

    def test_unknown_event_type(self):
        """Unknown event types should get default neutral values."""
        events = [{"event_type": "alien_abduction", "name": "???"}]
        signals = analyze_calendar_events(events)
        assert signals[0].dominant_emotion == "neutral"

    def test_empty_events(self):
        """Empty event list should return empty signals."""
        signals = analyze_calendar_events([])
        assert signals == []


# ---------------------------------------------------------------------------
# 4. Photo metadata analysis
# ---------------------------------------------------------------------------


class TestPhotoMetadata:
    def test_known_photo_context(self):
        """Wedding photos should map to joy."""
        photos = [{"event_type": "wedding", "people_count": 5}]
        signals = analyze_photo_metadata(photos)
        assert len(signals) == 1
        assert signals[0].dominant_emotion == "joy"
        assert signals[0].valence > 0.5

    def test_group_photo_boost(self):
        """Photos with many people should get a valence boost."""
        solo = analyze_photo_metadata([{"event_type": "default", "people_count": 0}])
        group = analyze_photo_metadata([{"event_type": "default", "people_count": 5}])
        assert group[0].valence > solo[0].valence

    def test_photo_with_location(self):
        """Location should be preserved in the signal."""
        photos = [{"event_type": "travel", "people_count": 2, "location": "Paris"}]
        signals = analyze_photo_metadata(photos)
        assert signals[0].location == "Paris"


# ---------------------------------------------------------------------------
# 5. Emotional diversity
# ---------------------------------------------------------------------------


class TestEmotionalDiversity:
    def test_single_emotion_zero_diversity(self):
        """A single emotion = zero diversity."""
        assert _compute_emotional_diversity({"joy": 1.0}) == 0.0

    def test_uniform_distribution_max_diversity(self):
        """Uniform distribution should give diversity close to 1.0."""
        dist = {"joy": 0.25, "grief": 0.25, "anxiety": 0.25, "calm": 0.25}
        diversity = _compute_emotional_diversity(dist)
        assert diversity >= 0.95

    def test_skewed_distribution_low_diversity(self):
        """Heavily skewed distribution should give lower diversity."""
        dist = {"joy": 0.9, "grief": 0.05, "anxiety": 0.05}
        diversity = _compute_emotional_diversity(dist)
        assert diversity < 0.7


# ---------------------------------------------------------------------------
# 6. Trajectory inference
# ---------------------------------------------------------------------------


class TestTrajectoryInference:
    def _make_period(self, label: str, valence: float) -> EmotionalPeriod:
        return EmotionalPeriod(
            period_label=label,
            dominant_emotion="neutral",
            average_valence=valence,
            average_arousal=0.4,
            emotion_distribution={"neutral": 1.0},
            emotional_diversity=0.0,
            artifact_count=1,
            themes=[],
        )

    def test_improving_trajectory(self):
        """Consistently rising valence = improving."""
        periods = [
            self._make_period("Q1", -0.5),
            self._make_period("Q2", -0.1),
            self._make_period("Q3", 0.3),
        ]
        assert _infer_trajectory(periods) == "improving"

    def test_declining_trajectory(self):
        """Consistently falling valence = declining."""
        periods = [
            self._make_period("Q1", 0.5),
            self._make_period("Q2", 0.1),
            self._make_period("Q3", -0.3),
        ]
        assert _infer_trajectory(periods) == "declining"

    def test_stable_trajectory(self):
        """Flat valence = stable."""
        periods = [
            self._make_period("Q1", 0.2),
            self._make_period("Q2", 0.2),
            self._make_period("Q3", 0.2),
        ]
        assert _infer_trajectory(periods) == "stable"

    def test_single_period_stable(self):
        """Single period defaults to stable."""
        periods = [self._make_period("overall", 0.3)]
        assert _infer_trajectory(periods) == "stable"


# ---------------------------------------------------------------------------
# 7. Timeline reconstruction
# ---------------------------------------------------------------------------


class TestTimelineReconstruction:
    def test_basic_reconstruction(self):
        """Mix of artifacts should produce a valid timeline."""
        artifacts = {
            "texts": ["I am so happy today!", "Feeling anxious about tomorrow."],
            "music": [{"genre": "pop"}, {"genre": "blues"}],
            "calendar": [{"event_type": "vacation", "name": "Trip"}],
            "photos": [{"event_type": "beach", "people_count": 3}],
        }
        timeline = reconstruct_emotional_timeline(artifacts)
        assert len(timeline) == 1
        assert timeline[0].period_label == "overall"
        assert timeline[0].artifact_count > 0

    def test_multi_period_reconstruction(self):
        """Multiple periods should split artifacts."""
        artifacts = {
            "texts": ["happy"] * 6,
            "music": [{"genre": "pop"}] * 6,
        }
        timeline = reconstruct_emotional_timeline(artifacts, period_labels=["H1", "H2"])
        assert len(timeline) == 2
        assert timeline[0].period_label == "H1"
        assert timeline[1].period_label == "H2"

    def test_empty_artifacts(self):
        """Empty artifacts should return neutral timeline."""
        timeline = reconstruct_emotional_timeline({})
        assert len(timeline) == 1
        assert timeline[0].dominant_emotion == "neutral"
        assert timeline[0].artifact_count == 0


# ---------------------------------------------------------------------------
# 8. Report generation
# ---------------------------------------------------------------------------


class TestReportGeneration:
    def test_full_report(self):
        """Full report should have all required fields."""
        artifacts = {
            "texts": ["I am grateful and happy"],
            "music": [{"genre": "jazz"}],
            "calendar": [{"event_type": "meeting", "name": "Standup"}],
            "photos": [{"event_type": "nature", "people_count": 0}],
        }
        report = generate_archaeology_report(
            user_id="test-user",
            artifacts=artifacts,
            period_start="2024-01",
            period_end="2024-12",
        )
        assert isinstance(report, ArchaeologyReport)
        assert report.user_id == "test-user"
        assert report.period_start == "2024-01"
        assert report.period_end == "2024-12"
        assert report.total_artifacts_analyzed == 4
        assert len(report.timeline) >= 1
        assert report.narrative_summary != ""
        assert report.emotional_trajectory in ("improving", "declining", "stable", "volatile")
        assert len(report.recommendations) >= 1

    def test_report_serialization(self):
        """report_to_dict should produce a JSON-serializable dict."""
        artifacts = {
            "texts": ["I feel anxious and worried about the future"],
        }
        report = generate_archaeology_report(
            user_id="u123",
            artifacts=artifacts,
        )
        d = report_to_dict(report)
        assert isinstance(d, dict)
        assert d["user_id"] == "u123"
        assert "timeline" in d
        assert "narrative_summary" in d
        assert "recommendations" in d
        assert isinstance(d["timeline"], list)


# ---------------------------------------------------------------------------
# 9. API route tests
# ---------------------------------------------------------------------------


class TestArchaeologyAPI:
    def test_status_endpoint(self):
        """GET /archaeology/status should return 200 with availability info."""
        resp = client.get("/archaeology/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert "texts" in data["supported_artifacts"]

    def test_analyze_text_endpoint(self):
        """POST /archaeology/analyze-text should return sentiment analysis."""
        resp = client.post("/archaeology/analyze-text", json={
            "texts": ["I love this wonderful day", "I feel sad and lonely"]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["entries_analyzed"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["net_sentiment"] > 0
        assert data["results"][1]["net_sentiment"] < 0

    def test_analyze_text_empty_list_rejected(self):
        """POST /archaeology/analyze-text with empty list should return 422."""
        resp = client.post("/archaeology/analyze-text", json={"texts": []})
        assert resp.status_code == 422

    def test_reconstruct_endpoint(self):
        """POST /archaeology/reconstruct should return a full report."""
        resp = client.post("/archaeology/reconstruct", json={
            "user_id": "test-user",
            "texts": ["I am happy and grateful"],
            "music": [{"genre": "jazz"}],
            "calendar": [{"event_type": "vacation", "name": "Beach trip"}],
            "photos": [{"event_type": "nature", "people_count": 2}],
            "period_start": "2024-01",
            "period_end": "2024-12",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "test-user"
        assert data["total_artifacts_analyzed"] == 4
        assert len(data["timeline"]) >= 1
        assert "narrative_summary" in data
        assert "recommendations" in data

    def test_reconstruct_minimal_input(self):
        """POST /archaeology/reconstruct with only user_id should succeed."""
        resp = client.post("/archaeology/reconstruct", json={
            "user_id": "minimal-user",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "minimal-user"
        assert data["total_artifacts_analyzed"] == 0

    def test_reconstruct_with_period_labels(self):
        """POST /archaeology/reconstruct with period labels splits timeline."""
        resp = client.post("/archaeology/reconstruct", json={
            "user_id": "periods-user",
            "texts": ["happy joy love"] * 4 + ["sad lonely broken"] * 4,
            "period_labels": ["H1", "H2"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["timeline"]) == 2
        assert data["timeline"][0]["period_label"] == "H1"
        assert data["timeline"][1]["period_label"] == "H2"

    def test_reconstruct_missing_user_id_rejected(self):
        """POST /archaeology/reconstruct without user_id should return 422."""
        resp = client.post("/archaeology/reconstruct", json={
            "texts": ["hello"],
        })
        assert resp.status_code == 422
