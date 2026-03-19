"""Tests for the emotional genome model and API routes (issue #444)."""

from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sample(ts, valence=0.0, arousal=0.5, stress=0.3, energy=0.5,
                 social=0.0, novelty=0.0, threat=0.0):
    from models.emotional_genome import EmotionSample
    return EmotionSample(
        timestamp=ts,
        valence=valence,
        arousal=arousal,
        stress=stress,
        energy=energy,
        social_context=social,
        novelty_context=novelty,
        threat_context=threat,
    )


@pytest.fixture
def basic_samples():
    """20 samples spanning a simulated 20-day period."""
    base_ts = 1700000000.0
    day = 86400.0
    samples = []
    rng = np.random.RandomState(42)
    for i in range(20):
        samples.append(_make_sample(
            ts=base_ts + i * day,
            valence=float(np.clip(0.2 + 0.3 * rng.randn(), -1, 1)),
            arousal=float(np.clip(0.5 + 0.15 * rng.randn(), 0, 1)),
            stress=float(np.clip(0.3 + 0.1 * rng.randn(), 0, 1)),
            energy=float(np.clip(0.6 + 0.1 * rng.randn(), 0, 1)),
            social=float(np.clip(0.1 * rng.randn(), -1, 1)),
            novelty=float(np.clip(0.3 + 0.2 * rng.randn(), 0, 1)),
            threat=float(np.clip(0.1 + 0.1 * rng.randn(), 0, 1)),
        ))
    return samples


@pytest.fixture
def biphasic_samples():
    """40 samples with a clear shift in the middle (two emotional eras)."""
    base_ts = 1700000000.0
    day = 86400.0
    samples = []
    # Phase 1: positive, low stress
    for i in range(20):
        samples.append(_make_sample(
            ts=base_ts + i * day,
            valence=0.6,
            arousal=0.4,
            stress=0.2,
            energy=0.7,
        ))
    # Phase 2: negative, high stress
    for i in range(20):
        samples.append(_make_sample(
            ts=base_ts + (20 + i) * day,
            valence=-0.4,
            arousal=0.7,
            stress=0.8,
            energy=0.3,
        ))
    return samples


@pytest.fixture
def few_samples():
    """Only 3 samples -- below minimum for many analyses."""
    base_ts = 1700000000.0
    return [
        _make_sample(base_ts, valence=0.5),
        _make_sample(base_ts + 100, valence=0.3),
        _make_sample(base_ts + 200, valence=0.1),
    ]


# ---------------------------------------------------------------------------
# EmotionalTraits dataclass
# ---------------------------------------------------------------------------

class TestEmotionalTraits:
    def test_default_values(self):
        from models.emotional_genome import EmotionalTraits
        t = EmotionalTraits()
        assert t.emotional_intensity == 0.5
        assert t.emotional_stability == 0.5
        assert t.positive_bias == 0.5

    def test_to_array_shape(self):
        from models.emotional_genome import EmotionalTraits
        t = EmotionalTraits(emotional_intensity=0.8, positive_bias=0.9)
        arr = t.to_array()
        assert arr.shape == (7,)
        assert arr[0] == pytest.approx(0.8)
        assert arr[6] == pytest.approx(0.9)

    def test_from_array_clips_bounds(self):
        from models.emotional_genome import EmotionalTraits
        arr = np.array([1.5, -0.3, 0.5, 0.5, 0.5, 0.5, 2.0])
        t = EmotionalTraits.from_array(arr)
        assert t.emotional_intensity == 1.0
        assert t.emotional_stability == 0.0
        assert t.positive_bias == 1.0

    def test_roundtrip_array(self):
        from models.emotional_genome import EmotionalTraits
        original = EmotionalTraits(
            emotional_intensity=0.7, emotional_stability=0.3,
            recovery_speed=0.8, social_sensitivity=0.6,
            novelty_reactivity=0.4, threat_sensitivity=0.9,
            positive_bias=0.2,
        )
        restored = EmotionalTraits.from_array(original.to_array())
        assert restored.emotional_intensity == pytest.approx(original.emotional_intensity)
        assert restored.positive_bias == pytest.approx(original.positive_bias)


# ---------------------------------------------------------------------------
# compute_emotional_traits
# ---------------------------------------------------------------------------

class TestComputeEmotionalTraits:
    def test_returns_traits_from_basic_data(self, basic_samples):
        from models.emotional_genome import compute_emotional_traits
        traits = compute_emotional_traits(basic_samples)
        # All traits should be in [0, 1]
        arr = traits.to_array()
        assert np.all(arr >= 0.0)
        assert np.all(arr <= 1.0)

    def test_too_few_samples_returns_defaults(self, few_samples):
        from models.emotional_genome import compute_emotional_traits, EmotionalTraits
        traits = compute_emotional_traits(few_samples)
        defaults = EmotionalTraits()
        assert traits.emotional_intensity == defaults.emotional_intensity
        assert traits.emotional_stability == defaults.emotional_stability

    def test_high_positive_data_gives_high_positive_bias(self):
        from models.emotional_genome import compute_emotional_traits
        base_ts = 1700000000.0
        samples = [
            _make_sample(base_ts + i * 100, valence=0.8, arousal=0.5)
            for i in range(10)
        ]
        traits = compute_emotional_traits(samples)
        assert traits.positive_bias > 0.8

    def test_stable_data_gives_high_stability(self):
        from models.emotional_genome import compute_emotional_traits
        base_ts = 1700000000.0
        # All samples have nearly identical valence
        samples = [
            _make_sample(base_ts + i * 100, valence=0.3, arousal=0.5)
            for i in range(10)
        ]
        traits = compute_emotional_traits(samples)
        assert traits.emotional_stability > 0.8

    def test_volatile_data_gives_low_stability(self):
        from models.emotional_genome import compute_emotional_traits
        base_ts = 1700000000.0
        # Alternating extreme valences
        samples = [
            _make_sample(base_ts + i * 100,
                         valence=0.9 if i % 2 == 0 else -0.9)
            for i in range(10)
        ]
        traits = compute_emotional_traits(samples)
        assert traits.emotional_stability < 0.5

    def test_threat_sensitivity_with_correlated_data(self):
        from models.emotional_genome import compute_emotional_traits
        base_ts = 1700000000.0
        # Threat goes up -> stress goes up (perfect correlation)
        samples = [
            _make_sample(base_ts + i * 100,
                         stress=i / 10.0,
                         threat=i / 10.0)
            for i in range(10)
        ]
        traits = compute_emotional_traits(samples)
        assert traits.threat_sensitivity > 0.7


# ---------------------------------------------------------------------------
# generate_emotional_fingerprint
# ---------------------------------------------------------------------------

class TestGenerateEmotionalFingerprint:
    def test_fingerprint_length(self, basic_samples):
        from models.emotional_genome import (
            compute_emotional_traits, generate_emotional_fingerprint,
            _FINGERPRINT_DIM,
        )
        traits = compute_emotional_traits(basic_samples)
        fp = generate_emotional_fingerprint(traits, basic_samples)
        assert len(fp) == _FINGERPRINT_DIM

    def test_fingerprint_is_normalized(self, basic_samples):
        from models.emotional_genome import (
            compute_emotional_traits, generate_emotional_fingerprint,
        )
        traits = compute_emotional_traits(basic_samples)
        fp = generate_emotional_fingerprint(traits, basic_samples)
        norm = float(np.linalg.norm(fp))
        assert norm == pytest.approx(1.0, abs=0.01)

    def test_fingerprint_with_few_samples(self, few_samples):
        from models.emotional_genome import (
            EmotionalTraits, generate_emotional_fingerprint,
            _FINGERPRINT_DIM,
        )
        fp = generate_emotional_fingerprint(EmotionalTraits(), few_samples)
        assert len(fp) == _FINGERPRINT_DIM

    def test_different_users_have_different_fingerprints(self):
        from models.emotional_genome import (
            compute_emotional_traits, generate_emotional_fingerprint,
        )
        base_ts = 1700000000.0
        # User A: positive, calm
        samples_a = [
            _make_sample(base_ts + i * 100, valence=0.7, arousal=0.3, stress=0.1)
            for i in range(15)
        ]
        # User B: negative, stressed
        samples_b = [
            _make_sample(base_ts + i * 100, valence=-0.5, arousal=0.8, stress=0.9)
            for i in range(15)
        ]
        traits_a = compute_emotional_traits(samples_a)
        traits_b = compute_emotional_traits(samples_b)
        fp_a = np.array(generate_emotional_fingerprint(traits_a, samples_a))
        fp_b = np.array(generate_emotional_fingerprint(traits_b, samples_b))
        # Cosine distance should be significant
        cosine_sim = float(np.dot(fp_a, fp_b))
        assert cosine_sim < 0.95  # Not identical


# ---------------------------------------------------------------------------
# detect_life_chapters
# ---------------------------------------------------------------------------

class TestDetectLifeChapters:
    def test_biphasic_data_finds_multiple_chapters(self, biphasic_samples):
        from models.emotional_genome import detect_life_chapters
        chapters = detect_life_chapters(biphasic_samples, max_chapters=5)
        assert len(chapters) >= 2

    def test_single_phase_returns_one_chapter(self):
        from models.emotional_genome import detect_life_chapters
        base_ts = 1700000000.0
        samples = [
            _make_sample(base_ts + i * 100, valence=0.5, arousal=0.5)
            for i in range(20)
        ]
        chapters = detect_life_chapters(samples, max_chapters=5)
        assert len(chapters) >= 1

    def test_few_samples_returns_single_chapter(self, few_samples):
        from models.emotional_genome import detect_life_chapters
        chapters = detect_life_chapters(few_samples, max_chapters=5)
        assert len(chapters) == 1

    def test_empty_samples_returns_empty(self):
        from models.emotional_genome import detect_life_chapters
        chapters = detect_life_chapters([], max_chapters=5)
        assert chapters == []

    def test_chapters_cover_all_samples(self, biphasic_samples):
        from models.emotional_genome import detect_life_chapters
        chapters = detect_life_chapters(biphasic_samples, max_chapters=5)
        total = sum(ch.sample_count for ch in chapters)
        assert total == len(biphasic_samples)

    def test_chapters_are_chronologically_ordered(self, biphasic_samples):
        from models.emotional_genome import detect_life_chapters
        chapters = detect_life_chapters(biphasic_samples, max_chapters=5)
        for i in range(len(chapters) - 1):
            assert chapters[i].start_timestamp <= chapters[i + 1].start_timestamp

    def test_chapter_has_label(self, basic_samples):
        from models.emotional_genome import detect_life_chapters
        chapters = detect_life_chapters(basic_samples, max_chapters=3)
        for ch in chapters:
            assert isinstance(ch.label, str)
            assert len(ch.label) > 0


# ---------------------------------------------------------------------------
# track_emotional_evolution
# ---------------------------------------------------------------------------

class TestTrackEmotionalEvolution:
    def test_returns_snapshots_for_enough_data(self, basic_samples):
        from models.emotional_genome import track_emotional_evolution
        snapshots = track_emotional_evolution(basic_samples, window_size=10, step_size=5)
        assert len(snapshots) >= 1

    def test_few_samples_returns_empty(self, few_samples):
        from models.emotional_genome import track_emotional_evolution
        snapshots = track_emotional_evolution(few_samples)
        assert len(snapshots) == 0

    def test_snapshots_have_valid_traits(self, basic_samples):
        from models.emotional_genome import track_emotional_evolution
        snapshots = track_emotional_evolution(basic_samples, window_size=10, step_size=5)
        for snap in snapshots:
            arr = snap.traits.to_array()
            assert np.all(arr >= 0.0)
            assert np.all(arr <= 1.0)

    def test_evolution_detects_trait_shift(self, biphasic_samples):
        from models.emotional_genome import track_emotional_evolution
        snapshots = track_emotional_evolution(
            biphasic_samples, window_size=10, step_size=5,
        )
        assert len(snapshots) >= 2
        # First snapshot should have higher positive_bias than last
        assert snapshots[0].traits.positive_bias > snapshots[-1].traits.positive_bias


# ---------------------------------------------------------------------------
# compute_genome_profile + profile_to_dict
# ---------------------------------------------------------------------------

class TestGenomeProfile:
    def test_full_profile_computation(self, basic_samples):
        from models.emotional_genome import compute_genome_profile
        profile = compute_genome_profile("test-user", basic_samples)
        assert profile.user_id == "test-user"
        assert profile.sample_count == 20
        assert len(profile.fingerprint) == 16
        assert len(profile.chapters) >= 1

    def test_profile_to_dict_is_serializable(self, basic_samples):
        import json
        from models.emotional_genome import compute_genome_profile, profile_to_dict
        profile = compute_genome_profile("dict-user", basic_samples)
        d = profile_to_dict(profile)
        # Must be JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0
        assert d["user_id"] == "dict-user"
        assert "traits" in d
        assert "fingerprint" in d
        assert "chapters" in d
        assert "evolution" in d
        assert d["sample_count"] == 20

    def test_profile_with_minimal_data(self):
        from models.emotional_genome import compute_genome_profile, profile_to_dict
        base_ts = 1700000000.0
        samples = [
            _make_sample(base_ts + i * 100, valence=0.5)
            for i in range(6)
        ]
        profile = compute_genome_profile("minimal-user", samples)
        d = profile_to_dict(profile)
        assert d["sample_count"] == 6
        assert isinstance(d["fingerprint"], list)


# ---------------------------------------------------------------------------
# API route tests (FastAPI TestClient)
# ---------------------------------------------------------------------------

class TestAPIRoutes:
    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.routes.emotional_genome import router, _profiles
        _profiles.clear()
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_status_endpoint(self, client):
        resp = client.get("/emotional-genome/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "available"
        assert data["cached_profiles"] == 0

    def test_profile_endpoint(self, client):
        base_ts = 1700000000.0
        payload = {
            "user_id": "api-user",
            "samples": [
                {
                    "timestamp": base_ts + i * 86400,
                    "valence": 0.3 + 0.02 * i,
                    "arousal": 0.5,
                    "stress": 0.3,
                    "energy": 0.6,
                    "social_context": 0.1,
                    "novelty_context": 0.2,
                    "threat_context": 0.1,
                }
                for i in range(15)
            ],
        }
        resp = client.post("/emotional-genome/profile", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["profile"]["user_id"] == "api-user"
        assert len(data["profile"]["fingerprint"]) == 16
        assert data["profile"]["sample_count"] == 15

    def test_profile_caches_result(self, client):
        base_ts = 1700000000.0
        payload = {
            "user_id": "cache-user",
            "samples": [
                {"timestamp": base_ts + i * 100, "valence": 0.5}
                for i in range(10)
            ],
        }
        client.post("/emotional-genome/profile", json=payload)
        resp = client.get("/emotional-genome/status")
        data = resp.json()
        assert data["cached_profiles"] == 1
        assert "cache-user" in data["profile_ids"]

    def test_profile_empty_user_id_rejected(self, client):
        payload = {
            "user_id": "",
            "samples": [{"timestamp": 1700000000.0}],
        }
        resp = client.post("/emotional-genome/profile", json=payload)
        assert resp.status_code == 422

    def test_profile_no_samples_rejected(self, client):
        payload = {
            "user_id": "bad-user",
            "samples": [],
        }
        resp = client.post("/emotional-genome/profile", json=payload)
        assert resp.status_code == 422
