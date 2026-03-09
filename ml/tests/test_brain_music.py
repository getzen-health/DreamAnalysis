"""Tests for BrainMusicGenerator — EEG-to-music mapping."""
import sys
import os

# Ensure ml/ is on the path when running from the tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from audio.brain_music import BrainMusicGenerator, SCALES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gen():
    return BrainMusicGenerator()


# ---------------------------------------------------------------------------
# generate_notes — structure tests
# ---------------------------------------------------------------------------

def test_generate_notes_returns_list(gen):
    notes = gen.generate_notes(0.5, 0.5, 0.2, n_notes=4)
    assert isinstance(notes, list)


def test_generate_notes_correct_count(gen):
    for n in (1, 4, 8, 16):
        notes = gen.generate_notes(0.0, 0.5, 0.2, n_notes=n)
        assert len(notes) == n, f"Expected {n} notes, got {len(notes)}"


def test_generate_notes_dict_keys(gen):
    notes = gen.generate_notes(0.0, 0.5, 0.2, n_notes=4)
    for note in notes:
        assert "note" in note
        assert "velocity" in note
        assert "duration" in note


def test_generate_notes_midi_range(gen):
    """All MIDI note numbers must be in the 60–84 (C4–C6) range."""
    for valence in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for arousal in (0.0, 0.5, 1.0):
            notes = gen.generate_notes(valence, arousal, 0.2, n_notes=8)
            for note in notes:
                assert 60 <= note["note"] <= 84, (
                    f"MIDI note {note['note']} out of range for valence={valence}, arousal={arousal}"
                )


def test_generate_notes_velocity_range(gen):
    """Velocity must stay in 30–100."""
    for valence in (-1.0, 0.0, 1.0):
        for arousal in (0.0, 0.5, 1.0):
            for alpha in (0.0, 0.5, 1.0):
                notes = gen.generate_notes(valence, arousal, alpha, n_notes=6)
                for note in notes:
                    assert 30 <= note["velocity"] <= 100, (
                        f"Velocity {note['velocity']} out of range"
                    )


def test_generate_notes_duration_positive(gen):
    notes = gen.generate_notes(0.0, 0.5, 0.2, n_notes=4)
    for note in notes:
        assert note["duration"] > 0.0


def test_n_notes_param_clamped(gen):
    """n_notes is clamped to [1, 16]."""
    assert len(gen.generate_notes(0.0, 0.5, 0.2, n_notes=0)) == 1
    assert len(gen.generate_notes(0.0, 0.5, 0.2, n_notes=100)) == 16


# ---------------------------------------------------------------------------
# map_emotion_to_params
# ---------------------------------------------------------------------------

def test_map_emotion_to_params_returns_required_keys(gen):
    params = gen.map_emotion_to_params(0.5, 0.5, 0.2)
    assert "scale" in params
    assert "tempo_bpm" in params
    assert "note_density" in params
    assert "velocity" in params


def test_high_arousal_higher_tempo(gen):
    low_arousal = gen.map_emotion_to_params(0.0, 0.0, 0.2)
    high_arousal = gen.map_emotion_to_params(0.0, 1.0, 0.2)
    assert high_arousal["tempo_bpm"] > low_arousal["tempo_bpm"]


def test_tempo_in_range(gen):
    for arousal in (0.0, 0.25, 0.5, 0.75, 1.0):
        params = gen.map_emotion_to_params(0.0, arousal, 0.2)
        assert 40.0 <= params["tempo_bpm"] <= 180.0


def test_note_density_in_range(gen):
    for arousal in (0.0, 0.5, 1.0):
        params = gen.map_emotion_to_params(0.0, arousal, 0.2)
        assert 1 <= params["note_density"] <= 8


def test_positive_valence_major_scale(gen):
    """Valence > 0.3 should select major_pentatonic."""
    params = gen.map_emotion_to_params(0.5, 0.5, 0.2)
    assert params["scale"] == "major_pentatonic"


def test_strongly_positive_valence_major_pentatonic(gen):
    params = gen.map_emotion_to_params(1.0, 0.5, 0.2)
    assert params["scale"] == "major_pentatonic"


def test_slightly_positive_valence_minor_pentatonic(gen):
    """0.0 < valence <= 0.3 → minor_pentatonic."""
    params = gen.map_emotion_to_params(0.15, 0.5, 0.2)
    assert params["scale"] == "minor_pentatonic"


def test_mildly_negative_valence_dorian(gen):
    """-0.3 < valence <= 0.0 → dorian."""
    params = gen.map_emotion_to_params(-0.2, 0.5, 0.2)
    assert params["scale"] == "dorian"


def test_negative_valence_chromatic(gen):
    """valence <= -0.3 → chromatic."""
    params = gen.map_emotion_to_params(-0.8, 0.5, 0.2)
    assert params["scale"] == "chromatic"


def test_strongly_negative_valence_chromatic(gen):
    params = gen.map_emotion_to_params(-1.0, 0.5, 0.2)
    assert params["scale"] == "chromatic"


def test_alpha_power_reduces_velocity(gen):
    """High alpha_power should produce lower velocity than low alpha_power."""
    low_alpha = gen.map_emotion_to_params(0.0, 0.5, 0.0)
    high_alpha = gen.map_emotion_to_params(0.0, 0.5, 1.0)
    assert low_alpha["velocity"] > high_alpha["velocity"]


def test_scale_in_known_scales(gen):
    """The selected scale must always be one of the defined scales."""
    for valence in (-1.0, -0.5, -0.1, 0.1, 0.5, 1.0):
        params = gen.map_emotion_to_params(valence, 0.5, 0.2)
        assert params["scale"] in SCALES


# ---------------------------------------------------------------------------
# get_state_description
# ---------------------------------------------------------------------------

def test_get_state_description_returns_string(gen):
    desc = gen.get_state_description(0.5, 0.5)
    assert isinstance(desc, str)


def test_get_state_description_non_empty(gen):
    for valence in (-1.0, 0.0, 1.0):
        for arousal in (0.0, 0.5, 1.0):
            desc = gen.get_state_description(valence, arousal)
            assert len(desc) > 0


def test_get_state_description_positive_high_arousal(gen):
    desc = gen.get_state_description(0.8, 0.9)
    # Should contain a positive and high-energy word
    assert any(word in desc for word in ("joyful", "positive", "energetic", "highly energetic"))


def test_get_state_description_negative_low_arousal(gen):
    desc = gen.get_state_description(-0.8, 0.1)
    assert any(word in desc for word in ("distressed", "tense", "neutral"))


def test_get_state_description_contains_and(gen):
    """Descriptions should join two dimensions with 'and'."""
    desc = gen.get_state_description(0.0, 0.5)
    assert " and " in desc


# ---------------------------------------------------------------------------
# SCALES dict sanity
# ---------------------------------------------------------------------------

def test_scales_dict_has_expected_keys():
    assert "major_pentatonic" in SCALES
    assert "minor_pentatonic" in SCALES
    assert "dorian" in SCALES
    assert "chromatic" in SCALES


def test_scale_intervals_start_at_zero():
    for name, intervals in SCALES.items():
        assert intervals[0] == 0, f"Scale {name} should start at 0 (root)"


def test_scale_intervals_are_sorted():
    for name, intervals in SCALES.items():
        assert intervals == sorted(intervals), f"Scale {name} intervals not sorted"


def test_chromatic_has_12_notes():
    assert len(SCALES["chromatic"]) == 12


def test_major_pentatonic_has_5_notes():
    assert len(SCALES["major_pentatonic"]) == 5


def test_minor_pentatonic_has_5_notes():
    assert len(SCALES["minor_pentatonic"]) == 5


def test_dorian_has_7_notes():
    assert len(SCALES["dorian"]) == 7


# ---------------------------------------------------------------------------
# Edge / boundary cases
# ---------------------------------------------------------------------------

def test_extreme_inputs_do_not_raise(gen):
    """No exception on boundary inputs."""
    gen.generate_notes(-1.0, 0.0, 0.0, n_notes=1)
    gen.generate_notes(1.0, 1.0, 1.0, n_notes=16)
    gen.generate_notes(0.0, 0.5, 100.0, n_notes=4)  # very high alpha_power


def test_zero_arousal_tempo_at_minimum(gen):
    params = gen.map_emotion_to_params(0.0, 0.0, 0.2)
    assert params["tempo_bpm"] == pytest.approx(40.0, abs=0.1)


def test_full_arousal_tempo_at_maximum(gen):
    params = gen.map_emotion_to_params(0.0, 1.0, 0.2)
    assert params["tempo_bpm"] == pytest.approx(180.0, abs=0.1)
