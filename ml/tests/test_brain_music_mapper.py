"""Tests for BrainMusicMapper — EEG emotion-to-musical-parameter mapping."""
import pytest

from models.brain_music import BrainMusicMapper


@pytest.fixture
def mapper():
    return BrainMusicMapper()


def test_positive_valence_major_scale(mapper):
    """Positive valence should produce major scale."""
    result = mapper.map({"valence": 0.5, "arousal": 0.5})
    assert result["scale"] == "major"


def test_negative_valence_minor_scale(mapper):
    """Negative valence should produce minor scale."""
    result = mapper.map({"valence": -0.5, "arousal": 0.5})
    assert result["scale"] == "minor"


def test_high_arousal_fast_tempo(mapper):
    """High arousal should produce fast tempo."""
    result = mapper.map({"arousal": 0.9, "valence": 0})
    assert result["tempo_bpm"] >= 120


def test_low_arousal_slow_tempo(mapper):
    """Low arousal should produce slow tempo."""
    result = mapper.map({"arousal": 0.1, "valence": 0})
    assert result["tempo_bpm"] <= 80


def test_tempo_range(mapper):
    """Tempo should be between 60 and 140 BPM."""
    for arousal in [0, 0.25, 0.5, 0.75, 1.0]:
        result = mapper.map({"arousal": arousal})
        assert 60 <= result["tempo_bpm"] <= 140


def test_alpha_dynamics(mapper):
    """Higher alpha should produce higher dynamics."""
    low = mapper.map({"alpha_power": 0.1, "arousal": 0.5})
    high = mapper.map({"alpha_power": 0.8, "arousal": 0.5})
    assert high["dynamics"] > low["dynamics"]


def test_dynamics_range(mapper):
    """Dynamics should be between 0.1 and 1.0."""
    for alpha in [0, 0.1, 0.3, 0.5, 0.8, 2.0]:
        result = mapper.map({"alpha_power": alpha})
        assert 0.1 <= result["dynamics"] <= 1.0


def test_theta_complexity_simple(mapper):
    """Low theta should produce simple harmonics."""
    result = mapper.map({"theta_power": 0.05})
    assert result["complexity"] == "simple"


def test_theta_complexity_rich(mapper):
    """High theta should produce rich harmonics."""
    result = mapper.map({"theta_power": 0.8})
    assert result["complexity"] == "rich"


def test_faa_fallback_when_no_valence(mapper):
    """When valence is 0, FAA should drive scale selection."""
    result = mapper.map({"frontal_asymmetry": 0.5, "valence": 0.0})
    assert result["scale"] == "major"


def test_mood_color_format(mapper):
    """Mood color should be valid hex."""
    result = mapper.map({"valence": 0.3, "arousal": 0.6})
    assert result["mood_color"].startswith("#")
    assert len(result["mood_color"]) == 7


def test_mood_color_positive_more_green(mapper):
    """Positive valence should have more green component."""
    pos = mapper.map({"valence": 0.8, "arousal": 0.5})
    neg = mapper.map({"valence": -0.8, "arousal": 0.5})
    pos_g = int(pos["mood_color"][3:5], 16)
    neg_g = int(neg["mood_color"][3:5], 16)
    assert pos_g > neg_g


def test_history_tracking(mapper):
    """History should accumulate mappings."""
    for _ in range(5):
        mapper.map({"valence": 0.3, "arousal": 0.5})
    assert len(mapper.get_history()) == 5


def test_history_cap(mapper):
    """History should cap at 100."""
    for _ in range(110):
        mapper.map({"valence": 0.3, "arousal": 0.5})
    assert len(mapper.get_history()) == 100


def test_average_mood(mapper):
    """Average mood should summarize recent history."""
    for _ in range(5):
        mapper.map({"valence": 0.5, "arousal": 0.7, "alpha_power": 0.3})
    avg = mapper.get_average_mood()
    assert "avg_tempo" in avg
    assert "dominant_scale" in avg
    assert avg["dominant_scale"] == "major"


def test_average_mood_empty(mapper):
    """Average mood with no history returns defaults."""
    avg = mapper.get_average_mood()
    assert avg["key"] == "C"


def test_output_fields(mapper):
    """All expected fields present in output."""
    result = mapper.map({"valence": 0.3, "arousal": 0.6, "alpha_power": 0.4, "theta_power": 0.3})
    expected = {"key", "scale", "tempo_bpm", "dynamics", "complexity", "intervals",
                "effective_valence", "arousal", "mood_color"}
    assert expected.issubset(set(result.keys()))


def test_empty_input(mapper):
    """Empty input dict should use defaults."""
    result = mapper.map({})
    assert "tempo_bpm" in result
    assert "scale" in result


def test_intervals_are_list(mapper):
    """Intervals should be a list of integers."""
    result = mapper.map({"theta_power": 0.4})
    assert isinstance(result["intervals"], list)
    assert all(isinstance(i, int) for i in result["intervals"])


def test_extreme_values(mapper):
    """Extreme input values should not crash."""
    result = mapper.map({"valence": -10, "arousal": 50, "alpha_power": -1, "theta_power": 100})
    assert result["tempo_bpm"] == 140
    assert result["dynamics"] >= 0.1
