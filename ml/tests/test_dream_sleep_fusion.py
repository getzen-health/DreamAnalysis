"""Tests for dream-sleep architecture fusion model — issue #437."""
import pytest

from models.dream_sleep_fusion import (
    DreamEntry,
    DreamSleepCorrelation,
    DreamSleepProfile,
    SleepMetrics,
    analyze_dream_content,
    compute_dream_sleep_profile,
    correlate_dream_sleep,
    get_dream_sleep_fusion_model,
    predict_dream_type,
    profile_to_dict,
)


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture
def positive_dream():
    return DreamEntry(
        text="I was flying over beautiful mountains, feeling joyful and free. "
             "The sun was warm and everything was peaceful and wonderful.",
        date="2026-03-19",
    )


@pytest.fixture
def nightmare_dream():
    return DreamEntry(
        text="A terrifying monster was chasing me through a dark forest. "
             "I was scared and screaming, trapped with no escape. "
             "The shadow attacked and I felt helpless and terrified.",
        date="2026-03-18",
    )


@pytest.fixture
def lucid_dream():
    return DreamEntry(
        text="I realized I was dreaming and became fully aware. "
             "I controlled the dream and decided to fly over the city.",
        date="2026-03-17",
        lucidity_score=0.8,
    )


@pytest.fixture
def good_sleep():
    return SleepMetrics(
        rem_pct=0.24,
        deep_pct=0.22,
        efficiency=0.92,
        duration_h=8.0,
        awakenings=1,
    )


@pytest.fixture
def poor_sleep():
    return SleepMetrics(
        rem_pct=0.30,
        deep_pct=0.10,
        efficiency=0.70,
        duration_h=5.5,
        awakenings=5,
    )


@pytest.fixture
def model():
    return get_dream_sleep_fusion_model()


# -- analyze_dream_content tests --------------------------------------------

class TestAnalyzeDreamContent:

    def test_empty_text_returns_neutral(self):
        result = analyze_dream_content("")
        assert result["primary_emotion"] == "neutral"
        assert result["primary_theme"] == "unclassified"
        assert result["word_count"] == 0
        assert result["lucidity_score"] == 0.0

    def test_none_text_returns_neutral(self):
        result = analyze_dream_content(None)
        assert result["primary_emotion"] == "neutral"
        assert result["themes"] == []

    def test_positive_dream_detects_joy(self):
        result = analyze_dream_content(
            "I felt so happy and joyful, everything was wonderful and beautiful"
        )
        assert "joy" in result["emotions"]
        assert result["primary_emotion"] == "joy"

    def test_fear_dream_detects_fear(self):
        result = analyze_dream_content(
            "I was terrified and scared, there was a monster causing panic and dread"
        )
        assert "fear" in result["emotions"]
        assert result["emotions"]["fear"] > 0

    def test_theme_detection_flying(self):
        result = analyze_dream_content("I was flying and soaring above the clouds")
        assert "flying" in result["themes"]
        assert result["primary_theme"] == "flying"

    def test_theme_detection_pursuit(self):
        result = analyze_dream_content(
            "Someone was chasing me and I had to run and escape"
        )
        assert "pursuit" in result["themes"]

    def test_lucid_markers_detected(self):
        result = analyze_dream_content("I realized I was dreaming and took control")
        assert result["lucidity_score"] > 0.5

    def test_word_count_accurate(self):
        result = analyze_dream_content("one two three four five")
        assert result["word_count"] == 5

    def test_multiple_emotions_detected(self):
        result = analyze_dream_content(
            "I felt happy and grateful but then became scared and anxious"
        )
        assert len(result["emotions"]) >= 2

    def test_multiple_themes_detected(self):
        result = analyze_dream_content(
            "I was flying over the ocean and then fell into the water"
        )
        assert len(result["themes"]) >= 2


# -- correlate_dream_sleep tests --------------------------------------------

class TestCorrelateDreamSleep:

    def test_positive_dream_good_sleep(self, positive_dream, good_sleep):
        corr = correlate_dream_sleep(positive_dream, good_sleep)
        assert isinstance(corr, DreamSleepCorrelation)
        assert corr.nightmare_risk < 0.3
        assert corr.sleep_quality_score > 70
        assert len(corr.insights) > 0

    def test_nightmare_poor_sleep_high_risk(self, nightmare_dream, poor_sleep):
        corr = correlate_dream_sleep(nightmare_dream, poor_sleep)
        assert corr.nightmare_risk > 0.2
        assert corr.primary_emotion in ("fear", "anxiety", "anger")
        assert corr.primary_theme in ("pursuit", "nightmare")

    def test_lucid_dream_detected(self, lucid_dream, good_sleep):
        corr = correlate_dream_sleep(lucid_dream, good_sleep)
        assert corr.lucidity_score > 0.5

    def test_correlation_scores_present(self, positive_dream, good_sleep):
        corr = correlate_dream_sleep(positive_dream, good_sleep)
        assert "rem_emotion_correlation" in corr.correlation_scores
        assert "deep_sleep_dream_suppression" in corr.correlation_scores
        assert "efficiency_pleasantness" in corr.correlation_scores
        assert "awakening_nightmare_link" in corr.correlation_scores

    def test_sleep_quality_score_range(self, positive_dream, good_sleep):
        corr = correlate_dream_sleep(positive_dream, good_sleep)
        assert 0.0 <= corr.sleep_quality_score <= 100.0

    def test_nightmare_risk_bounded(self, nightmare_dream, poor_sleep):
        corr = correlate_dream_sleep(nightmare_dream, poor_sleep)
        assert 0.0 <= corr.nightmare_risk <= 1.0


# -- predict_dream_type tests -----------------------------------------------

class TestPredictDreamType:

    def test_basic_prediction_structure(self, good_sleep):
        result = predict_dream_type(good_sleep)
        assert "dream_probabilities" in result
        assert "predicted_type" in result
        assert "nightmare_risk" in result
        assert "recall_probability" in result
        assert "insights" in result

    def test_probabilities_sum_to_one(self, good_sleep):
        result = predict_dream_type(good_sleep)
        total = sum(result["dream_probabilities"].values())
        assert abs(total - 1.0) < 0.01

    def test_high_stress_increases_nightmare_risk(self):
        sleep = SleepMetrics(rem_pct=0.28, deep_pct=0.12, efficiency=0.72,
                             duration_h=6.0, awakenings=4)
        low_stress = predict_dream_type(sleep, recent_stress=0.1)
        high_stress = predict_dream_type(sleep, recent_stress=0.9)
        assert high_stress["nightmare_risk"] > low_stress["nightmare_risk"]

    def test_high_rem_increases_recall(self):
        high_rem = SleepMetrics(rem_pct=0.35)
        low_rem = SleepMetrics(rem_pct=0.10)
        high_result = predict_dream_type(high_rem)
        low_result = predict_dream_type(low_rem)
        assert high_result["recall_probability"] > low_result["recall_probability"]

    def test_good_sleep_low_stress_favors_positive(self):
        sleep = SleepMetrics(rem_pct=0.24, deep_pct=0.22, efficiency=0.92,
                             duration_h=8.0, awakenings=0)
        result = predict_dream_type(sleep, recent_stress=0.1)
        probs = result["dream_probabilities"]
        assert probs["vivid_positive"] > probs["nightmare"]

    def test_recent_emotions_affect_prediction(self):
        sleep = SleepMetrics(rem_pct=0.25, efficiency=0.80)
        neg_emo = {"fear": 0.8, "anxiety": 0.7}
        result = predict_dream_type(sleep, recent_emotions=neg_emo)
        assert result["nightmare_risk"] > 0.1

    def test_all_dream_types_have_positive_probability(self, good_sleep):
        result = predict_dream_type(good_sleep)
        for dtype, prob in result["dream_probabilities"].items():
            assert prob > 0, f"{dtype} has zero probability"


# -- compute_dream_sleep_profile tests --------------------------------------

class TestComputeProfile:

    def test_empty_inputs_return_empty_profile(self):
        profile = compute_dream_sleep_profile([], [])
        assert isinstance(profile, DreamSleepProfile)
        assert profile.total_entries == 0
        assert profile.dominant_emotion == "neutral"

    def test_single_entry_profile(self, positive_dream, good_sleep):
        profile = compute_dream_sleep_profile([positive_dream], [good_sleep])
        assert profile.total_entries == 1
        assert profile.avg_sleep_quality > 0

    def test_multi_entry_profile(self, positive_dream, nightmare_dream, good_sleep, poor_sleep):
        dreams = [positive_dream, nightmare_dream, positive_dream]
        sleeps = [good_sleep, poor_sleep, good_sleep]
        profile = compute_dream_sleep_profile(dreams, sleeps)
        assert profile.total_entries == 3
        assert len(profile.theme_frequency) > 0
        assert len(profile.emotion_frequency) > 0
        assert len(profile.insights) > 0

    def test_profile_to_dict_serializable(self, positive_dream, good_sleep):
        profile = compute_dream_sleep_profile([positive_dream], [good_sleep])
        d = profile_to_dict(profile)
        assert isinstance(d, dict)
        assert "total_entries" in d
        assert "dominant_emotion" in d
        assert "theme_sleep_correlations" in d
        assert "dream_type_predictions" in d

    def test_mismatched_lengths_uses_minimum(self):
        dreams = [
            DreamEntry(text="flying in the sky", date="2026-01-01"),
            DreamEntry(text="running from monster", date="2026-01-02"),
        ]
        sleeps = [SleepMetrics()]  # only 1
        profile = compute_dream_sleep_profile(dreams, sleeps)
        assert profile.total_entries == 1


# -- Singleton / facade tests -----------------------------------------------

class TestModelFacade:

    def test_singleton_returns_same_instance(self):
        m1 = get_dream_sleep_fusion_model()
        m2 = get_dream_sleep_fusion_model()
        assert m1 is m2

    def test_facade_analyze_content(self, model):
        result = model.analyze_content("happy joyful flying dream")
        assert result["primary_emotion"] == "joy"

    def test_facade_correlate(self, model, positive_dream, good_sleep):
        corr = model.correlate(positive_dream, good_sleep)
        assert isinstance(corr, DreamSleepCorrelation)

    def test_facade_predict_type(self, model, good_sleep):
        result = model.predict_type(good_sleep)
        assert "dream_probabilities" in result

    def test_facade_compute_profile(self, model, positive_dream, good_sleep):
        profile = model.compute_profile([positive_dream], [good_sleep])
        assert isinstance(profile, DreamSleepProfile)
