"""Tests for all 6 ML models."""

import numpy as np


class TestSleepStaging:
    def setup_method(self):
        from models.sleep_staging import SleepStagingModel
        self.model = SleepStagingModel()

    def test_predict_returns_valid_stage(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert result["stage"] in ["Wake", "N1", "N2", "N3", "REM"]

    def test_predict_has_confidence(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert 0 <= result["confidence"] <= 1

    def test_predict_has_probabilities(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert "probabilities" in result

    def test_short_signal(self, fs):
        short = np.random.randn(64) * 20
        result = self.model.predict(short, fs)
        assert "stage" in result


class TestEmotionClassifier:
    def setup_method(self):
        from models.emotion_classifier import EmotionClassifier
        self.model = EmotionClassifier()

    def test_predict_returns_valid_emotion(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert result["emotion"] in ["happy", "sad", "angry", "fearful", "relaxed", "focused"]

    def test_predict_has_valence_arousal(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert -1 <= result["valence"] <= 1
        assert -1 <= result["arousal"] <= 1

    def test_predict_has_band_powers(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert "band_powers" in result
        bands = result["band_powers"]
        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
            assert band in bands

    def test_predict_has_stress_index(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert "stress_index" in result
        assert 0 <= result["stress_index"] <= 100


class TestDreamDetector:
    def setup_method(self):
        from models.dream_detector import DreamDetector
        self.model = DreamDetector()

    def test_predict_returns_boolean(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert isinstance(result["is_dreaming"], bool)

    def test_predict_has_probability(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert 0 <= result["probability"] <= 1

    def test_predict_has_rem_likelihood(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert "rem_likelihood" in result


class TestFlowStateDetector:
    def setup_method(self):
        from models.flow_state_detector import FlowStateDetector
        self.model = FlowStateDetector()

    def test_predict_returns_valid_state(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert result["state"] in ["no_flow", "micro_flow", "flow", "deep_flow"]

    def test_predict_has_flow_score(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert 0 <= result["flow_score"] <= 1

    def test_predict_has_components(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert "components" in result
        comp = result["components"]
        for key in ["absorption", "effortlessness", "focus_quality"]:
            assert key in comp


class TestCreativityDetector:
    def setup_method(self):
        from models.creativity_detector import CreativityDetector
        self.model = CreativityDetector()

    def test_predict_returns_valid_state(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert result["state"] in ["analytical", "transitional", "creative", "insight"]

    def test_predict_has_creativity_score(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert 0 <= result["creativity_score"] <= 1


class TestMemoryEncodingPredictor:
    def setup_method(self):
        from models.creativity_detector import MemoryEncodingPredictor
        self.model = MemoryEncodingPredictor()

    def test_predict_returns_valid_state(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert result["state"] in ["poor_encoding", "weak_encoding", "active_encoding", "deep_encoding"]

    def test_predict_has_probability(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert 0 <= result["will_remember_probability"] <= 1
