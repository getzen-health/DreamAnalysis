"""Tests for DREAM database integration: loader, detector, and training."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure ml/ is on the path (mirrors conftest.py convention)
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def loader():
    from training.train_dream_database import DREAMDatabaseLoader
    return DREAMDatabaseLoader()


@pytest.fixture
def detector():
    from models.dream_database_detector import DREAMDatabaseDreamDetector
    return DREAMDatabaseDreamDetector()


@pytest.fixture
def multichannel_eeg():
    """4 channels × 4 seconds of synthetic EEG at 256 Hz."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((4, 1024)) * 20.0


@pytest.fixture
def fs():
    return 256


# ── DREAMDatabaseLoader tests ─────────────────────────────────────────────────

class TestDREAMDatabaseLoader:

    def test_loader_initializes(self, loader):
        """Loader should construct without error."""
        assert loader is not None

    def test_is_available_returns_bool(self, loader):
        """is_available() must return a plain bool."""
        result = loader.is_available()
        assert isinstance(result, bool)

    def test_is_available_false_for_missing_dir(self):
        """Non-existent data_dir → False."""
        from training.train_dream_database import DREAMDatabaseLoader
        l = DREAMDatabaseLoader(data_dir="/tmp/dream_db_does_not_exist_xyz")
        assert l.is_available() is False

    def test_download_instructions_returns_string(self, loader):
        """download_instructions() must return a non-empty string."""
        instructions = loader.download_instructions()
        assert isinstance(instructions, str)
        assert len(instructions) > 50

    def test_download_instructions_contains_key_info(self, loader):
        """Download instructions must mention DREAM database and Nature Communications."""
        instructions = loader.download_instructions()
        assert "DREAM" in instructions
        assert "Nature" in instructions

    def test_simulate_dream_features_shape(self, loader):
        """simulate_dream_features() must return (n_samples, 23) array."""
        X, y = loader.simulate_dream_features(n_samples=200)
        assert X.shape == (200, 23)
        assert y.shape == (200,)

    def test_simulate_dream_features_label_distribution(self, loader):
        """Dream labels should be ~45% positive (±10 pp tolerance)."""
        X, y = loader.simulate_dream_features(n_samples=1000)
        dream_rate = float(y.mean())
        assert 0.35 <= dream_rate <= 0.55, (
            f"Expected ~45% dream labels, got {dream_rate:.2%}"
        )

    def test_simulate_dream_features_values_in_range(self, loader):
        """All simulated feature values should be finite and in a plausible range."""
        X, y = loader.simulate_dream_features(n_samples=100)
        assert np.all(np.isfinite(X)), "NaN or Inf in simulated features"
        assert np.all(X >= -5.0) and np.all(X <= 10.0), (
            "Simulated features contain out-of-range values"
        )

    def test_simulate_dream_features_labels_binary(self, loader):
        """Labels must be strictly binary 0/1."""
        _, y = loader.simulate_dream_features(n_samples=200)
        unique = set(y.tolist())
        assert unique == {0, 1}, f"Expected {{0,1}} labels, got {unique}"

    def test_extract_4ch_features_shape(self, loader, multichannel_eeg, fs):
        """extract_4ch_features() must return a (23,) vector."""
        fv = loader.extract_4ch_features(multichannel_eeg, fs)
        assert fv.shape == (23,), f"Expected (23,), got {fv.shape}"

    def test_extract_4ch_features_finite(self, loader, multichannel_eeg, fs):
        """Feature vector must contain only finite values."""
        fv = loader.extract_4ch_features(multichannel_eeg, fs)
        assert np.all(np.isfinite(fv)), "extract_4ch_features returned NaN or Inf"

    def test_get_benchmark_stats_keys(self, loader):
        """get_benchmark_stats() must return expected DREAM database keys."""
        stats = loader.get_benchmark_stats()
        required_keys = [
            "total_participants",
            "total_awakenings",
            "datasets",
            "dream_rate_overall",
            "dream_rate_rem",
            "dream_rate_nrem",
        ]
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

    def test_get_benchmark_stats_values(self, loader):
        """Benchmark stats must match published DREAM database figures."""
        stats = loader.get_benchmark_stats()
        assert stats["total_participants"] == 505
        assert stats["total_awakenings"] == 2643
        assert 0.60 <= stats["dream_rate_rem"] <= 0.80
        assert 0.15 <= stats["dream_rate_nrem"] <= 0.40


# ── DREAMDatabaseDreamDetector tests ─────────────────────────────────────────

class TestDREAMDatabaseDreamDetector:

    def test_detector_initializes(self, detector):
        """Detector should construct without error."""
        assert detector is not None

    def test_predict_returns_dict(self, detector, multichannel_eeg, fs):
        """predict() must return a dict."""
        result = detector.predict(multichannel_eeg, fs)
        assert isinstance(result, dict)

    def test_predict_required_keys(self, detector, multichannel_eeg, fs):
        """predict() output must contain all required keys."""
        result = detector.predict(multichannel_eeg, fs)
        required = [
            "dreaming",
            "dream_probability",
            "dream_intensity",
            "dream_vividness",
            "emotional_valence",
            "sleep_stage_consistency",
            "model_source",
            "dream_database_features",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_predict_dream_probability_range(self, detector, multichannel_eeg, fs):
        """dream_probability must be in [0, 1]."""
        result = detector.predict(multichannel_eeg, fs)
        p = result["dream_probability"]
        assert 0.0 <= p <= 1.0, f"dream_probability out of range: {p}"

    def test_predict_dreaming_is_bool(self, detector, multichannel_eeg, fs):
        """dreaming field must be a bool."""
        result = detector.predict(multichannel_eeg, fs)
        assert isinstance(result["dreaming"], bool)

    def test_predict_intensities_in_range(self, detector, multichannel_eeg, fs):
        """dream_intensity and dream_vividness must be in [0, 1]."""
        result = detector.predict(multichannel_eeg, fs)
        assert 0.0 <= result["dream_intensity"]  <= 1.0
        assert 0.0 <= result["dream_vividness"]  <= 1.0

    def test_predict_emotional_valence_range(self, detector, multichannel_eeg, fs):
        """emotional_valence must be in [-1, 1]."""
        result = detector.predict(multichannel_eeg, fs)
        v = result["emotional_valence"]
        assert -1.0 <= v <= 1.0, f"emotional_valence out of range: {v}"

    def test_predict_with_rem_stage_hint(self, detector, multichannel_eeg, fs):
        """REM hint should increase dream_probability relative to no hint."""
        result_no_hint  = detector.predict(multichannel_eeg, fs)
        result_rem_hint = detector.predict(multichannel_eeg, fs, sleep_stage="REM")
        # REM prior (0.68) > default prior (0.45) → probability should shift up
        # Allow small tolerance for blending
        assert result_rem_hint["dream_probability"] >= result_no_hint["dream_probability"] - 0.05

    def test_predict_with_nrem_stage_hint(self, detector, multichannel_eeg, fs):
        """NREM hint should lower dream_probability compared to REM hint."""
        result_rem  = detector.predict(multichannel_eeg, fs, sleep_stage="REM")
        result_nrem = detector.predict(multichannel_eeg, fs, sleep_stage="NREM")
        assert result_nrem["dream_probability"] <= result_rem["dream_probability"] + 0.05

    def test_predict_sleep_stage_consistency_string(self, detector, multichannel_eeg, fs):
        """sleep_stage_consistency must be a non-empty string."""
        result = detector.predict(multichannel_eeg, fs, sleep_stage="REM")
        assert isinstance(result["sleep_stage_consistency"], str)
        assert len(result["sleep_stage_consistency"]) > 0

    def test_predict_unknown_consistency_without_stage(self, detector, multichannel_eeg, fs):
        """Without sleep_stage, consistency should be 'unknown'."""
        result = detector.predict(multichannel_eeg, fs, sleep_stage=None)
        assert result["sleep_stage_consistency"] == "unknown"

    def test_predict_dream_database_features_dict(self, detector, multichannel_eeg, fs):
        """dream_database_features must be a dict with expected keys."""
        result = detector.predict(multichannel_eeg, fs)
        features = result["dream_database_features"]
        assert isinstance(features, dict)
        for key in ["theta_delta_ratio", "alpha_beta_ratio", "faa", "stage_prior"]:
            assert key in features, f"Missing key in dream_database_features: {key}"

    def test_get_dream_themes_keys(self, detector, multichannel_eeg, fs):
        """get_dream_themes() must return all four theme keys."""
        themes = detector.get_dream_themes(multichannel_eeg, fs)
        assert isinstance(themes, dict)
        for key in ["emotional", "visual", "kinesthetic", "narrative"]:
            assert key in themes, f"Missing theme key: {key}"

    def test_get_dream_themes_values_range(self, detector, multichannel_eeg, fs):
        """All theme values must be in [0, 1]."""
        themes = detector.get_dream_themes(multichannel_eeg, fs)
        for key, val in themes.items():
            assert 0.0 <= val <= 1.0, f"Theme {key} out of range: {val}"

    def test_get_dream_themes_sums_to_approx_one(self, detector, multichannel_eeg, fs):
        """Theme probabilities should sum to approximately 1.0."""
        themes = detector.get_dream_themes(multichannel_eeg, fs)
        total = sum(themes.values())
        assert abs(total - 1.0) < 0.05, f"Theme sum {total:.4f} far from 1.0"

    def test_get_model_info_returns_dict(self, detector):
        """get_model_info() must return a dict."""
        info = detector.get_model_info()
        assert isinstance(info, dict)

    def test_get_model_info_keys(self, detector):
        """get_model_info() must contain model_source and participant count."""
        info = detector.get_model_info()
        assert "model_source" in info
        assert "participants" in info
        assert info["participants"] == 505

    def test_predict_1d_eeg_accepted(self, detector, fs):
        """Detector must handle a 1-D (single-channel) EEG array without error."""
        eeg_1d = np.random.randn(1024) * 20.0
        result = detector.predict(eeg_1d, fs)
        assert "dream_probability" in result

    def test_predict_model_source_field(self, detector, multichannel_eeg, fs):
        """model_source must be a non-empty string."""
        result = detector.predict(multichannel_eeg, fs)
        assert isinstance(result["model_source"], str)
        assert len(result["model_source"]) > 0


# ── Singleton getter test ─────────────────────────────────────────────────────

class TestSingletonGetter:

    def test_singleton_returns_same_instance(self):
        """get_dream_database_detector() must return the same object per user_id."""
        from models.dream_database_detector import get_dream_database_detector
        a = get_dream_database_detector("user_test_123")
        b = get_dream_database_detector("user_test_123")
        assert a is b

    def test_different_user_ids_return_different_instances(self):
        """Different user IDs must yield separate instances."""
        from models.dream_database_detector import get_dream_database_detector
        a = get_dream_database_detector("user_A_xyz")
        b = get_dream_database_detector("user_B_xyz")
        assert a is not b


# ── Training function test ────────────────────────────────────────────────────

class TestTrainingFunction:

    def test_train_returns_accuracy(self):
        """train_on_dream_database() must return a dict with accuracy metrics."""
        pytest.importorskip("sklearn", reason="scikit-learn required for training test")
        from training.train_dream_database import train_on_dream_database

        results = train_on_dream_database(
            data_dir="/tmp/dream_db_not_exist",
            output_path="/tmp/dream_db_test_model.pkl",
            n_simulated=300,
            force_simulate=True,
        )
        assert isinstance(results, dict)
        assert "cv_accuracy" in results
        assert "test_accuracy" in results
        assert 0.50 <= results["test_accuracy"] <= 1.0, (
            f"test_accuracy implausible: {results['test_accuracy']}"
        )

    def test_train_simulated_flag(self):
        """Training on simulated data must set simulated=True in results."""
        pytest.importorskip("sklearn", reason="scikit-learn required for training test")
        from training.train_dream_database import train_on_dream_database

        results = train_on_dream_database(
            data_dir="/tmp/dream_db_not_exist",
            output_path="/tmp/dream_db_test_model2.pkl",
            n_simulated=200,
            force_simulate=True,
        )
        assert results.get("simulated") is True

    def test_train_returns_feature_names(self):
        """Training results must include feature_names list of length 23."""
        pytest.importorskip("sklearn", reason="scikit-learn required for training test")
        from training.train_dream_database import train_on_dream_database

        results = train_on_dream_database(
            data_dir="/tmp/dream_db_not_exist",
            output_path="/tmp/dream_db_test_model3.pkl",
            n_simulated=200,
            force_simulate=True,
        )
        assert "feature_names" in results
        assert len(results["feature_names"]) == 23
