"""Tests for SHAP explainability in the emotion classifier.

Verifies that:
1. The `explanation` key is always present in prediction output
2. When SHAP is available + LGBM model loaded, explanation has entries
3. Each explanation entry has the correct schema (feature, impact, direction)
4. The heuristic path returns named feature contributions
5. SHAP failure never breaks prediction
6. Artifact-rejected epochs return explanation: []
"""

import numpy as np
import pytest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def emotion_model():
    from models.emotion_classifier import EmotionClassifier
    return EmotionClassifier()


@pytest.fixture
def multichannel_eeg():
    """4 channels x 4 seconds of synthetic EEG at 256 Hz."""
    np.random.seed(42)
    return np.random.randn(4, 1024) * 20


@pytest.fixture
def single_channel_eeg():
    """1D single-channel EEG."""
    np.random.seed(42)
    return np.random.randn(1024) * 20


@pytest.fixture
def artifact_eeg():
    """EEG with amplitude > 75 uV to trigger artifact rejection."""
    eeg = np.random.randn(4, 1024) * 200  # well above 75 uV threshold
    return eeg


class TestExplanationFieldPresent:
    """The `explanation` key must always be present in prediction output."""

    def test_multichannel_prediction_has_explanation(self, emotion_model, multichannel_eeg):
        result = emotion_model.predict(multichannel_eeg, 256.0)
        assert "explanation" in result, "explanation key missing from prediction result"

    def test_single_channel_prediction_has_explanation(self, emotion_model, single_channel_eeg):
        result = emotion_model.predict(single_channel_eeg, 256.0)
        assert "explanation" in result, "explanation key missing from single-channel result"

    def test_explanation_is_list(self, emotion_model, multichannel_eeg):
        result = emotion_model.predict(multichannel_eeg, 256.0)
        assert isinstance(result["explanation"], list)


class TestExplanationSchema:
    """Each entry in explanation must have feature, impact, direction."""

    def test_explanation_entries_have_correct_keys(self, emotion_model, multichannel_eeg):
        result = emotion_model.predict(multichannel_eeg, 256.0)
        for entry in result["explanation"]:
            assert "feature" in entry, f"Missing 'feature' key in {entry}"
            assert "impact" in entry, f"Missing 'impact' key in {entry}"
            assert "direction" in entry, f"Missing 'direction' key in {entry}"

    def test_direction_values(self, emotion_model, multichannel_eeg):
        result = emotion_model.predict(multichannel_eeg, 256.0)
        for entry in result["explanation"]:
            assert entry["direction"] in ("increases", "decreases"), \
                f"Invalid direction '{entry['direction']}'"

    def test_impact_is_float(self, emotion_model, multichannel_eeg):
        result = emotion_model.predict(multichannel_eeg, 256.0)
        for entry in result["explanation"]:
            assert isinstance(entry["impact"], float), \
                f"impact should be float, got {type(entry['impact'])}"

    def test_feature_is_string(self, emotion_model, multichannel_eeg):
        result = emotion_model.predict(multichannel_eeg, 256.0)
        for entry in result["explanation"]:
            assert isinstance(entry["feature"], str), \
                f"feature should be str, got {type(entry['feature'])}"

    def test_at_most_three_entries(self, emotion_model, multichannel_eeg):
        result = emotion_model.predict(multichannel_eeg, 256.0)
        assert len(result["explanation"]) <= 3, \
            f"Expected at most 3 explanation entries, got {len(result['explanation'])}"


class TestArtifactRejection:
    """Artifact-rejected epochs should return explanation: []."""

    def test_artifact_epoch_has_empty_explanation(self, artifact_eeg):
        from models.emotion_classifier import EmotionClassifier
        model = EmotionClassifier()
        # Disable higher-priority models so LGBM path with artifact rejection runs
        model._eegnet = None
        model._reve = None
        model._reve_foundation = None

        # First call — no prior EMA, so model runs even with artifact
        _ = model.predict(artifact_eeg, 256.0)
        # Second call — EMA exists, artifact triggers frozen EMA path
        result = model.predict(artifact_eeg, 256.0)
        assert result.get("artifact_detected", False) is True
        assert result["explanation"] == []


class TestHeuristicExplanation:
    """When falling through to _predict_features, explanation uses named features."""

    def test_heuristic_path_has_named_features(self):
        from models.emotion_classifier import EmotionClassifier
        # Create a model with no LGBM loaded (forces heuristic path)
        model = EmotionClassifier()
        model.mega_lgbm_model = None
        model.lgbm_muse_model = None
        model.sklearn_model = None
        model.onnx_session = None
        model._tsception = None
        model._eegnet = None
        model._reve = None
        model._reve_foundation = None
        model._benchmark_accuracy = 0.0

        eeg = np.random.randn(4, 1024) * 20
        result = model.predict(eeg, 256.0)
        assert "explanation" in result
        assert len(result["explanation"]) > 0, \
            "Heuristic path should produce non-empty explanation"
        # Check features are named EEG terms, not generic "feature_N"
        for entry in result["explanation"]:
            assert not entry["feature"].startswith("feature_"), \
                f"Heuristic feature should be named, got '{entry['feature']}'"


class TestSHAPGracefulDegradation:
    """SHAP failure must never break prediction."""

    def test_prediction_works_without_shap(self, multichannel_eeg):
        from models.emotion_classifier import EmotionClassifier
        model = EmotionClassifier()

        # Mock shap import to raise ImportError
        with patch.dict('sys.modules', {'shap': None}):
            result = model.predict(multichannel_eeg, 256.0)
            assert "emotion" in result
            assert "explanation" in result
            assert isinstance(result["explanation"], list)

    def test_prediction_works_when_shap_raises(self, multichannel_eeg):
        from models.emotion_classifier import EmotionClassifier
        model = EmotionClassifier()

        # Force SHAP explainer to raise on shap_values
        model._shap_explainer_mega = "broken"
        model._shap_explainer_muse = "broken"

        result = model.predict(multichannel_eeg, 256.0)
        assert "emotion" in result
        assert "explanation" in result


class TestSHAPWithLGBM:
    """When LGBM path runs and shap is installed, explanation should be non-empty."""

    def test_mega_lgbm_produces_shap_explanation(self):
        """If shap is installed and mega LGBM is loaded, explanation has entries."""
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        from models.emotion_classifier import EmotionClassifier
        model = EmotionClassifier()
        if model.mega_lgbm_model is None:
            pytest.skip("mega LGBM model not available")

        # Disable higher-priority models to force mega LGBM path
        model._eegnet = None
        model._reve = None
        model._reve_foundation = None
        model.lgbm_muse_model = None

        eeg = np.random.randn(4, 1024) * 20
        result = model.predict(eeg, 256.0)
        assert result.get("model_type") == "mega-lgbm-pca"
        assert len(result["explanation"]) == 3, \
            f"Expected 3 explanation entries from SHAP, got {len(result['explanation'])}"
        # Verify feature names are from _FEATURE_NAMES_85
        from models.emotion_classifier import _FEATURE_NAMES_85
        for entry in result["explanation"]:
            assert entry["feature"] in _FEATURE_NAMES_85, \
                f"Feature '{entry['feature']}' not in _FEATURE_NAMES_85"

    def test_muse_lgbm_produces_shap_explanation(self):
        """If shap is installed and Muse LGBM is loaded, explanation has entries."""
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        from models.emotion_classifier import EmotionClassifier
        model = EmotionClassifier()
        if model.lgbm_muse_model is None:
            pytest.skip("Muse-native LGBM model not available")

        # Disable higher-priority models to force Muse LGBM path
        model._eegnet = None
        model._reve = None
        model._reve_foundation = None
        model.mega_lgbm_model = None

        eeg = np.random.randn(4, 1024) * 20
        result = model.predict(eeg, 256.0)
        assert result.get("model_type") == "lgbm-muse"
        assert len(result["explanation"]) == 3, \
            f"Expected 3 explanation entries from SHAP, got {len(result['explanation'])}"


class TestFeatureNames85:
    """The _FEATURE_NAMES_85 constant must be correct."""

    def test_feature_names_length(self):
        from models.emotion_classifier import _FEATURE_NAMES_85
        assert len(_FEATURE_NAMES_85) == 85

    def test_feature_names_format(self):
        from models.emotion_classifier import _FEATURE_NAMES_85
        # First 80 should be band_channel_stat format
        for name in _FEATURE_NAMES_85[:80]:
            parts = name.split("_")
            assert len(parts) >= 3, f"Expected band_channel_stat format, got '{name}'"

        # Last 5 should be DASM_band
        for name in _FEATURE_NAMES_85[80:]:
            assert name.startswith("DASM_"), f"Expected DASM_ prefix, got '{name}'"

    def test_all_bands_present(self):
        from models.emotion_classifier import _FEATURE_NAMES_85
        bands = {"delta", "theta", "alpha", "beta", "gamma"}
        found_bands = set()
        for name in _FEATURE_NAMES_85[:80]:
            found_bands.add(name.split("_")[0])
        assert found_bands == bands

    def test_all_channels_present(self):
        from models.emotion_classifier import _FEATURE_NAMES_85
        channels = {"TP9", "AF7", "AF8", "TP10"}
        found_channels = set()
        for name in _FEATURE_NAMES_85[:80]:
            found_channels.add(name.split("_")[1])
        assert found_channels == channels
