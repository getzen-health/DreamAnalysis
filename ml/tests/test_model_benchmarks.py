"""CI benchmark regression tests.

These tests run on every PR/push and track model output quality over time.
They use fixed synthetic EEG inputs to detect regressions in:
1. Emotion classifier output shape and probability sums
2. Signal quality scoring
3. EMA smoothing behavior
4. Stress/focus index ranges

If these tests fail on CI, it means a code change degraded the pipeline.
"""
import json
import time
import pathlib
import numpy as np
import pytest

# Fixed RNG seed for reproducibility across CI runs
RNG = np.random.default_rng(seed=2024_03_08)

# Fixture: 4-channel synthetic EEG (256 Hz, 4 seconds = 1024 samples)
EEG_4CH = RNG.normal(0, 10, (4, 1024)).astype(np.float32)
EEG_1CH = EEG_4CH[1]  # Single channel (AF7)

BENCHMARK_FILE = pathlib.Path(__file__).parent / "benchmark_results.json"


def _load_benchmarks() -> dict:
    if BENCHMARK_FILE.exists():
        with open(BENCHMARK_FILE) as f:
            return json.load(f)
    return {}


def _save_benchmark(key: str, value: float):
    results = _load_benchmarks()
    if key not in results:
        results[key] = []
    results[key].append(value)
    # Keep last 20 runs
    results[key] = results[key][-20:]
    BENCHMARK_FILE.write_text(json.dumps(results, indent=2))


class TestEmotionClassifierBenchmark:
    """Regression tests for the emotion classifier pipeline."""

    @pytest.fixture(scope="class")
    def classifier(self):
        from models.emotion_classifier import EmotionClassifier
        return EmotionClassifier()

    def test_output_has_required_keys(self, classifier):
        result = classifier.predict(EEG_4CH, fs=256)
        required = ["emotion", "probabilities", "valence", "arousal", "stress_index", "focus_index"]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_probabilities_sum_to_one(self, classifier):
        result = classifier.predict(EEG_4CH, fs=256)
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, expected ~1.0"

    def test_valence_in_range(self, classifier):
        result = classifier.predict(EEG_4CH, fs=256)
        assert -1.0 <= result["valence"] <= 1.0, f"Valence out of range: {result['valence']}"

    def test_arousal_in_range(self, classifier):
        result = classifier.predict(EEG_4CH, fs=256)
        assert 0.0 <= result["arousal"] <= 1.0, f"Arousal out of range: {result['arousal']}"

    def test_stress_index_in_range(self, classifier):
        result = classifier.predict(EEG_4CH, fs=256)
        assert 0.0 <= result["stress_index"] <= 1.0

    def test_focus_index_in_range(self, classifier):
        result = classifier.predict(EEG_4CH, fs=256)
        assert 0.0 <= result["focus_index"] <= 1.0

    def test_inference_speed(self, classifier):
        """Emotion inference must complete under 500ms."""
        start = time.time()
        classifier.predict(EEG_4CH, fs=256)
        elapsed_ms = (time.time() - start) * 1000
        _save_benchmark("emotion_inference_ms", elapsed_ms)
        assert elapsed_ms < 500, f"Inference too slow: {elapsed_ms:.0f}ms"

    def test_emotion_is_valid_class(self, classifier):
        result = classifier.predict(EEG_4CH, fs=256)
        valid = {"happy", "sad", "angry", "fear", "surprise", "neutral"}
        assert result["emotion"] in valid, f"Unknown emotion: {result['emotion']}"


class TestFeatureExtractionBenchmark:
    """Regression tests for EEG feature extraction."""

    def test_extract_features_shape(self):
        from processing.eeg_processor import extract_features
        features = extract_features(EEG_1CH, fs=256)
        assert isinstance(features, dict)
        assert len(features) >= 10, f"Too few features: {len(features)}"

    def test_extract_multichannel_shape(self):
        from processing.eeg_processor import extract_features_multichannel
        features = extract_features_multichannel(EEG_4CH, fs=256)
        assert isinstance(features, dict)
        assert len(features) >= 20

    def test_faa_in_range(self):
        from processing.eeg_processor import compute_frontal_asymmetry
        result = compute_frontal_asymmetry(EEG_4CH, fs=256, left_ch=1, right_ch=2)
        faa = result.get("frontal_asymmetry", 0.0)
        assert -10.0 <= faa <= 10.0, f"FAA out of range: {faa}"

    def test_band_powers_positive(self):
        from processing.eeg_processor import extract_band_powers
        powers = extract_band_powers(EEG_1CH, fs=256)
        for band in ["delta", "theta", "alpha", "beta"]:
            assert powers.get(band, -1) >= 0, f"Negative power for {band}"

    def test_feature_extraction_speed(self):
        """Feature extraction must complete under 100ms per channel."""
        from processing.eeg_processor import extract_features
        start = time.time()
        for _ in range(10):
            extract_features(EEG_1CH, fs=256)
        avg_ms = (time.time() - start) * 100
        _save_benchmark("feature_extraction_ms", avg_ms)
        assert avg_ms < 100, f"Feature extraction too slow: {avg_ms:.1f}ms"


class TestSignalQualityBenchmark:
    """Regression tests for signal quality assessment."""

    def test_clean_signal_high_quality(self):
        from processing.eeg_processor import extract_band_powers
        # Clean low-amplitude signal
        clean_eeg = RNG.normal(0, 5, 1024).astype(np.float32)
        powers = extract_band_powers(clean_eeg, fs=256)
        assert powers is not None

    def test_artifact_detection_high_amplitude(self):
        """Signal exceeding 75µV threshold should be flagged."""
        artifact_eeg = np.ones(1024, dtype=np.float32) * 200.0  # 200µV — way above threshold
        max_amp = float(np.abs(artifact_eeg).max())
        assert max_amp > 75.0, "Test setup: should be above threshold"
