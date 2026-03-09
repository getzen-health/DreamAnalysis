"""Tests for PPG feature extraction and heart-brain coupling.

Synthetic data used throughout:
  - PPG: 1 Hz sinusoid + 2nd harmonic (mimics 60 BPM systolic peaks)
  - EEG: 4-channel Gaussian noise, 10 µV RMS
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.ppg_features import PPGFeatureExtractor
from processing.heart_brain import compute_hep, compute_heart_brain_coupling

# ── Shared synthetic fixtures ─────────────────────────────────────────────────

FS_PPG = 64
DURATION_S = 10

_t_ppg = np.linspace(0, DURATION_S, int(DURATION_S * FS_PPG))
# 1 Hz sinusoid → peaks every second → 10 beats in 10 s → ~60 BPM
PPG_60BPM = np.sin(2 * np.pi * 1.0 * _t_ppg) + 0.3 * np.sin(2 * np.pi * 2.0 * _t_ppg)

FS_EEG = 256
_t_eeg = np.linspace(0, DURATION_S, int(DURATION_S * FS_EEG))
rng = np.random.default_rng(42)
EEG_4CH = rng.standard_normal((4, len(_t_eeg))) * 10  # 4-channel, ~10 µV RMS

# All-zeros PPG — simulates disconnected sensor
PPG_SILENT = np.zeros(int(DURATION_S * FS_PPG))

# Very short PPG — fewer than 3 peaks (1.8 s of 1 Hz signal → at most 2 peaks)
PPG_SHORT = PPG_60BPM[:int(1.8 * FS_PPG)]  # 1.8 seconds → ≤ 2 peaks


# ── PPGFeatureExtractor: detect_peaks ─────────────────────────────────────────

class TestDetectPeaks:
    def setup_method(self):
        self.extractor = PPGFeatureExtractor(fs=FS_PPG)

    def test_finds_peaks_in_60bpm_signal(self):
        peaks = self.extractor.detect_peaks(PPG_60BPM)
        # 10 s at 60 BPM = ~10 peaks; accept 8-12 to be robust to edge effects
        assert 8 <= len(peaks) <= 12, f"Expected ~10 peaks, got {len(peaks)}"

    def test_peak_indices_are_integers(self):
        peaks = self.extractor.detect_peaks(PPG_60BPM)
        assert peaks.dtype in (np.int32, np.int64, int)

    def test_peak_indices_within_signal_bounds(self):
        peaks = self.extractor.detect_peaks(PPG_60BPM)
        assert np.all(peaks >= 0)
        assert np.all(peaks < len(PPG_60BPM))

    def test_no_peaks_in_silent_signal(self):
        peaks = self.extractor.detect_peaks(PPG_SILENT)
        assert len(peaks) == 0

    def test_no_crash_on_single_sample(self):
        peaks = self.extractor.detect_peaks(np.array([1.0]))
        assert isinstance(peaks, np.ndarray)


# ── PPGFeatureExtractor: extract_hrv ─────────────────────────────────────────

class TestExtractHRV:
    def setup_method(self):
        self.extractor = PPGFeatureExtractor(fs=FS_PPG)

    def test_returns_all_required_keys(self):
        result = self.extractor.extract_hrv(PPG_60BPM)
        required = {
            "mean_ibi_ms", "hr_bpm", "sdnn_ms", "rmssd_ms",
            "pnn50", "lf_power", "hf_power", "lf_hf_ratio", "n_beats",
        }
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    def test_hr_bpm_approximately_60_for_1hz_signal(self):
        result = self.extractor.extract_hrv(PPG_60BPM)
        # 1 Hz peaks → 1000 ms IBI → 60 BPM; allow ±10 BPM margin
        assert 50 <= result["hr_bpm"] <= 70, (
            f"Expected ~60 BPM, got {result['hr_bpm']}"
        )

    def test_mean_ibi_approximately_1000ms_for_60bpm(self):
        result = self.extractor.extract_hrv(PPG_60BPM)
        # 60 BPM → IBI = 1000 ms; allow ±100 ms
        assert 900 <= result["mean_ibi_ms"] <= 1100, (
            f"Expected ~1000 ms IBI, got {result['mean_ibi_ms']}"
        )

    def test_fewer_than_3_peaks_returns_zero_hr_bpm(self):
        """When fewer than 3 peaks are found, HRV is undefined; hr_bpm should be 0."""
        result = self.extractor.extract_hrv(PPG_SHORT)
        assert result["hr_bpm"] == 0.0

    def test_fewer_than_3_peaks_no_crash(self):
        result = self.extractor.extract_hrv(PPG_SHORT)
        assert isinstance(result, dict)

    def test_silent_signal_returns_zeros(self):
        result = self.extractor.extract_hrv(PPG_SILENT)
        assert result["hr_bpm"] == 0.0
        assert result["n_beats"] == 0

    def test_n_beats_positive_for_valid_signal(self):
        result = self.extractor.extract_hrv(PPG_60BPM)
        assert result["n_beats"] > 0

    def test_pnn50_in_valid_range(self):
        result = self.extractor.extract_hrv(PPG_60BPM)
        assert 0.0 <= result["pnn50"] <= 1.0, (
            f"pNN50 out of range [0,1]: {result['pnn50']}"
        )

    def test_lf_hf_ratio_nonnegative(self):
        result = self.extractor.extract_hrv(PPG_60BPM)
        assert result["lf_hf_ratio"] >= 0.0


# ── compute_hep ───────────────────────────────────────────────────────────────

class TestComputeHEP:
    def setup_method(self):
        extractor = PPGFeatureExtractor(fs=FS_PPG)
        self.r_peak_times = extractor.get_r_peak_times(PPG_60BPM)

    def test_returns_all_required_keys(self):
        result = compute_hep(EEG_4CH, self.r_peak_times, fs_eeg=FS_EEG)
        required = {
            "hep_amplitude", "hep_peak_latency_ms",
            "n_epochs_averaged", "hep_waveform", "coupling_quality",
        }
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    def test_no_rpeaks_returns_coupling_quality_zero(self):
        result = compute_hep(EEG_4CH, np.array([]), fs_eeg=FS_EEG)
        assert result["coupling_quality"] == 0.0

    def test_no_rpeaks_n_epochs_zero(self):
        result = compute_hep(EEG_4CH, np.array([]), fs_eeg=FS_EEG)
        assert result["n_epochs_averaged"] == 0

    def test_n_epochs_averaged_nonnegative(self):
        result = compute_hep(EEG_4CH, self.r_peak_times, fs_eeg=FS_EEG)
        assert result["n_epochs_averaged"] >= 0

    def test_coupling_quality_in_range_01(self):
        result = compute_hep(EEG_4CH, self.r_peak_times, fs_eeg=FS_EEG)
        assert 0.0 <= result["coupling_quality"] <= 1.0, (
            f"coupling_quality out of range: {result['coupling_quality']}"
        )

    def test_hep_amplitude_nonnegative(self):
        result = compute_hep(EEG_4CH, self.r_peak_times, fs_eeg=FS_EEG)
        assert result["hep_amplitude"] >= 0.0

    def test_hep_waveform_is_list(self):
        result = compute_hep(EEG_4CH, self.r_peak_times, fs_eeg=FS_EEG)
        assert isinstance(result["hep_waveform"], list)

    def test_1d_eeg_accepted(self):
        eeg_1d = EEG_4CH[0]  # single channel
        result = compute_hep(eeg_1d, self.r_peak_times, fs_eeg=FS_EEG)
        assert result["n_epochs_averaged"] >= 0

    def test_hep_peak_latency_ms_nonnegative(self):
        result = compute_hep(EEG_4CH, self.r_peak_times, fs_eeg=FS_EEG)
        assert result["hep_peak_latency_ms"] >= 0.0


# ── compute_heart_brain_coupling ─────────────────────────────────────────────

class TestComputeHeartBrainCoupling:
    def test_merges_hep_and_hrv_results(self):
        result = compute_heart_brain_coupling(EEG_4CH, PPG_60BPM, FS_EEG, FS_PPG)
        # Keys from HEP
        assert "hep_amplitude" in result
        assert "n_epochs_averaged" in result
        # Keys from HRV
        assert "hr_bpm" in result
        assert "rmssd_ms" in result

    def test_interoceptive_index_in_range_01(self):
        result = compute_heart_brain_coupling(EEG_4CH, PPG_60BPM, FS_EEG, FS_PPG)
        idx = result["interoceptive_index"]
        assert 0.0 <= idx <= 1.0, f"interoceptive_index out of range: {idx}"

    def test_delta_hrv_coupling_is_float(self):
        result = compute_heart_brain_coupling(EEG_4CH, PPG_60BPM, FS_EEG, FS_PPG)
        assert isinstance(result["delta_hrv_coupling"], float)

    def test_delta_hrv_coupling_finite(self):
        result = compute_heart_brain_coupling(EEG_4CH, PPG_60BPM, FS_EEG, FS_PPG)
        assert np.isfinite(result["delta_hrv_coupling"]), (
            f"delta_hrv_coupling is not finite: {result['delta_hrv_coupling']}"
        )

    def test_no_crash_with_silent_ppg(self):
        result = compute_heart_brain_coupling(EEG_4CH, PPG_SILENT, FS_EEG, FS_PPG)
        assert isinstance(result, dict)
        assert result["n_beats"] == 0

    def test_n_epochs_averaged_nonneg_with_valid_data(self):
        result = compute_heart_brain_coupling(EEG_4CH, PPG_60BPM, FS_EEG, FS_PPG)
        assert result["n_epochs_averaged"] >= 0
