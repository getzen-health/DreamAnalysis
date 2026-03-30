"""
YASA Sleep Staging Integration (planned)

YASA (Yet Another Spindle Algorithm) provides automated sleep staging
with published accuracy benchmarks. This module will wrap YASA's
SleepStaging class for comparison against our custom model.

Requirements:
  pip install yasa mne

Usage (planned):
  from yasa_integration import stage_with_yasa
  stages = stage_with_yasa(eeg_data, fs=256)

Current custom model accuracy: 92.98% (ISRUC dataset)
YASA published accuracy: ~85% on Sleep-EDF (cross-dataset)

Integration steps:
1. Install yasa + mne in ml/requirements.txt
2. Convert BrainFlow data to MNE Raw format
3. Run YASA staging alongside custom model
4. Compare results in /api/sleep/compare endpoint
5. Report both accuracies honestly in the UI

Status: Planned, not yet active. See issue #556.
"""


def stage_with_yasa(eeg_data, fs: float = 256.0, channel_names: list = None):
    """Placeholder — will wrap yasa.SleepStaging when installed."""
    raise NotImplementedError("YASA not yet installed. Run: pip install yasa mne")
