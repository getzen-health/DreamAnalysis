"""Shared fixtures for all tests."""

import sys
import numpy as np
import pytest
from pathlib import Path

# Add ml/ to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_eeg():
    """4 seconds of clean synthetic EEG at 256 Hz."""
    return np.random.randn(1024) * 20  # ~20 uV RMS


@pytest.fixture
def multichannel_eeg():
    """4 channels x 4 seconds of synthetic EEG."""
    return np.random.randn(4, 1024) * 20


@pytest.fixture
def flat_signal():
    """Flat-line signal (disconnected electrode)."""
    return np.ones(1024) * 0.001


@pytest.fixture
def noisy_signal():
    """EEG dominated by 60Hz line noise."""
    t = np.arange(1024) / 256
    return np.random.randn(1024) * 10 + 100 * np.sin(2 * np.pi * 60 * t)


@pytest.fixture
def railed_signal():
    """Saturated/railed signal (>200 uV)."""
    return np.random.randn(1024) * 300


@pytest.fixture
def fs():
    """Standard sampling frequency."""
    return 256
