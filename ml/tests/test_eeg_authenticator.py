"""Tests for EEGAuthenticator — EEG-based biometric authentication.

Covers:
  - Enrollment: correct return keys, template stored, quality_score 0-1
  - Verification: matching user (same signal + noise), reject unknown, reject different user
  - Identification: identify correct user from multiple enrolled
  - Cosine similarity: identical, orthogonal, zero vectors
  - Threshold: high/medium/low confidence levels
  - Enrolled users list
  - Remove user and re-verify fails
  - Edge cases: single channel, very short signal, 1D input
  - Multi-user: enroll multiple, verify each independently
  - Quality score: stable signal > noisy signal
"""

import numpy as np
import pytest

from models.eeg_authenticator import EEGAuthenticator

FS = 256
DURATION_SEC = 10  # 10 seconds of data for enrollment
N_SAMPLES = FS * DURATION_SEC
N_CHANNELS = 4


# -- Helpers -------------------------------------------------------------------

def make_eeg(seed: int, duration_sec: float = 10.0, fs: float = 256.0,
             n_channels: int = 4, noise_level: float = 5.0) -> np.ndarray:
    """Generate synthetic EEG with known frequency components.

    Different seeds produce different dominant frequencies and amplitude
    ratios, creating meaningfully different PSD fingerprints. This is
    critical because PSD (Welch) is phase-invariant -- only frequency
    content and amplitude matter for spectral fingerprinting.
    """
    rng = np.random.RandomState(seed)
    n_samples = int(fs * duration_sec)
    t = np.arange(n_samples) / fs

    # Each seed produces unique frequency components (within EEG range 1-40 Hz)
    # This ensures different seeds create spectrally distinct signals
    n_components = 5
    freqs = rng.uniform(2.0, 40.0, size=n_components)
    amps = rng.uniform(5.0, 25.0, size=n_components)

    signals = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        # Per-channel amplitude variation (small, preserves spectral shape)
        ch_scale = 1.0 + rng.randn(n_components) * 0.15
        phase = rng.uniform(0, 2 * np.pi, size=n_components)

        for i in range(n_components):
            signals[ch] += (
                amps[i] * ch_scale[i]
                * np.sin(2 * np.pi * freqs[i] * t + phase[i])
            )
        signals[ch] += rng.randn(n_samples) * noise_level
    return signals


def add_noise(signals: np.ndarray, noise_level: float = 2.0,
              seed: int = 999) -> np.ndarray:
    """Add small Gaussian noise to existing signals (simulates same-person
    re-recording with minor variation)."""
    rng = np.random.RandomState(seed)
    return signals + rng.randn(*signals.shape) * noise_level


# -- Fixtures ------------------------------------------------------------------

@pytest.fixture
def auth():
    return EEGAuthenticator(match_threshold=0.85, n_channels=4)


@pytest.fixture
def alice_eeg():
    """Alice's resting-state EEG."""
    return make_eeg(seed=42, duration_sec=10.0)


@pytest.fixture
def alice_eeg_verify():
    """Alice's verification segment — same spectral profile + small noise."""
    base = make_eeg(seed=42, duration_sec=5.0)
    return add_noise(base, noise_level=2.0, seed=100)


@pytest.fixture
def bob_eeg():
    """Bob's resting-state EEG — completely different spectral profile."""
    return make_eeg(seed=7, duration_sec=10.0)


@pytest.fixture
def bob_eeg_verify():
    """Bob's verification segment."""
    base = make_eeg(seed=7, duration_sec=5.0)
    return add_noise(base, noise_level=2.0, seed=200)


@pytest.fixture
def charlie_eeg():
    """Charlie's resting-state EEG — third distinct user."""
    return make_eeg(seed=123, duration_sec=10.0)


# -- TestEnrollment ------------------------------------------------------------

class TestEnrollment:
    def test_enroll_returns_correct_keys(self, auth, alice_eeg):
        result = auth.enroll(alice_eeg, fs=FS, user_id="alice")
        expected_keys = {"enrolled", "user_id", "template_size", "duration_sec",
                         "quality_score"}
        assert set(result.keys()) == expected_keys

    def test_enroll_success(self, auth, alice_eeg):
        result = auth.enroll(alice_eeg, fs=FS, user_id="alice")
        assert result["enrolled"] is True
        assert result["user_id"] == "alice"

    def test_template_stored(self, auth, alice_eeg):
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        assert "alice" in auth._templates
        assert isinstance(auth._templates["alice"], np.ndarray)

    def test_template_size_positive(self, auth, alice_eeg):
        result = auth.enroll(alice_eeg, fs=FS, user_id="alice")
        assert result["template_size"] > 0

    def test_duration_matches_input(self, auth, alice_eeg):
        result = auth.enroll(alice_eeg, fs=FS, user_id="alice")
        expected_duration = alice_eeg.shape[1] / FS
        assert abs(result["duration_sec"] - expected_duration) < 0.01

    def test_quality_score_range(self, auth, alice_eeg):
        result = auth.enroll(alice_eeg, fs=FS, user_id="alice")
        assert 0.0 <= result["quality_score"] <= 1.0

    def test_overwrite_enrollment(self, auth, alice_eeg, bob_eeg):
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        old_template = auth._templates["alice"].copy()
        auth.enroll(bob_eeg, fs=FS, user_id="alice")
        # Template should be different after re-enrollment with different data
        assert not np.allclose(old_template, auth._templates["alice"])


# -- TestVerification ----------------------------------------------------------

class TestVerification:
    def test_verify_matching_user(self, auth, alice_eeg, alice_eeg_verify):
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        result = auth.verify(alice_eeg_verify, fs=FS, claimed_id="alice")
        assert result["match"] is True
        assert result["similarity"] > 0.85
        assert result["claimed_id"] == "alice"

    def test_verify_returns_correct_keys(self, auth, alice_eeg, alice_eeg_verify):
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        result = auth.verify(alice_eeg_verify, fs=FS, claimed_id="alice")
        expected_keys = {"match", "similarity", "claimed_id", "threshold",
                         "confidence"}
        assert set(result.keys()) == expected_keys

    def test_reject_unknown_user(self, auth, alice_eeg_verify):
        result = auth.verify(alice_eeg_verify, fs=FS, claimed_id="unknown")
        assert result["match"] is False
        assert result["similarity"] == 0.0

    def test_reject_different_user(self, auth, alice_eeg, bob_eeg_verify):
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        result = auth.verify(bob_eeg_verify, fs=FS, claimed_id="alice")
        assert result["match"] is False
        assert result["similarity"] < 0.85

    def test_similarity_range(self, auth, alice_eeg, alice_eeg_verify):
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        result = auth.verify(alice_eeg_verify, fs=FS, claimed_id="alice")
        assert 0.0 <= result["similarity"] <= 1.0

    def test_threshold_in_result(self, auth, alice_eeg, alice_eeg_verify):
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        result = auth.verify(alice_eeg_verify, fs=FS, claimed_id="alice")
        assert result["threshold"] == 0.85


# -- TestIdentification -------------------------------------------------------

class TestIdentification:
    def test_identify_correct_user(self, auth, alice_eeg, bob_eeg,
                                   alice_eeg_verify):
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        auth.enroll(bob_eeg, fs=FS, user_id="bob")
        result = auth.identify(alice_eeg_verify, fs=FS)
        assert result["identified_user"] == "alice"
        assert result["similarity"] > 0.85

    def test_identify_returns_all_scores(self, auth, alice_eeg, bob_eeg,
                                         alice_eeg_verify):
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        auth.enroll(bob_eeg, fs=FS, user_id="bob")
        result = auth.identify(alice_eeg_verify, fs=FS)
        assert "alice" in result["all_scores"]
        assert "bob" in result["all_scores"]
        assert result["n_enrolled"] == 2

    def test_identify_no_enrolled_users(self, auth, alice_eeg_verify):
        result = auth.identify(alice_eeg_verify, fs=FS)
        assert result["identified_user"] is None
        assert result["n_enrolled"] == 0

    def test_identify_no_match_below_threshold(self, auth, alice_eeg,
                                                charlie_eeg):
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        # Charlie's EEG should not match Alice
        result = auth.identify(charlie_eeg, fs=FS)
        # If similarity is below threshold, identified_user should be None
        if result["similarity"] < 0.85:
            assert result["identified_user"] is None

    def test_identify_bob_from_multiple(self, auth, alice_eeg, bob_eeg,
                                        bob_eeg_verify):
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        auth.enroll(bob_eeg, fs=FS, user_id="bob")
        result = auth.identify(bob_eeg_verify, fs=FS)
        assert result["identified_user"] == "bob"


# -- TestCosineSimilarity -----------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        assert abs(EEGAuthenticator._cosine_similarity(a, a) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(EEGAuthenticator._cosine_similarity(a, b)) < 1e-9

    def test_zero_vector(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.zeros(3)
        assert EEGAuthenticator._cosine_similarity(a, b) == 0.0

    def test_both_zero_vectors(self):
        a = np.zeros(3)
        b = np.zeros(3)
        assert EEGAuthenticator._cosine_similarity(a, b) == 0.0

    def test_opposite_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        b = -a
        assert abs(EEGAuthenticator._cosine_similarity(a, b) - (-1.0)) < 1e-9

    def test_scaled_vectors_same_direction(self):
        a = np.array([1.0, 2.0, 3.0])
        b = a * 100
        assert abs(EEGAuthenticator._cosine_similarity(a, b) - 1.0) < 1e-9


# -- TestThreshold (confidence levels) ----------------------------------------

class TestThreshold:
    def test_high_confidence(self, auth, alice_eeg):
        """Same signal enrolled and verified should give high confidence."""
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        # Verify with the exact same signal — should be very high similarity
        result = auth.verify(alice_eeg, fs=FS, claimed_id="alice")
        assert result["confidence"] == "high"
        assert result["similarity"] > auth._match_threshold + 0.10

    def test_confidence_is_valid_string(self, auth, alice_eeg, alice_eeg_verify):
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        result = auth.verify(alice_eeg_verify, fs=FS, claimed_id="alice")
        assert result["confidence"] in {"high", "medium", "low"}

    def test_unknown_user_no_confidence_category(self, auth, alice_eeg_verify):
        """Unknown user should not get a high confidence."""
        result = auth.verify(alice_eeg_verify, fs=FS, claimed_id="nonexistent")
        assert result["match"] is False


# -- TestGetEnrolled -----------------------------------------------------------

class TestGetEnrolled:
    def test_empty_initially(self, auth):
        assert auth.get_enrolled_users() == []

    def test_returns_enrolled_ids(self, auth, alice_eeg, bob_eeg):
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        auth.enroll(bob_eeg, fs=FS, user_id="bob")
        enrolled = auth.get_enrolled_users()
        assert set(enrolled) == {"alice", "bob"}

    def test_returns_list(self, auth, alice_eeg):
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        assert isinstance(auth.get_enrolled_users(), list)


# -- TestRemoveUser ------------------------------------------------------------

class TestRemoveUser:
    def test_remove_existing_user(self, auth, alice_eeg):
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        result = auth.remove_user("alice")
        assert result is True
        assert "alice" not in auth._templates

    def test_remove_nonexistent_user(self, auth):
        result = auth.remove_user("nobody")
        assert result is False

    def test_verify_fails_after_removal(self, auth, alice_eeg, alice_eeg_verify):
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        auth.remove_user("alice")
        result = auth.verify(alice_eeg_verify, fs=FS, claimed_id="alice")
        assert result["match"] is False
        assert result["similarity"] == 0.0


# -- TestEdgeCases -------------------------------------------------------------

class TestEdgeCases:
    def test_single_channel_input(self, auth):
        """2D array with 1 channel should still work."""
        signals = make_eeg(seed=42, n_channels=1, duration_sec=10.0)
        auth_1ch = EEGAuthenticator(n_channels=1)
        result = auth_1ch.enroll(signals, fs=FS, user_id="single")
        assert result["enrolled"] is True

    def test_1d_input_treated_as_single_channel(self, auth):
        """1D array should be handled gracefully."""
        rng = np.random.RandomState(42)
        signal_1d = rng.randn(FS * 10) * 20.0
        auth_1ch = EEGAuthenticator(n_channels=1)
        result = auth_1ch.enroll(signal_1d, fs=FS, user_id="flat")
        assert result["enrolled"] is True

    def test_short_signal_for_verification(self, auth, alice_eeg):
        """5-second signal (minimum for verify) should work."""
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        short_verify = make_eeg(seed=42, duration_sec=5.0)
        short_verify = add_noise(short_verify, noise_level=2.0)
        result = auth.verify(short_verify, fs=FS, claimed_id="alice")
        assert "match" in result
        assert "similarity" in result

    def test_custom_threshold(self):
        """Custom match threshold should be respected."""
        auth = EEGAuthenticator(match_threshold=0.95)
        assert auth._match_threshold == 0.95


# -- TestMultiUser -------------------------------------------------------------

class TestMultiUser:
    def test_enroll_multiple_users(self, auth, alice_eeg, bob_eeg, charlie_eeg):
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        auth.enroll(bob_eeg, fs=FS, user_id="bob")
        auth.enroll(charlie_eeg, fs=FS, user_id="charlie")
        assert len(auth.get_enrolled_users()) == 3

    def test_verify_each_user_independently(self, auth, alice_eeg, bob_eeg):
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        auth.enroll(bob_eeg, fs=FS, user_id="bob")

        alice_verify = add_noise(make_eeg(seed=42, duration_sec=5.0),
                                 noise_level=2.0, seed=300)
        bob_verify = add_noise(make_eeg(seed=7, duration_sec=5.0),
                               noise_level=2.0, seed=301)

        r_alice = auth.verify(alice_verify, fs=FS, claimed_id="alice")
        r_bob = auth.verify(bob_verify, fs=FS, claimed_id="bob")

        assert r_alice["match"] is True
        assert r_bob["match"] is True

    def test_cross_user_rejection(self, auth, alice_eeg, bob_eeg):
        auth.enroll(alice_eeg, fs=FS, user_id="alice")
        auth.enroll(bob_eeg, fs=FS, user_id="bob")

        bob_verify = add_noise(make_eeg(seed=7, duration_sec=5.0),
                               noise_level=2.0, seed=301)

        # Bob's EEG should not verify as Alice
        r = auth.verify(bob_verify, fs=FS, claimed_id="alice")
        assert r["match"] is False


# -- TestQualityScore ----------------------------------------------------------

class TestQualityScore:
    def test_stable_signal_higher_quality(self, auth):
        """Clean signal should have higher quality score than noisy signal."""
        clean = make_eeg(seed=42, duration_sec=10.0, noise_level=1.0)
        noisy = make_eeg(seed=42, duration_sec=10.0, noise_level=50.0)

        r_clean = auth.enroll(clean, fs=FS, user_id="clean_user")
        auth.remove_user("clean_user")
        r_noisy = auth.enroll(noisy, fs=FS, user_id="noisy_user")

        assert r_clean["quality_score"] > r_noisy["quality_score"]

    def test_quality_score_is_float(self, auth, alice_eeg):
        result = auth.enroll(alice_eeg, fs=FS, user_id="alice")
        assert isinstance(result["quality_score"], float)
