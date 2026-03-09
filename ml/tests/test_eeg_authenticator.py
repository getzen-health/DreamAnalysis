"""Tests for EEGAuthenticator — biometric authentication via PSD fingerprinting."""

import numpy as np
import pytest

from models.eeg_authenticator import EEGAuthenticator, get_eeg_authenticator, _cosine_similarity


FS = 256  # Hz


def make_eeg(n_channels: int = 4, duration_s: float = 15.0, seed: int = 42) -> np.ndarray:
    """Generate consistent synthetic EEG with per-seed spectral fingerprint."""
    rng = np.random.default_rng(seed)
    n_samples = int(FS * duration_s)
    t = np.linspace(0, duration_s, n_samples)
    # Each seed gets a unique alpha peak (8-12 Hz) and beta ratio
    alpha_hz = 9.0 + (seed % 3)
    beta_hz = 18.0 + (seed % 5)
    signal = (
        rng.normal(0, 5, (n_channels, n_samples))
        + 8 * np.sin(2 * np.pi * alpha_hz * t)
        + 4 * np.sin(2 * np.pi * beta_hz * t)
    )
    return signal.astype(np.float32)


@pytest.fixture
def auth():
    return EEGAuthenticator()


# ── Unit tests: EEGAuthenticator ──────────────────────────────────────────────

class TestInit:
    def test_initial_status(self, auth):
        status = auth.get_status()
        assert status["enrolled_users"] == []
        assert status["ready_users"] == []
        assert "verify_threshold" in status
        assert "enroll_min_segments" in status


class TestEnroll:
    def test_enroll_returns_dict(self, auth):
        eeg = make_eeg()
        result = auth.enroll(eeg, fs=FS, user_id="alice")
        assert isinstance(result, dict)
        assert "enrolled_segments" in result
        assert "template_ready" in result
        assert "user_id" in result

    def test_enroll_increments_segments(self, auth):
        eeg = make_eeg()
        for i in range(1, 4):
            result = auth.enroll(eeg, fs=FS, user_id="alice")
            assert result["enrolled_segments"] == i

    def test_enroll_not_ready_before_3_segments(self, auth):
        eeg = make_eeg()
        for _ in range(2):
            result = auth.enroll(eeg, fs=FS, user_id="alice")
        assert result["template_ready"] is False

    def test_enroll_ready_after_3_segments(self, auth):
        eeg = make_eeg()
        for _ in range(3):
            result = auth.enroll(eeg, fs=FS, user_id="alice")
        assert result["template_ready"] is True

    def test_enroll_updates_status(self, auth):
        eeg = make_eeg()
        for _ in range(3):
            auth.enroll(eeg, fs=FS, user_id="alice")
        status = auth.get_status()
        assert "alice" in status["enrolled_users"]
        assert "alice" in status["ready_users"]

    def test_enroll_short_signal_returns_error(self, auth):
        # 0.5 seconds — too short (min is 10 seconds)
        eeg_short = np.random.randn(4, int(FS * 0.5)).astype(np.float32)
        result = auth.enroll(eeg_short, fs=FS, user_id="alice")
        assert "error" in result

    def test_enroll_multiple_users(self, auth):
        eeg = make_eeg()
        for user in ["alice", "bob", "carol"]:
            for _ in range(3):
                auth.enroll(eeg, fs=FS, user_id=user)
        status = auth.get_status()
        assert set(status["ready_users"]) == {"alice", "bob", "carol"}


class TestVerify:
    def test_verify_before_enroll_returns_error(self, auth):
        eeg = make_eeg()
        result = auth.verify(eeg, fs=FS, user_id="ghost")
        assert result["match"] is False
        assert result["template_ready"] is False
        assert "error" in result

    def test_verify_same_person_matches(self, auth):
        """Same spectral fingerprint should match."""
        eeg = make_eeg(seed=1)
        for _ in range(3):
            auth.enroll(eeg, fs=FS, user_id="alice")
        result = auth.verify(eeg, fs=FS, user_id="alice")
        assert result["template_ready"] is True
        assert result["similarity"] > 0.5
        # High similarity (same signal) should produce match
        assert result["match"] is True

    def test_verify_returns_required_keys(self, auth):
        eeg = make_eeg()
        for _ in range(3):
            auth.enroll(eeg, fs=FS, user_id="alice")
        result = auth.verify(eeg, fs=FS, user_id="alice")
        for key in ["match", "similarity", "threshold", "template_ready", "user_id"]:
            assert key in result

    def test_verify_similarity_in_range(self, auth):
        eeg = make_eeg()
        for _ in range(3):
            auth.enroll(eeg, fs=FS, user_id="alice")
        result = auth.verify(eeg, fs=FS, user_id="alice")
        assert 0.0 <= result["similarity"] <= 1.0

    def test_verify_short_signal_returns_error(self, auth):
        eeg = make_eeg()
        for _ in range(3):
            auth.enroll(eeg, fs=FS, user_id="alice")
        eeg_short = np.random.randn(4, int(FS * 0.5)).astype(np.float32)
        result = auth.verify(eeg_short, fs=FS, user_id="alice")
        assert "error" in result
        assert result["match"] is False


class TestIdentify:
    def test_identify_no_users_returns_error(self, auth):
        eeg = make_eeg()
        result = auth.identify(eeg, fs=FS)
        assert result["identified_user"] is None
        assert "error" in result

    def test_identify_returns_candidates(self, auth):
        eeg = make_eeg()
        for user in ["alice", "bob"]:
            for _ in range(3):
                auth.enroll(eeg, fs=FS, user_id=user)
        result = auth.identify(eeg, fs=FS)
        assert "candidates" in result
        assert len(result["candidates"]) == 2

    def test_identify_candidates_sorted_desc(self, auth):
        eeg = make_eeg()
        for user in ["alice", "bob"]:
            for _ in range(3):
                auth.enroll(eeg, fs=FS, user_id=user)
        result = auth.identify(eeg, fs=FS)
        sims = [c["similarity"] for c in result["candidates"]]
        assert sims == sorted(sims, reverse=True)

    def test_identify_returns_required_keys(self, auth):
        eeg = make_eeg()
        for _ in range(3):
            auth.enroll(eeg, fs=FS, user_id="alice")
        result = auth.identify(eeg, fs=FS)
        for key in ["identified_user", "similarity", "threshold", "candidates"]:
            assert key in result

    def test_identify_same_signal(self, auth):
        eeg = make_eeg(seed=7)
        for _ in range(3):
            auth.enroll(eeg, fs=FS, user_id="alice")
        result = auth.identify(eeg, fs=FS)
        # Same signal — should identify alice
        assert result["identified_user"] == "alice"


class TestDelete:
    def test_delete_existing_user(self, auth):
        eeg = make_eeg()
        for _ in range(3):
            auth.enroll(eeg, fs=FS, user_id="alice")
        result = auth.delete_template("alice")
        assert result["deleted"] is True
        status = auth.get_status()
        assert "alice" not in status["enrolled_users"]

    def test_delete_nonexistent_user(self, auth):
        result = auth.delete_template("nobody")
        assert result["deleted"] is False

    def test_verify_after_delete_fails(self, auth):
        eeg = make_eeg()
        for _ in range(3):
            auth.enroll(eeg, fs=FS, user_id="alice")
        auth.delete_template("alice")
        result = auth.verify(eeg, fs=FS, user_id="alice")
        assert result["match"] is False
        assert result["template_ready"] is False


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector(self):
        a = np.zeros(5)
        b = np.ones(5)
        assert _cosine_similarity(a, b) == 0.0

    def test_range(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            a = rng.standard_normal(50).astype(np.float32)
            b = rng.standard_normal(50).astype(np.float32)
            sim = _cosine_similarity(a, b)
            assert -1.0 <= sim <= 1.0


class TestSingleton:
    def test_same_instance(self):
        a = get_eeg_authenticator()
        b = get_eeg_authenticator()
        assert a is b

    def test_is_eeg_authenticator(self):
        assert isinstance(get_eeg_authenticator(), EEGAuthenticator)


class TestMultichannelAndShapes:
    def test_1d_input_enroll(self, auth):
        eeg_1d = make_eeg(n_channels=1, duration_s=15.0).reshape(-1)
        result = auth.enroll(eeg_1d, fs=FS, user_id="alice")
        assert "enrolled_segments" in result

    def test_single_channel_2d(self, auth):
        eeg_1ch = make_eeg(n_channels=1, duration_s=15.0)
        for _ in range(3):
            auth.enroll(eeg_1ch, fs=FS, user_id="alice")
        result = auth.verify(eeg_1ch, fs=FS, user_id="alice")
        assert "similarity" in result
