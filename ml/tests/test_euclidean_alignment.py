"""Tests for Euclidean Alignment (He & Wu, 2018).

Validates:
  - _matrix_invsqrt produces correct inverse square root
  - euclidean_align whitens per-subject covariance to ~identity
  - EuclideanAligner online streaming class
  - Integration: EA-aligned features differ from unaligned
  - Edge cases: single epoch, mismatched channels, etc.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.eeg_processor import (
    _matrix_invsqrt,
    euclidean_align,
    EuclideanAligner,
)


class TestMatrixInvSqrt:
    """Test the matrix inverse square root helper."""

    def test_identity_matrix(self):
        """R^{-1/2} of identity should be identity."""
        I = np.eye(4)
        result = _matrix_invsqrt(I)
        np.testing.assert_allclose(result, I, atol=1e-6)

    def test_diagonal_matrix(self):
        """R^{-1/2} of diag([4, 9, 16]) should be diag([0.5, 1/3, 0.25])."""
        D = np.diag([4.0, 9.0, 16.0])
        result = _matrix_invsqrt(D)
        expected = np.diag([0.5, 1.0 / 3.0, 0.25])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_symmetric_positive_definite(self):
        """R^{-1/2} @ R^{-1/2} should equal R^{-1}."""
        rng = np.random.RandomState(42)
        A = rng.randn(4, 4)
        R = A @ A.T + 0.01 * np.eye(4)  # guaranteed SPD
        R_invsqrt = _matrix_invsqrt(R)

        # R^{-1/2} @ R^{-1/2} should equal R^{-1}
        R_inv_approx = R_invsqrt @ R_invsqrt
        R_inv_true = np.linalg.inv(R)
        np.testing.assert_allclose(R_inv_approx, R_inv_true, atol=1e-4)

    def test_result_is_symmetric(self):
        """R^{-1/2} of a symmetric matrix should also be symmetric."""
        rng = np.random.RandomState(7)
        A = rng.randn(4, 4)
        R = A @ A.T + 0.1 * np.eye(4)
        result = _matrix_invsqrt(R)
        np.testing.assert_allclose(result, result.T, atol=1e-8)

    def test_near_singular_with_regularization(self):
        """Should handle near-singular matrices via regularization."""
        # Rank-deficient matrix (rank 1)
        v = np.array([1.0, 2.0, 3.0, 4.0])
        R = np.outer(v, v)  # rank-1, 3 zero eigenvalues
        result = _matrix_invsqrt(R, reg=1e-4)
        assert np.all(np.isfinite(result))


class TestEuclideanAlign:
    """Test the batch Euclidean Alignment function."""

    def test_output_shape_matches_input(self):
        """Aligned epochs should have same shape as input."""
        rng = np.random.RandomState(42)
        epochs = rng.randn(20, 4, 256)
        aligned, R = euclidean_align(epochs)
        assert aligned.shape == epochs.shape

    def test_mean_covariance_near_identity(self):
        """After alignment, mean covariance should be approximately identity."""
        rng = np.random.RandomState(42)
        # Create epochs with non-identity covariance
        A = rng.randn(4, 4) * 5  # mixing matrix
        raw_epochs = rng.randn(50, 4, 512)
        # Apply subject-specific mixing
        mixed = np.array([A @ epoch for epoch in raw_epochs])

        aligned, R = euclidean_align(mixed)

        # Compute mean covariance of aligned data
        n_ep, n_ch, n_samp = aligned.shape
        cov_mean = np.zeros((n_ch, n_ch))
        for i in range(n_ep):
            cov_mean += aligned[i] @ aligned[i].T / n_samp
        cov_mean /= n_ep

        # Should be close to identity (within tolerance)
        np.testing.assert_allclose(cov_mean, np.eye(n_ch), atol=0.3)

    def test_ref_matrix_is_returned(self):
        """Should return the reference covariance matrix."""
        rng = np.random.RandomState(42)
        epochs = rng.randn(10, 4, 128)
        aligned, R = euclidean_align(epochs)
        assert R.shape == (4, 4)
        # R should be symmetric
        np.testing.assert_allclose(R, R.T, atol=1e-8)

    def test_precomputed_ref_matrix(self):
        """Using a pre-computed ref_matrix should give same result."""
        rng = np.random.RandomState(42)
        epochs = rng.randn(20, 4, 256)

        # First pass: compute reference
        aligned1, R = euclidean_align(epochs)

        # Second pass: use pre-computed reference
        aligned2, _ = euclidean_align(epochs, ref_matrix=R)

        np.testing.assert_allclose(aligned1, aligned2, atol=1e-10)

    def test_single_epoch(self):
        """Should work with a single epoch (degenerates but should not crash)."""
        rng = np.random.RandomState(42)
        epochs = rng.randn(1, 4, 128)
        aligned, R = euclidean_align(epochs)
        assert aligned.shape == (1, 4, 128)
        assert np.all(np.isfinite(aligned))

    def test_different_subjects_aligned(self):
        """Two subjects with different covariances should become more similar after EA."""
        rng = np.random.RandomState(42)
        base = rng.randn(20, 4, 256)

        # Subject 1: scale channels differently
        A1 = np.diag([1.0, 3.0, 0.5, 2.0])
        subj1 = np.array([A1 @ ep for ep in base])

        # Subject 2: different scaling
        A2 = np.diag([2.0, 0.5, 3.0, 1.0])
        subj2 = np.array([A2 @ ep for ep in base])

        # Before EA: covariances are different
        cov1_before = np.mean([s @ s.T / 256 for s in subj1], axis=0)
        cov2_before = np.mean([s @ s.T / 256 for s in subj2], axis=0)
        dist_before = np.linalg.norm(cov1_before - cov2_before, ord="fro")

        # After EA: covariances should be closer
        aligned1, _ = euclidean_align(subj1)
        aligned2, _ = euclidean_align(subj2)
        cov1_after = np.mean([s @ s.T / 256 for s in aligned1], axis=0)
        cov2_after = np.mean([s @ s.T / 256 for s in aligned2], axis=0)
        dist_after = np.linalg.norm(cov1_after - cov2_after, ord="fro")

        assert dist_after < dist_before, (
            f"EA should reduce inter-subject covariance distance: "
            f"before={dist_before:.4f}, after={dist_after:.4f}"
        )


class TestEuclideanAligner:
    """Test the online streaming EuclideanAligner class."""

    def test_not_ready_initially(self):
        """Should not be ready until min_epochs accumulated."""
        aligner = EuclideanAligner(n_channels=4, min_epochs=5)
        assert not aligner.is_ready
        assert aligner.n_epochs_seen == 0

    def test_becomes_ready_after_min_epochs(self):
        """Should become ready after accumulating min_epochs."""
        rng = np.random.RandomState(42)
        aligner = EuclideanAligner(n_channels=4, min_epochs=5)

        for i in range(5):
            epoch = rng.randn(4, 128)
            aligner.add_epoch(epoch)

        assert aligner.is_ready
        assert aligner.n_epochs_seen == 5

    def test_returns_unchanged_when_not_ready(self):
        """align() should return input unchanged before is_ready."""
        rng = np.random.RandomState(42)
        aligner = EuclideanAligner(n_channels=4, min_epochs=10)
        epoch = rng.randn(4, 128)
        result = aligner.align(epoch)
        np.testing.assert_array_equal(result, epoch)

    def test_returns_aligned_when_ready(self):
        """align() should transform data after is_ready."""
        rng = np.random.RandomState(42)
        aligner = EuclideanAligner(n_channels=4, min_epochs=5)

        for i in range(10):
            aligner.add_epoch(rng.randn(4, 128))

        epoch = rng.randn(4, 128)
        result = aligner.align(epoch)
        # Should be different from input (unless by coincidence)
        assert not np.array_equal(result, epoch)
        assert result.shape == epoch.shape
        assert np.all(np.isfinite(result))

    def test_update_and_align(self):
        """update_and_align should both accumulate and return aligned."""
        rng = np.random.RandomState(42)
        aligner = EuclideanAligner(n_channels=4, min_epochs=3)

        results = []
        for i in range(5):
            epoch = rng.randn(4, 128)
            result = aligner.update_and_align(epoch)
            results.append(result)

        assert aligner.n_epochs_seen == 5
        assert aligner.is_ready

    def test_set_ref_matrix(self):
        """Setting a pre-computed ref matrix should make aligner ready."""
        rng = np.random.RandomState(42)
        R = np.eye(4) * 2.0  # simple diagonal covariance
        aligner = EuclideanAligner(n_channels=4, min_epochs=100)
        assert not aligner.is_ready

        aligner.set_ref_matrix(R)
        assert aligner.is_ready

        epoch = rng.randn(4, 128)
        result = aligner.align(epoch)
        # With R = 2*I, R^{-1/2} = (1/sqrt(2))*I, so data scaled by ~0.707
        expected_scale = 1.0 / np.sqrt(2.0)
        np.testing.assert_allclose(result, epoch * expected_scale, atol=1e-4)

    def test_get_ref_matrix(self):
        """get_ref_matrix should return the accumulated covariance."""
        rng = np.random.RandomState(42)
        aligner = EuclideanAligner(n_channels=4, min_epochs=3)

        for i in range(5):
            aligner.add_epoch(rng.randn(4, 128))

        R = aligner.get_ref_matrix()
        assert R is not None
        assert R.shape == (4, 4)
        np.testing.assert_allclose(R, R.T, atol=1e-8)

    def test_reset_clears_state(self):
        """reset() should clear all accumulated state."""
        rng = np.random.RandomState(42)
        aligner = EuclideanAligner(n_channels=4, min_epochs=3)

        for i in range(5):
            aligner.add_epoch(rng.randn(4, 128))
        assert aligner.is_ready

        aligner.reset()
        assert not aligner.is_ready
        assert aligner.n_epochs_seen == 0
        assert aligner.get_ref_matrix() is None

    def test_ignores_wrong_channel_count(self):
        """Should silently ignore epochs with wrong number of channels."""
        aligner = EuclideanAligner(n_channels=4, min_epochs=3)
        aligner.add_epoch(np.random.randn(3, 128))  # wrong: 3 channels
        aligner.add_epoch(np.random.randn(5, 128))  # wrong: 5 channels
        assert aligner.n_epochs_seen == 0

    def test_ignores_1d_input(self):
        """Should ignore 1D arrays."""
        aligner = EuclideanAligner(n_channels=4, min_epochs=3)
        aligner.add_epoch(np.random.randn(128))  # 1D
        assert aligner.n_epochs_seen == 0


class TestEuclideanAlignIntegration:
    """Integration tests: EA + feature extraction pipeline."""

    def test_aligned_features_are_finite(self):
        """Features extracted from EA-aligned data should be finite."""
        from processing.eeg_processor import extract_features, preprocess

        rng = np.random.RandomState(42)
        epochs = rng.randn(10, 4, 1024) * 20  # 4 sec at 256 Hz
        aligned, _ = euclidean_align(epochs)

        for epoch in aligned:
            for ch in range(4):
                proc = preprocess(epoch[ch], 256.0)
                feats = extract_features(proc, 256.0)
                for key, val in feats.items():
                    assert np.isfinite(val), f"Non-finite {key} after EA"

    def test_batch_ea_matches_online_ea(self):
        """Batch euclidean_align and online EuclideanAligner should agree."""
        rng = np.random.RandomState(42)
        epochs = rng.randn(20, 4, 256)

        # Batch
        aligned_batch, R_batch = euclidean_align(epochs)

        # Online: feed all epochs to build reference, then align
        aligner = EuclideanAligner(n_channels=4, min_epochs=1)
        for ep in epochs:
            aligner.add_epoch(ep)

        # The reference matrices should be equal
        R_online = aligner.get_ref_matrix()
        np.testing.assert_allclose(R_batch, R_online, atol=1e-8)

        # Aligned outputs should be equal
        for i in range(len(epochs)):
            aligned_online = aligner.align(epochs[i])
            np.testing.assert_allclose(
                aligned_online, aligned_batch[i], atol=1e-6,
                err_msg=f"Mismatch at epoch {i}"
            )
