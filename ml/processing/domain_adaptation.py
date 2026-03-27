"""Domain adaptation for EEG feature alignment.

CORAL: Correlation Alignment (Sun & Saenko, 2016)
Aligns source domain (research EEG) to target domain (consumer Muse 2).

Unlike the diagonal CORAL-lite in models/domain_adapter.py (z-score normalization),
this implements full CORAL with covariance whitening and re-coloring.  Use this when
you have enough target samples (>=30) to estimate a stable covariance matrix.

References:
    Sun & Saenko (2016) — Return of Frustratingly Easy Domain Adaptation
    He & Wu (2019) — Transfer Learning for Brain-Computer Interfaces

GitHub issue: #541
"""

from typing import Optional

import numpy as np


class CORALAdapter:
    """CORAL domain adaptation -- aligns feature covariances between domains.

    Whitens source features by Cs^{-1/2} and re-colors them with Ct^{1/2},
    shifting the source distribution to match the target covariance structure.

    Usage::

        adapter = CORALAdapter()
        adapter.fit(source_features, target_features)
        aligned = adapter.transform(new_source_features)
        adapter.save("coral_adapter.npz")
        loaded  = CORALAdapter.load("coral_adapter.npz")
    """

    def __init__(self, reg: float = 1e-5) -> None:
        self.reg = reg
        self.source_mean: Optional[np.ndarray] = None
        self.source_cov_whitening: Optional[np.ndarray] = None
        self.target_mean: Optional[np.ndarray] = None
        self.target_cov_coloring: Optional[np.ndarray] = None
        self.fitted = False

    def fit(
        self, source_features: np.ndarray, target_features: np.ndarray
    ) -> "CORALAdapter":
        """Fit CORAL from source (research) and target (Muse 2) feature matrices.

        Args:
            source_features: (n_source, n_features) array from research datasets.
            target_features: (n_target, n_features) array from Muse 2 recordings.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If feature dimensions mismatch or arrays are empty.
        """
        if source_features.ndim == 1:
            source_features = source_features.reshape(1, -1)
        if target_features.ndim == 1:
            target_features = target_features.reshape(1, -1)

        if source_features.shape[1] != target_features.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: source has {source_features.shape[1]}, "
                f"target has {target_features.shape[1]}"
            )
        if source_features.shape[0] == 0 or target_features.shape[0] == 0:
            raise ValueError("Source and target feature matrices must be non-empty.")

        n_features = source_features.shape[1]

        # Compute means
        self.source_mean = source_features.mean(axis=0)
        self.target_mean = target_features.mean(axis=0)

        # Covariance matrices with regularization for numerical stability
        Cs = np.cov((source_features - self.source_mean).T) + self.reg * np.eye(
            n_features
        )
        Ct = np.cov((target_features - self.target_mean).T) + self.reg * np.eye(
            n_features
        )

        # Handle single-feature edge case: np.cov returns a scalar for 1D input
        if n_features == 1:
            Cs = np.atleast_2d(Cs)
            Ct = np.atleast_2d(Ct)

        # Whitening: Cs^{-1/2}
        Us, Ss, _ = np.linalg.svd(Cs)
        self.source_cov_whitening = (
            Us @ np.diag(1.0 / np.sqrt(Ss + self.reg)) @ Us.T
        )

        # Coloring: Ct^{1/2}
        Ut, St, _ = np.linalg.svd(Ct)
        self.target_cov_coloring = Ut @ np.diag(np.sqrt(St + self.reg)) @ Ut.T

        self.fitted = True
        return self

    def transform(self, source_features: np.ndarray) -> np.ndarray:
        """Transform source features to align with target domain.

        Args:
            source_features: (n_samples, n_features) or (n_features,) array.

        Returns:
            Aligned feature array with same shape as input.

        Raises:
            ValueError: If fit() has not been called.
        """
        if not self.fitted:
            raise ValueError("Call fit() before transform()")

        single = source_features.ndim == 1
        if single:
            source_features = source_features.reshape(1, -1)

        centered = source_features - self.source_mean
        aligned = (
            centered @ self.source_cov_whitening @ self.target_cov_coloring
        ) + self.target_mean

        return aligned.ravel() if single else aligned

    def save(self, path: str) -> None:
        """Save fitted adapter to numpy archive.

        Args:
            path: File path ending in .npz (or without extension -- numpy adds it).
        """
        if not self.fitted:
            raise ValueError("Cannot save unfitted adapter.")
        np.savez(
            path,
            source_mean=self.source_mean,
            source_cov_whitening=self.source_cov_whitening,
            target_mean=self.target_mean,
            target_cov_coloring=self.target_cov_coloring,
            fitted=self.fitted,
        )

    @classmethod
    def load(cls, path: str) -> "CORALAdapter":
        """Load a fitted adapter from numpy archive.

        Args:
            path: Path to .npz file saved by save().

        Returns:
            A fitted CORALAdapter instance.
        """
        data = np.load(path)
        adapter = cls()
        adapter.source_mean = data["source_mean"]
        adapter.source_cov_whitening = data["source_cov_whitening"]
        adapter.target_mean = data["target_mean"]
        adapter.target_cov_coloring = data["target_cov_coloring"]
        adapter.fitted = bool(data["fitted"])
        return adapter
