"""Per-User EEG Calibration System.

Everyone's brain produces different baseline amplitudes. Without calibration,
a feature value of "alpha=12uV" means nothing — it could be high for one person
and low for another.

Calibration Protocol (2 minutes total):
1. Eyes Open Rest (30s)   — baseline alpha suppression
2. Eyes Closed Rest (30s) — baseline alpha peak
3. Focused Counting (30s) — baseline beta activation
4. Relaxed Breathing (30s) — baseline theta/alpha balance

From these 4 conditions, we compute per-user z-score normalization parameters
for all band powers and derived features. After calibration, feature values
are expressed as "standard deviations from YOUR baseline" rather than raw uV.

This is what makes the difference between a toy and a real BCI system.
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from processing.eeg_processor import extract_band_powers, extract_features, preprocess


CALIBRATION_DIR = Path(__file__).parent.parent / "data" / "calibrations"
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

CALIBRATION_STEPS = [
    {
        "name": "eyes_open",
        "instruction": "Sit still with eyes OPEN. Look at a fixed point.",
        "duration": 30,
        "purpose": "Baseline with visual input (alpha suppressed)",
    },
    {
        "name": "eyes_closed",
        "instruction": "Close your eyes and relax. Don't try to think about anything.",
        "duration": 30,
        "purpose": "Baseline alpha peak (most people show alpha increase)",
    },
    {
        "name": "focused_task",
        "instruction": "Count backwards from 100 by 7s (100, 93, 86...). Keep eyes open.",
        "duration": 30,
        "purpose": "Baseline cognitive load (beta increase, alpha decrease)",
    },
    {
        "name": "relaxed_breathing",
        "instruction": "Close your eyes. Breathe slowly: 4s in, 6s out.",
        "duration": 30,
        "purpose": "Baseline relaxation (theta/alpha increase)",
    },
]


class UserCalibration:
    """Stores per-user baseline statistics for z-score normalization."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.is_calibrated = False
        self.calibrated_at = None

        # Per-condition band power stats
        self.condition_stats = {}  # condition_name → {band → {mean, std}}

        # Global baseline (across all conditions)
        self.global_band_means = {}   # band → mean
        self.global_band_stds = {}    # band → std

        # Feature-level stats (17 features)
        self.feature_means = None  # np.array of shape (17,)
        self.feature_stds = None   # np.array of shape (17,)

        # Derived thresholds
        self.alpha_peak_freq = None     # Individual alpha frequency (IAF)
        self.alpha_reactivity = None    # Eyes closed / eyes open alpha ratio
        self.beta_reactivity = None     # Focused / resting beta ratio
        self.theta_alpha_ratio_rest = None  # Resting theta/alpha

    def normalize_band_powers(self, band_powers: Dict[str, float]) -> Dict[str, float]:
        """Convert raw band powers to z-scores relative to user's baseline.

        Returns values in standard deviations from baseline.
        z=0 means "exactly your baseline", z=2 means "2 SDs above your normal".
        """
        if not self.is_calibrated:
            return band_powers

        normalized = {}
        for band, value in band_powers.items():
            mean = self.global_band_means.get(band, 0)
            std = self.global_band_stds.get(band, 1)
            std = max(std, 1e-6)  # Prevent division by zero
            normalized[band] = (value - mean) / std

        return normalized

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Convert raw 17-feature vector to z-scores relative to user's baseline."""
        if not self.is_calibrated or self.feature_means is None:
            return features

        features = np.array(features, dtype=float)
        stds = np.maximum(self.feature_stds, 1e-6)
        return (features - self.feature_means) / stds

    def get_condition_baseline(self, condition: str) -> Optional[Dict[str, float]]:
        """Get band power means for a specific calibration condition."""
        if condition in self.condition_stats:
            return {band: stats["mean"] for band, stats in self.condition_stats[condition].items()}
        return None

    def save(self):
        """Save calibration to disk."""
        filepath = CALIBRATION_DIR / f"{self.user_id}.json"
        data = {
            "user_id": self.user_id,
            "is_calibrated": self.is_calibrated,
            "calibrated_at": self.calibrated_at,
            "condition_stats": self.condition_stats,
            "global_band_means": self.global_band_means,
            "global_band_stds": self.global_band_stds,
            "feature_means": self.feature_means.tolist() if self.feature_means is not None else None,
            "feature_stds": self.feature_stds.tolist() if self.feature_stds is not None else None,
            "alpha_peak_freq": self.alpha_peak_freq,
            "alpha_reactivity": self.alpha_reactivity,
            "beta_reactivity": self.beta_reactivity,
            "theta_alpha_ratio_rest": self.theta_alpha_ratio_rest,
        }
        filepath.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, user_id: str) -> "UserCalibration":
        """Load calibration from disk. Returns uncalibrated if not found."""
        cal = cls(user_id)
        filepath = CALIBRATION_DIR / f"{user_id}.json"

        if filepath.exists():
            data = json.loads(filepath.read_text())
            cal.is_calibrated = data.get("is_calibrated", False)
            cal.calibrated_at = data.get("calibrated_at")
            cal.condition_stats = data.get("condition_stats", {})
            cal.global_band_means = data.get("global_band_means", {})
            cal.global_band_stds = data.get("global_band_stds", {})
            if data.get("feature_means") is not None:
                cal.feature_means = np.array(data["feature_means"])
                cal.feature_stds = np.array(data["feature_stds"])
            cal.alpha_peak_freq = data.get("alpha_peak_freq")
            cal.alpha_reactivity = data.get("alpha_reactivity")
            cal.beta_reactivity = data.get("beta_reactivity")
            cal.theta_alpha_ratio_rest = data.get("theta_alpha_ratio_rest")

        return cal


class CalibrationRunner:
    """Runs the calibration protocol and computes per-user baselines."""

    def __init__(self, fs: int = 256):
        self.fs = fs
        self.condition_data = {}  # condition → list of (band_powers, features)

    def add_epoch(self, condition: str, signal: np.ndarray):
        """Add a 4-second epoch of EEG data for a calibration condition.

        Call this repeatedly as data comes in during each calibration step.
        """
        if condition not in self.condition_data:
            self.condition_data[condition] = {"bands": [], "features": []}

        processed = preprocess(signal, self.fs)
        bands = extract_band_powers(processed, self.fs)
        features_dict = extract_features(processed, self.fs)
        # Convert feature dict to sorted array for consistent ordering
        feature_values = np.array([v for _, v in sorted(features_dict.items())])

        self.condition_data[condition]["bands"].append(bands)
        self.condition_data[condition]["features"].append(feature_values)

    def compute_calibration(self, user_id: str) -> UserCalibration:
        """Compute calibration from collected epochs.

        Should be called after all 4 conditions are complete.
        """
        cal = UserCalibration(user_id)

        if not self.condition_data:
            return cal

        all_bands = []
        all_features = []

        # Per-condition statistics
        for condition, data in self.condition_data.items():
            if not data["bands"]:
                continue

            condition_bands = {}
            for band in BANDS:
                values = [b.get(band, 0) for b in data["bands"]]
                condition_bands[band] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
                all_bands.extend([(band, v) for v in values])

            cal.condition_stats[condition] = condition_bands

            if data["features"]:
                feature_matrix = np.array(data["features"])
                all_features.append(feature_matrix)

        # Global band statistics (across all conditions)
        for band in BANDS:
            values = [v for b, v in all_bands if b == band]
            if values:
                cal.global_band_means[band] = float(np.mean(values))
                cal.global_band_stds[band] = float(np.std(values))

        # Global feature statistics
        if all_features:
            all_feat = np.vstack(all_features)
            cal.feature_means = np.mean(all_feat, axis=0)
            cal.feature_stds = np.std(all_feat, axis=0)

        # Derived metrics
        ec = cal.condition_stats.get("eyes_closed", {})
        eo = cal.condition_stats.get("eyes_open", {})
        ft = cal.condition_stats.get("focused_task", {})

        # Alpha reactivity: eyes closed alpha / eyes open alpha
        ec_alpha = ec.get("alpha", {}).get("mean", 0)
        eo_alpha = eo.get("alpha", {}).get("mean", 0)
        if eo_alpha > 0:
            cal.alpha_reactivity = ec_alpha / eo_alpha
        else:
            cal.alpha_reactivity = 1.0

        # Beta reactivity: focused beta / resting beta
        ft_beta = ft.get("beta", {}).get("mean", 0)
        eo_beta = eo.get("beta", {}).get("mean", 0)
        if eo_beta > 0:
            cal.beta_reactivity = ft_beta / eo_beta
        else:
            cal.beta_reactivity = 1.0

        # Theta/alpha ratio at rest
        eo_theta = eo.get("theta", {}).get("mean", 0)
        if eo_alpha > 0:
            cal.theta_alpha_ratio_rest = eo_theta / eo_alpha
        else:
            cal.theta_alpha_ratio_rest = 1.0

        cal.is_calibrated = True
        cal.calibrated_at = time.time()
        cal.save()

        return cal

    def get_progress(self) -> Dict:
        """Get calibration progress."""
        total_steps = len(CALIBRATION_STEPS)
        completed = sum(1 for step in CALIBRATION_STEPS
                        if step["name"] in self.condition_data
                        and len(self.condition_data[step["name"]]["bands"]) >= 3)

        return {
            "total_steps": total_steps,
            "completed_steps": completed,
            "is_complete": completed >= total_steps,
            "steps": [
                {
                    "name": step["name"],
                    "instruction": step["instruction"],
                    "duration": step["duration"],
                    "epochs_collected": len(self.condition_data.get(step["name"], {}).get("bands", [])),
                    "complete": step["name"] in self.condition_data
                               and len(self.condition_data.get(step["name"], {}).get("bands", [])) >= 3,
                }
                for step in CALIBRATION_STEPS
            ],
        }
