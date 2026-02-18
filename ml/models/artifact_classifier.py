"""Supervised Artifact Classifier — ML-based artifact type detection.

Replaces/augments heuristic-based artifact detection with a trained classifier
that can identify 6 artifact types from EEG features:
  0: clean        — no artifact present
  1: eye_blink    — frontal slow positive deflection
  2: muscle_emg   — broadband high-frequency contamination
  3: electrode_pop — sudden voltage jump from contact loss
  4: motion       — high-amplitude transient from movement
  5: powerline    — 50/60 Hz interference

Can be trained on:
  - TUH Artifact Corpus (TUAR) labels
  - Synthetic data from noise_augmentation module
  - User-labeled artifact segments
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import signal as scipy_signal

# Numpy 2.x compat
_trapezoid = getattr(np, "trapezoid", np.trapz)

ARTIFACT_CLASSES = {
    0: "clean",
    1: "eye_blink",
    2: "muscle_emg",
    3: "electrode_pop",
    4: "motion",
    5: "powerline",
}

ARTIFACT_NAME_TO_ID = {v: k for k, v in ARTIFACT_CLASSES.items()}


def extract_artifact_features(segment: np.ndarray, fs: float = 256.0) -> np.ndarray:
    """Extract features specifically designed for artifact classification.

    24-dimensional feature vector capturing artifact-specific characteristics.

    Args:
        segment: 1D EEG segment (typically 1-2 seconds).
        fs: Sampling frequency.

    Returns:
        Feature vector (24 dims).
    """
    features = []
    n = len(segment)

    # --- Amplitude features ---
    features.append(np.std(segment))                  # 1. RMS amplitude
    features.append(np.ptp(segment))                  # 2. Peak-to-peak
    features.append(float(np.max(np.abs(segment))))   # 3. Max absolute

    # --- Derivative features (detect sudden jumps) ---
    diff1 = np.diff(segment)
    features.append(np.std(diff1))                    # 4. First derivative std
    features.append(float(np.max(np.abs(diff1))))     # 5. Max derivative (electrode pop)

    diff2 = np.diff(diff1)
    features.append(np.std(diff2))                    # 6. Second derivative std

    # --- Spectral features ---
    nperseg = min(len(segment), int(fs))
    freqs, psd = scipy_signal.welch(segment, fs=fs, nperseg=nperseg)
    total_power = _trapezoid(psd, freqs) + 1e-10

    # Band powers (relative)
    bands = [
        (0.5, 4),    # 7. Delta
        (4, 8),      # 8. Theta
        (8, 12),     # 9. Alpha
        (12, 30),    # 10. Beta
        (30, 50),    # 11. Low gamma
        (50, 100),   # 12. High gamma (if available)
    ]
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < min(hi, fs / 2))
        if mask.any():
            bp = _trapezoid(psd[mask], freqs[mask]) / total_power
        else:
            bp = 0.0
        features.append(bp)

    # --- Line noise features (50/60 Hz) ---
    for line_freq in [50.0, 60.0]:
        if line_freq < fs / 2:
            mask = (freqs >= line_freq - 2) & (freqs <= line_freq + 2)
            if mask.any():
                features.append(_trapezoid(psd[mask], freqs[mask]) / total_power)  # 13, 14
            else:
                features.append(0.0)
        else:
            features.append(0.0)

    # --- Temporal shape features ---
    # Kurtosis (eye blinks have high kurtosis)
    from scipy.stats import kurtosis, skew
    features.append(float(kurtosis(segment)))         # 15. Kurtosis
    features.append(float(skew(segment)))             # 16. Skewness

    # Zero crossing rate (muscle artifacts have high ZCR)
    zcr = np.sum(np.diff(np.sign(segment)) != 0) / len(segment)
    features.append(zcr)                              # 17. Zero crossing rate

    # --- Energy distribution ---
    # Ratio of max to median amplitude (spiky = artifact)
    median_amp = np.median(np.abs(segment)) + 1e-10
    features.append(float(np.max(np.abs(segment)) / median_amp))  # 18. Crest factor

    # Energy in first half vs second half (asymmetry)
    half = n // 2
    e1 = np.sum(segment[:half] ** 2) + 1e-10
    e2 = np.sum(segment[half:] ** 2) + 1e-10
    features.append(float(e1 / (e1 + e2)))            # 19. Temporal asymmetry

    # --- Hjorth parameters ---
    var0 = np.var(segment) + 1e-10
    var1 = np.var(diff1) + 1e-10
    var2 = np.var(diff2) + 1e-10
    features.append(np.sqrt(var0))                     # 20. Activity
    features.append(np.sqrt(var1 / var0))              # 21. Mobility
    features.append(np.sqrt(var2 / var1) / (np.sqrt(var1 / var0) + 1e-10))  # 22. Complexity

    # --- Spectral entropy ---
    psd_norm = psd / (np.sum(psd) + 1e-10)
    psd_pos = psd_norm[psd_norm > 0]
    spectral_entropy = -np.sum(psd_pos * np.log2(psd_pos + 1e-10))
    features.append(spectral_entropy)                  # 23. Spectral entropy

    # --- High frequency ratio (muscle indicator) ---
    hf_mask = freqs >= 30.0
    if hf_mask.any():
        hf_ratio = _trapezoid(psd[hf_mask], freqs[hf_mask]) / total_power
    else:
        hf_ratio = 0.0
    features.append(hf_ratio)                          # 24. HF ratio

    return np.array(features[:24], dtype=np.float64)


class ArtifactClassifier:
    """Supervised artifact type classifier.

    Uses LightGBM (preferred) or RandomForest to classify EEG segments
    into 6 artifact types. Can be trained on real labeled data (TUH) or
    synthetic augmented data.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = None
        self.feature_names = [
            "rms_amp", "peak_to_peak", "max_abs",
            "d1_std", "d1_max", "d2_std",
            "delta_power", "theta_power", "alpha_power",
            "beta_power", "low_gamma_power", "high_gamma_power",
            "line_50hz", "line_60hz",
            "kurtosis", "skewness", "zcr", "crest_factor",
            "temporal_asymmetry",
            "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
            "spectral_entropy", "hf_ratio",
        ]

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def load(self, model_path: str):
        """Load a trained artifact classifier."""
        data = joblib.load(model_path)
        self.model = data["model"]
        self.scaler = data.get("scaler")

    def save(self, model_path: str):
        """Save the artifact classifier."""
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "artifact_classes": ARTIFACT_CLASSES,
        }, model_path)

    def classify(
        self, segment: np.ndarray, fs: float = 256.0
    ) -> Dict:
        """Classify an EEG segment's artifact type.

        Args:
            segment: 1D EEG segment.
            fs: Sampling frequency.

        Returns:
            Dict with 'artifact_type', 'confidence', 'probabilities'.
        """
        features = extract_artifact_features(segment, fs)
        features = features.reshape(1, -1)

        if self.scaler is not None:
            features = self.scaler.transform(features)

        if self.model is not None:
            proba = self.model.predict_proba(features)[0]
            pred = int(np.argmax(proba))
            return {
                "artifact_type": ARTIFACT_CLASSES[pred],
                "artifact_id": pred,
                "confidence": float(proba[pred]),
                "probabilities": {
                    ARTIFACT_CLASSES[i]: float(p)
                    for i, p in enumerate(proba)
                },
            }

        # Fallback: heuristic classification
        return self._heuristic_classify(features[0], fs)

    def classify_signal(
        self, signal: np.ndarray, fs: float = 256.0, window_sec: float = 1.0
    ) -> List[Dict]:
        """Classify artifacts across entire signal using sliding windows.

        Args:
            signal: 1D EEG signal.
            fs: Sampling frequency.
            window_sec: Classification window size in seconds.

        Returns:
            List of per-window classification results.
        """
        window_samples = int(window_sec * fs)
        step = window_samples // 2  # 50% overlap
        results = []

        for start in range(0, len(signal) - window_samples + 1, step):
            segment = signal[start:start + window_samples]
            result = self.classify(segment, fs)
            result["start_sec"] = start / fs
            result["end_sec"] = (start + window_samples) / fs
            results.append(result)

        return results

    def get_clean_mask(
        self, signal: np.ndarray, fs: float = 256.0, window_sec: float = 1.0
    ) -> np.ndarray:
        """Get a boolean mask indicating clean (artifact-free) samples.

        Args:
            signal: 1D EEG signal.
            fs: Sampling frequency.
            window_sec: Classification window size.

        Returns:
            Boolean array same length as signal (True = clean).
        """
        results = self.classify_signal(signal, fs, window_sec)
        mask = np.ones(len(signal), dtype=bool)

        for result in results:
            if result["artifact_type"] != "clean":
                s = int(result["start_sec"] * fs)
                e = int(result["end_sec"] * fs)
                mask[s:e] = False

        return mask

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
    ) -> Dict:
        """Train the artifact classifier.

        Args:
            X: Feature matrix (n_samples, 24).
            y: Labels 0-5 corresponding to ARTIFACT_CLASSES.
            test_size: Fraction for test split.

        Returns:
            Training results dict.
        """
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, accuracy_score

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )

        # Try LightGBM first, fall back to RandomForest
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=8,
                num_leaves=63,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
            )
            model_type = "LightGBM"
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            model_type = "RandomForest"

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)

        present_classes = sorted(set(y_test) | set(y_pred))
        present_names = [ARTIFACT_CLASSES[c] for c in present_classes]
        report = classification_report(
            y_test, y_pred, labels=present_classes,
            target_names=present_names, output_dict=True,
        )

        return {
            "model_type": model_type,
            "accuracy": float(accuracy),
            "cv_accuracy": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "per_class": report,
        }

    def train_on_synthetic(
        self, n_samples: int = 5000, fs: float = 256.0, epoch_sec: float = 1.0
    ) -> Dict:
        """Train on synthetic augmented data when real labeled data unavailable.

        Generates clean EEG + applies specific artifact types to create
        labeled training data.

        Returns:
            Training results dict.
        """
        from simulation.eeg_simulator import simulate_eeg
        from processing.noise_augmentation import (
            add_gaussian_noise, add_electrode_drift, add_motion_artifact,
            add_powerline_noise, add_emg_contamination, add_eye_blink,
        )

        n_samples_per_class = n_samples // 6
        n_signal = int(epoch_sec * fs)
        X_all, y_all = [], []

        states = ["rest", "focus", "meditation", "deep_sleep", "rem", "stress"]

        for _ in range(n_samples_per_class):
            state = np.random.choice(states)
            result = simulate_eeg(state=state, duration=epoch_sec, fs=fs)
            clean = np.array(result["signals"][0])[:n_signal]

            # Class 0: Clean
            features = extract_artifact_features(clean, fs)
            X_all.append(features)
            y_all.append(0)

            # Class 1: Eye blink
            blinked = add_eye_blink(clean.copy(), fs, n_blinks=1, amplitude=150.0)
            features = extract_artifact_features(blinked, fs)
            X_all.append(features)
            y_all.append(1)

            # Class 2: Muscle EMG
            emg = add_emg_contamination(clean.copy(), fs, intensity=0.4)
            features = extract_artifact_features(emg, fs)
            X_all.append(features)
            y_all.append(2)

            # Class 3: Electrode pop
            popped = clean.copy()
            pop_pos = np.random.randint(n_signal // 4, 3 * n_signal // 4)
            popped[pop_pos:pop_pos + 3] += np.random.choice([-1, 1]) * 300
            features = extract_artifact_features(popped, fs)
            X_all.append(features)
            y_all.append(3)

            # Class 4: Motion artifact
            moved = add_motion_artifact(clean.copy(), fs, n_artifacts=1, max_amplitude=250.0)
            features = extract_artifact_features(moved, fs)
            X_all.append(features)
            y_all.append(4)

            # Class 5: Powerline
            powerlined = add_powerline_noise(clean.copy(), fs, amplitude=8.0)
            features = extract_artifact_features(powerlined, fs)
            X_all.append(features)
            y_all.append(5)

        X = np.array(X_all)
        y = np.array(y_all)

        print(f"  Generated {len(y)} synthetic artifact samples "
              f"({n_samples_per_class} per class)")

        return self.train(X, y)

    def _heuristic_classify(
        self, features: np.ndarray, fs: float
    ) -> Dict:
        """Fallback heuristic classification when no trained model available."""
        # Feature indices
        kurtosis_val = features[14]
        hf_ratio = features[23]
        d1_max = features[4]
        line_50 = features[12]
        line_60 = features[13]
        crest = features[17]

        scores = {
            "clean": 0.5,
            "eye_blink": 0.0,
            "muscle_emg": 0.0,
            "electrode_pop": 0.0,
            "motion": 0.0,
            "powerline": 0.0,
        }

        if abs(kurtosis_val) > 5.0:
            scores["eye_blink"] += 0.4
        if hf_ratio > 0.35:
            scores["muscle_emg"] += 0.4
        if d1_max > 100:
            scores["electrode_pop"] += 0.4
        if crest > 10:
            scores["motion"] += 0.3
        if line_50 > 0.15 or line_60 > 0.15:
            scores["powerline"] += 0.4

        best = max(scores, key=scores.get)
        total = sum(scores.values()) + 1e-10

        return {
            "artifact_type": best,
            "artifact_id": ARTIFACT_NAME_TO_ID[best],
            "confidence": scores[best] / total,
            "probabilities": {k: v / total for k, v in scores.items()},
        }
