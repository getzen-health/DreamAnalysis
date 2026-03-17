"""Meditation Depth Classifier from EEG signals.

Classifies meditation depth into 3 levels (reduced from 5 for better cross-subject accuracy):
  0: relaxed     — eyes-closed rest, alpha dominant, minimal meditation depth
  1: meditating  — sustained theta increase, alpha stabilization, reduced mental chatter
  2: deep        — strong theta dominance, beta suppressed, deep absorption

Scientific basis:
- Meditation increases frontal midline theta (Kubota et al., 2001)
- Alpha coherence increases during focused meditation (Travis & Shear, 2010)
- Advanced meditators show gamma power 25x higher than novices (Lutz et al., 2004)
- Theta/gamma coupling correlates with meditation expertise
- Default mode network (DMN) suppression = reduced delta/low-alpha

Meditation traditions mapped to EEG signatures:
- Focused attention (Shamatha): high alpha, moderate theta
- Open monitoring (Vipassana): high theta, wide alpha
- Non-dual (Dzogchen/Sahaj): gamma bursts, theta-gamma coupling
- Loving-kindness (Metta): alpha + gamma, high coherence

Reference: Lutz et al. (2004), Cahn & Polich (2006), Travis & Shear (2010)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from processing.eeg_processor import (
    extract_band_powers, extract_features, preprocess, compute_hjorth_parameters,
    spectral_entropy, compute_frontal_midline_theta
)

MEDITATION_DEPTHS = ["relaxed", "meditating", "deep"]


class MeditationClassifier:
    """EEG-based meditation depth classifier."""

    # Feature names used by the Random Forest classifier
    RF_FEATURE_NAMES = [
        "theta_power", "alpha_power", "beta_power",
        "theta_alpha_ratio", "alpha_beta_ratio", "frontal_midline_theta",
    ]

    def __init__(self, model_path: Optional[str] = None):
        self.model_type = "feature-based"
        self.sklearn_model = None
        self.feature_names = None
        self.scaler = None
        self.baseline_alpha = None
        self.baseline_theta = None
        # Random Forest components (Issue #45)
        self.rf_model = None
        self.rf_scaler = None
        # Track depth over session
        self._depth_history = []

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        if model_path.endswith(".pkl"):
            try:
                import joblib
                data = joblib.load(model_path)
                self.sklearn_model = data["model"]
                self.feature_names = data["feature_names"]
                self.scaler = data.get("scaler")
                self.model_type = "sklearn"
            except Exception:
                pass

    def calibrate(self, resting_eeg: np.ndarray, fs: float = 256.0):
        """Calibrate with eyes-closed resting EEG baseline."""
        processed = preprocess(resting_eeg, fs)
        bands = extract_band_powers(processed, fs)
        self.baseline_alpha = bands.get("alpha", 0.25)
        self.baseline_theta = bands.get("theta", 0.15)

    def predict(self, eeg: np.ndarray, fs: float = 256.0) -> Dict:
        """Classify meditation depth from EEG.

        Returns:
            Dict with 'depth', 'depth_index', 'meditation_score' (0-1),
            'confidence', 'tradition_match', and component scores.
        """
        # Try Random Forest first (Issue #45 — 86.7% accuracy on theta/alpha/beta)
        if self.rf_model is not None:
            try:
                return self._predict_rf(eeg, fs)
            except Exception:
                pass  # fall through to sklearn or heuristics

        if self.sklearn_model is not None:
            try:
                return self._predict_sklearn(eeg, fs)
            except Exception:
                pass  # fall through to feature-based

        # Extract single channel for band power analysis (multichannel handled by FMT below)
        signal = eeg[0] if eeg.ndim == 2 else eeg
        processed = preprocess(signal, fs)
        bands = extract_band_powers(processed, fs)
        hjorth = compute_hjorth_parameters(processed)
        se = spectral_entropy(processed, fs)

        alpha = bands.get("alpha", 0)
        beta = bands.get("beta", 0)
        theta = bands.get("theta", 0)
        delta = bands.get("delta", 0)

        base_alpha = self.baseline_alpha or 0.25
        base_theta = self.baseline_theta or 0.15

        # === Meditation Components ===

        # 1. Alpha Stabilization (relaxation foundation)
        alpha_ratio = alpha / (base_alpha + 1e-10)
        alpha_stability = float(np.clip(np.tanh(alpha_ratio - 0.5), 0, 1))

        # 2. Theta Elevation (deepening meditation)
        theta_increase = theta / (base_theta + 1e-10) - 1.0
        theta_depth = float(np.clip(np.tanh(theta_increase), 0, 1))

        # 3. Beta Quieting (reduced mental chatter)
        beta_quiet = float(np.clip(1.0 - np.tanh(beta * 5), 0, 1))

        # 4. Delta Management (avoiding sleep — some delta is OK)
        # Too much delta = sleep, not meditation
        delta_balance = float(np.clip(1.0 - np.tanh(delta * 2 - 0.5), 0, 1))

        # 5. Spectral Narrowing (focused meditation = lower entropy)
        spectral_focus = float(np.clip(1.0 - se, 0, 1))

        # 6. Signal Calmness (low Hjorth activity = calm brain)
        activity = hjorth.get("activity", 0.01) if isinstance(hjorth, dict) else 0.01
        calmness = float(np.clip(1.0 - np.tanh(activity * 5), 0, 1))

        # === Overall Meditation Score ===
        # NOTE: gamma_transcendence and theta_gamma_coupling were removed — on Muse 2,
        # AF7/AF8 gamma (30-100 Hz) is predominantly EMG (muscle) artifact, not neural
        # signal. Jaw clenching falsely inflated "deep" readings. Weights redistributed
        # to alpha_stability (+0.05), theta_depth (+0.10), beta_quiet (+0.05),
        # delta_balance (+0.05), spectral_focus (+0.05), calmness (+0.05). FMT applied
        # as additive boost below (the gold-standard meditation marker, Kubota 2001).
        meditation_score = float(np.clip(
            0.25 * alpha_stability +
            0.30 * theta_depth +
            0.20 * beta_quiet +
            0.10 * delta_balance +
            0.08 * spectral_focus +
            0.07 * calmness,
            0, 1
        ))

        # === Frontal Midline Theta (FMT) Boost ===
        # FMT at 4-8 Hz from AF7 is the gold-standard marker for meditation depth
        # (Kubota et al., 2001; Cahn & Polich, 2006). Applied after base score so
        # it can only increase depth readings when genuine theta is present.
        # AF7 = ch1 in BrainFlow Muse 2 channel order (TP9, AF7, AF8, TP10).
        fmt_power = 0.0
        fmt_de = 0.0
        try:
            if eeg.ndim == 2 and eeg.shape[0] >= 2:
                fmt = compute_frontal_midline_theta(eeg[1], fs)  # AF7 = ch1
            else:
                # Single-channel input: use it directly (already preprocessed above)
                fmt = compute_frontal_midline_theta(processed, fs)
            fmt_power = fmt.get("fmt_power", 0.0)
            fmt_de = fmt.get("fmt_de", 0.0)
            if fmt_power > 0.3:
                meditation_score = min(1.0, meditation_score + 0.15)
            if fmt_de > 1.5:
                meditation_score = min(1.0, meditation_score + 0.10)
        except Exception:
            pass  # FMT failure must not break meditation inference

        # Track depth
        self._depth_history.append(meditation_score)
        if len(self._depth_history) > 120:
            self._depth_history = self._depth_history[-120:]

        # Session duration effect (meditation deepens over time)
        session_minutes = len(self._depth_history) * 0.5  # ~30s per epoch
        session_bonus = float(np.clip(session_minutes / 20.0 * 0.1, 0, 0.1))

        adjusted_score = float(np.clip(meditation_score + session_bonus, 0, 1))

        # === Classify Depth (3-class) ===
        if adjusted_score >= 0.55:
            depth_idx = 2  # deep
        elif adjusted_score >= 0.25:
            depth_idx = 1  # meditating
        else:
            depth_idx = 0  # relaxed

        # === Tradition Match ===
        # non_dual and loving_kindness previously used gamma_transcendence; replaced
        # with FMT-based score (fmt_depth) as the reliable deep-state proxy.
        fmt_depth = float(np.clip(fmt_power / 0.5, 0, 1))  # normalised 0-1 at 0.5 FMT
        tradition_scores = {
            "focused_attention": float(alpha_stability * 0.5 + beta_quiet * 0.3 + spectral_focus * 0.2),
            "open_monitoring": float(theta_depth * 0.4 + alpha_stability * 0.3 + calmness * 0.3),
            "non_dual": float(fmt_depth * 0.4 + theta_depth * 0.3 + calmness * 0.3),
            "loving_kindness": float(alpha_stability * 0.3 + fmt_depth * 0.3 + calmness * 0.4),
        }
        best_tradition = max(tradition_scores, key=tradition_scores.get)

        # Confidence (3-class thresholds: 0.0 / 0.25 / 0.55 / 1.0)
        thresholds = [0.0, 0.25, 0.55, 1.0]
        mid = (thresholds[depth_idx] + thresholds[depth_idx + 1]) / 2
        dist = abs(adjusted_score - mid)
        range_size = thresholds[depth_idx + 1] - thresholds[depth_idx]
        confidence = float(np.clip(1.0 - dist / (range_size / 2 + 1e-10), 0.3, 0.95))

        return {
            "depth": MEDITATION_DEPTHS[depth_idx],
            "depth_index": depth_idx,
            "meditation_score": round(adjusted_score, 3),
            "confidence": round(confidence, 3),
            "tradition_match": best_tradition,
            "tradition_scores": {k: round(v, 3) for k, v in tradition_scores.items()},
            "session_minutes": round(session_minutes, 1),
            "components": {
                "alpha_stability": round(alpha_stability, 3),
                "theta_depth": round(theta_depth, 3),
                "beta_quiet": round(beta_quiet, 3),
                "delta_balance": round(delta_balance, 3),
                "fmt_power": round(fmt_power, 3),
                "fmt_de": round(fmt_de, 3),
                "spectral_focus": round(spectral_focus, 3),
                "calmness": round(calmness, 3),
            },
            "band_powers": bands,
        }

    def _predict_sklearn(self, eeg: np.ndarray, fs: float) -> Dict:
        signal = eeg[0] if eeg.ndim == 2 else eeg
        processed = preprocess(signal, fs)
        features = extract_features(processed, fs)
        bands = extract_band_powers(processed, fs)
        fv = np.array([features.get(k, 0.0) for k in self.feature_names]).reshape(1, -1)
        if self.scaler is not None:
            fv = self.scaler.transform(fv)

        probs = self.sklearn_model.predict_proba(fv)[0]
        depth_idx = int(np.argmax(probs))

        return {
            "depth": MEDITATION_DEPTHS[min(depth_idx, 4)],
            "depth_index": min(depth_idx, 4),
            "meditation_score": round(float(np.dot(probs, np.linspace(0, 1, len(probs)))), 3),
            "confidence": round(float(probs[depth_idx]), 3),
            "tradition_match": "focused_attention",
            "tradition_scores": {},
            "session_minutes": 0.0,
            "components": {},
            "band_powers": bands,
        }

    def get_session_stats(self) -> Dict:
        """Get meditation session statistics."""
        if not self._depth_history:
            return {"n_epochs": 0}

        scores = np.array(self._depth_history)
        return {
            "n_epochs": len(scores),
            "session_minutes": round(len(scores) * 0.5, 1),
            "avg_depth": round(float(np.mean(scores)), 3),
            "max_depth": round(float(np.max(scores)), 3),
            "time_in_deep": round(float(np.mean(scores >= 0.6)) * 100, 1),
            "deepening_trend": round(float(
                np.mean(scores[-10:]) - np.mean(scores[:10])
            ) if len(scores) >= 20 else 0.0, 3),
        }

    # ── Random Forest classifier (Issue #45) ──────────────────────────────────

    def _extract_rf_features(self, eeg: np.ndarray, fs: float) -> np.ndarray:
        """Extract the 6 features used by the Random Forest classifier.

        Features (in order matching RF_FEATURE_NAMES):
          theta_power, alpha_power, beta_power,
          theta_alpha_ratio, alpha_beta_ratio, frontal_midline_theta
        """
        signal = eeg[0] if eeg.ndim == 2 else eeg
        processed = preprocess(signal, fs)
        bands = extract_band_powers(processed, fs)
        theta = bands.get("theta", 1e-10)
        alpha = bands.get("alpha", 1e-10)
        beta = bands.get("beta", 1e-10)
        fmt = compute_frontal_midline_theta(processed, fs)
        return np.array([
            theta,
            alpha,
            beta,
            theta / max(alpha, 1e-10),
            alpha / max(beta, 1e-10),
            fmt.get("fmt_power", 0.0),
        ], dtype=np.float64)

    def _predict_rf(self, eeg: np.ndarray, fs: float) -> Dict:
        """Run inference using the trained Random Forest model."""
        signal = eeg[0] if eeg.ndim == 2 else eeg
        processed = preprocess(signal, fs)
        bands = extract_band_powers(processed, fs)
        fv = self._extract_rf_features(eeg, fs).reshape(1, -1)
        if self.rf_scaler is not None:
            fv = self.rf_scaler.transform(fv)
        probs = self.rf_model.predict_proba(fv)[0]
        depth_idx = int(np.argmax(probs))
        meditation_score = float(np.dot(probs, np.linspace(0, 1, len(probs))))
        return {
            "depth": MEDITATION_DEPTHS[min(depth_idx, len(MEDITATION_DEPTHS) - 1)],
            "depth_index": min(depth_idx, len(MEDITATION_DEPTHS) - 1),
            "meditation_score": round(meditation_score, 3),
            "confidence": round(float(probs[depth_idx]), 3),
            "tradition_match": "focused_attention",
            "tradition_scores": {},
            "session_minutes": 0.0,
            "components": {},
            "band_powers": bands,
            "model_type": "random_forest",
        }

    def train_rf(
        self,
        eeg_epochs: List[np.ndarray],
        labels: List[int],
        fs: float = 256.0,
    ) -> Dict:
        """Train a Random Forest classifier on labelled EEG epochs.

        Based on the 2025 clinical trial achieving 86.7% accuracy using
        RandomForest on theta/alpha/beta features from consumer EEG.

        Args:
            eeg_epochs: List of 1D EEG arrays, one per epoch.
            labels: Integer depth labels per epoch (0=relaxed, 1=meditating, 2=deep).
            fs: Sampling frequency in Hz.

        Returns:
            Dict with 'accuracy', 'n_samples', 'n_classes'.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score

        X = np.array([self._extract_rf_features(ep, fs) for ep in eeg_epochs])
        y = np.array(labels)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_scaled, y)

        cv_scores = cross_val_score(clf, X_scaled, y, cv=min(5, len(set(y))), scoring="accuracy")

        self.rf_model = clf
        self.rf_scaler = scaler
        self.model_type = "random_forest"

        return {
            "accuracy": round(float(cv_scores.mean()), 4),
            "accuracy_std": round(float(cv_scores.std()), 4),
            "n_samples": len(y),
            "n_classes": len(set(y)),
        }

    def save_rf(self, path: str) -> None:
        """Persist the trained Random Forest model to disk using joblib.

        Args:
            path: File path to save the model (e.g. 'models/saved/meditation_rf.pkl').
        """
        import joblib

        if self.rf_model is None:
            raise RuntimeError("No RF model trained yet. Call train_rf() first.")

        joblib.dump(
            {
                "rf_model": self.rf_model,
                "rf_scaler": self.rf_scaler,
                "feature_names": self.RF_FEATURE_NAMES,
            },
            path,
        )

    def load_rf(self, path: str) -> None:
        """Load a previously saved Random Forest model from disk.

        Args:
            path: File path to the saved model (joblib format).
        """
        import joblib

        data = joblib.load(path)
        self.rf_model = data["rf_model"]
        self.rf_scaler = data.get("rf_scaler")
        self.model_type = "random_forest"

    def train_from_session_data(
        self,
        sessions_dir: str = "ml/sessions",
        fs: float = 256.0,
    ) -> Optional[Dict]:
        """Train RF from .npz session files that contain meditation labels.

        Scans ``sessions_dir`` for .npz files that include a 'meditation_label'
        array. Each entry must align with the corresponding EEG epoch in the
        'eeg' array. Sessions without labels are silently skipped.

        Args:
            sessions_dir: Directory containing .npz session files.
            fs: Sampling frequency in Hz.

        Returns:
            Training result dict from train_rf(), or None if no labelled data found.
        """
        sessions_path = Path(sessions_dir)
        if not sessions_path.exists():
            return None

        all_epochs: List[np.ndarray] = []
        all_labels: List[int] = []

        for npz_file in sorted(sessions_path.glob("*.npz")):
            try:
                data = np.load(npz_file, allow_pickle=True)
                if "eeg" not in data or "meditation_label" not in data:
                    continue
                eeg_epochs = data["eeg"]           # shape: (n_epochs, n_samples)
                labels = data["meditation_label"]  # shape: (n_epochs,)
                for epoch, label in zip(eeg_epochs, labels):
                    # Skip epochs with missing/negative labels
                    if label is None or int(label) < 0:
                        continue
                    all_epochs.append(np.asarray(epoch, dtype=np.float64))
                    all_labels.append(int(label))
            except Exception:
                continue  # corrupt file — skip silently

        if len(all_epochs) < 10:
            return None  # not enough data to train meaningfully

        return self.train_rf(all_epochs, all_labels, fs=fs)
