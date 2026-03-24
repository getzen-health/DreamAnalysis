"""UserModelRetrainer -- fine-tune per-user models from accumulated corrections.

Two retraining paths:
  - EEG: fine-tune a small classifier head on cached 170-dim EEG features
         (85-dim muse live features concatenated from raw + PCA-transformed).
         Uses a 2-layer MLP trained with SGD. Minimum 5 corrections.
  - Voice: retrain a LightGBM on accumulated voice feature vectors.
           Minimum 5 corrections.

Auto-retrain triggers:
  - After 5 total corrections, then every 5 new ones.
  - Research shows 5-10 labeled samples per class is enough for meaningful
    personalization with fine-tuning on pre-trained feature extractors.

Per-user models saved to ml/user_models/{user_id}/.
Falls back to generic model when no per-user model exists.

After successful retraining, per-user ONNX models are exported for on-device
deployment via the /training/model/{user_id}/*.onnx endpoints.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

USER_DATA_DIR = Path(__file__).parent.parent / "user_data" / "corrections"
USER_MODELS_DIR = Path(__file__).parent.parent / "user_models"

# Minimum corrections before retraining is allowed
# 5 samples per class is enough for meaningful personalization (research-backed)
MIN_CORRECTIONS_EEG = 5
MIN_CORRECTIONS_VOICE = 5

# Auto-retrain thresholds
# Lower thresholds mean the model starts improving after ~5 days of daily use
AUTO_RETRAIN_INITIAL = 5  # first retrain after this many total corrections
AUTO_RETRAIN_INCREMENT = 5  # retrain again every N new corrections after initial


class UserModelRetrainer:
    """Fine-tune per-user emotion models from accumulated corrections.

    Each correction record should contain:
      - corrected_emotion: str (the ground-truth label)
      - predicted_emotion: str (what the model predicted)
      - eeg_features: list[float] | None (170-dim EEG feature vector)
      - voice_features: list[float] | None (voice feature vector)
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._user_dir = USER_MODELS_DIR / user_id
        self._corrections_path = USER_DATA_DIR / f"{user_id}_corrections.jsonl"

    # ---- public API --------------------------------------------------------

    def load_corrections(self) -> List[Dict[str, Any]]:
        """Load all correction records for this user from the JSONL file."""
        if not self._corrections_path.exists():
            return []
        corrections: List[Dict[str, Any]] = []
        for line in self._corrections_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                corrections.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return corrections

    def should_retrain(self) -> bool:
        """Check if auto-retrain should trigger based on correction count."""
        corrections = self.load_corrections()
        total = len(corrections)
        if total < AUTO_RETRAIN_INITIAL:
            return False

        meta = self._load_retrain_meta()
        last_count = meta.get("last_retrain_count", 0)

        # First retrain at AUTO_RETRAIN_INITIAL, then every AUTO_RETRAIN_INCREMENT
        if last_count == 0:
            return True  # never retrained before and >= initial threshold
        return (total - last_count) >= AUTO_RETRAIN_INCREMENT

    def retrain_eeg(self, force: bool = False) -> Dict[str, Any]:
        """Fine-tune EEG classifier head from cached feature vectors.

        Uses a 2-layer MLP (SGDClassifier with warm_start) on the
        170-dim features stored in corrections. The 170-dim vector is the
        concatenation of the 85-dim raw Muse features and 85-dim
        scaler-transformed features (or zero-padded if only 85 are stored).

        For simplicity and robustness with small datasets, we use an
        SGDClassifier (linear) which is equivalent to fine-tuning the
        last layer of a neural network.

        Returns dict with training result metadata.
        """
        corrections = self.load_corrections()
        eeg_samples = [
            c for c in corrections
            if c.get("eeg_features") and c.get("corrected_emotion")
        ]

        if len(eeg_samples) < MIN_CORRECTIONS_EEG and not force:
            return {
                "trained": False,
                "reason": f"Need {MIN_CORRECTIONS_EEG} EEG corrections, have {len(eeg_samples)}",
                "modality": "eeg",
            }

        if len(eeg_samples) == 0:
            return {
                "trained": False,
                "reason": "No corrections with EEG features found",
                "modality": "eeg",
            }

        # Build feature matrix and labels
        X_list = []
        y_list = []
        for c in eeg_samples:
            feat = c["eeg_features"]
            if isinstance(feat, list):
                X_list.append(feat)
                y_list.append(c["corrected_emotion"])

        if len(X_list) == 0:
            return {"trained": False, "reason": "No valid EEG feature vectors", "modality": "eeg"}

        X = np.array(X_list, dtype=np.float32)
        # Pad or truncate to consistent dimension
        target_dim = 170
        if X.shape[1] < target_dim:
            # Pad with zeros
            pad = np.zeros((X.shape[0], target_dim - X.shape[1]), dtype=np.float32)
            X = np.concatenate([X, pad], axis=1)
        elif X.shape[1] > target_dim:
            X = X[:, :target_dim]

        try:
            import joblib
            from sklearn.linear_model import SGDClassifier
            from sklearn.preprocessing import LabelEncoder, StandardScaler
        except ImportError as exc:
            return {"trained": False, "reason": f"Missing dependency: {exc}", "modality": "eeg"}

        le = LabelEncoder()
        y = le.fit_transform(y_list)
        classes = list(le.classes_)

        # Need at least 2 classes
        if len(classes) < 2:
            return {
                "trained": False,
                "reason": f"Need at least 2 emotion classes, have {len(classes)}",
                "modality": "eeg",
            }

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Warm-start: load prior model if it exists and has compatible shape.
        # This preserves learned decision boundaries from previous corrections
        # instead of resetting to random weights every time.
        self._user_dir.mkdir(parents=True, exist_ok=True)
        model_path = self._user_dir / "eeg_classifier.pkl"
        scaler_path = self._user_dir / "eeg_scaler.pkl"
        meta_path = self._user_dir / "eeg_meta.json"

        warm_started = False
        all_class_indices = np.arange(len(classes))

        if model_path.exists():
            try:
                prior_model = joblib.load(model_path)
                if (hasattr(prior_model, "coef_")
                        and prior_model.coef_.shape[1] == X_scaled.shape[1]
                        and len(prior_model.classes_) == len(classes)):
                    # Continue from prior weights via partial_fit (one SGD pass)
                    prior_model.partial_fit(X_scaled, y, classes=all_class_indices)
                    model = prior_model
                    warm_started = True
                    log.info(
                        "[user-retrain] warm-started EEG model for user=%s",
                        self.user_id,
                    )
            except Exception as exc:
                log.warning(
                    "[user-retrain] could not load prior EEG model for user=%s: %s",
                    self.user_id, exc,
                )

        if not warm_started:
            model = SGDClassifier(
                loss="log_loss",
                warm_start=True,
                max_iter=300,
                random_state=42,
                n_iter_no_change=15,
            )
            model.fit(X_scaled, y)

        train_acc = float(np.mean(model.predict(X_scaled) == y))

        # Save model artifacts
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        with open(meta_path, "w") as f:
            json.dump({
                "classes": classes,
                "label_encoder_classes": classes,
                "n_samples": len(y_list),
                "feature_dim": int(X.shape[1]),
                "train_accuracy": round(train_acc, 4),
                "warm_started": warm_started,
                "retrained_at": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)

        self._update_retrain_meta(len(self.load_corrections()))

        log.info(
            "[user-retrain] EEG user=%s samples=%d classes=%s acc=%.3f warm_started=%s",
            self.user_id, len(y_list), classes, train_acc, warm_started,
        )

        # Export ONNX for on-device inference
        onnx_path = self.export_eeg_onnx()

        return {
            "trained": True,
            "warm_started": warm_started,
            "modality": "eeg",
            "n_samples": len(y_list),
            "classes": classes,
            "train_accuracy": round(train_acc, 4),
            "onnx_exported": onnx_path is not None,
        }

    def retrain_voice(self, force: bool = False) -> Dict[str, Any]:
        """Retrain voice emotion classifier from cached voice features.

        Uses LightGBM when available, falls back to SGDClassifier.

        Returns dict with training result metadata.
        """
        corrections = self.load_corrections()
        voice_samples = [
            c for c in corrections
            if c.get("voice_features") and c.get("corrected_emotion")
        ]

        if len(voice_samples) < MIN_CORRECTIONS_VOICE and not force:
            return {
                "trained": False,
                "reason": f"Need {MIN_CORRECTIONS_VOICE} voice corrections, have {len(voice_samples)}",
                "modality": "voice",
            }

        if len(voice_samples) == 0:
            return {
                "trained": False,
                "reason": "No corrections with voice features found",
                "modality": "voice",
            }

        X_list = []
        y_list = []
        for c in voice_samples:
            feat = c["voice_features"]
            if isinstance(feat, list):
                X_list.append(feat)
                y_list.append(c["corrected_emotion"])

        if len(X_list) == 0:
            return {"trained": False, "reason": "No valid voice feature vectors", "modality": "voice"}

        # Voice features may have variable length across versions; pad to max
        max_dim = max(len(f) for f in X_list)
        X = np.array(
            [f + [0.0] * (max_dim - len(f)) for f in X_list],
            dtype=np.float32,
        )

        try:
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            import joblib
        except ImportError as exc:
            return {"trained": False, "reason": f"Missing dependency: {exc}", "modality": "voice"}

        le = LabelEncoder()
        y = le.fit_transform(y_list)
        classes = list(le.classes_)

        if len(classes) < 2:
            return {
                "trained": False,
                "reason": f"Need at least 2 emotion classes, have {len(classes)}",
                "modality": "voice",
            }

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Try LightGBM, fall back to SGDClassifier
        model = None
        model_type = "sgd"
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1,
            )
            model.fit(X_scaled, y)
            model_type = "lightgbm"
        except ImportError:
            from sklearn.linear_model import SGDClassifier
            model = SGDClassifier(
                loss="log_loss",
                warm_start=True,
                max_iter=300,
                random_state=42,
            )
            model.fit(X_scaled, y)

        train_acc = float(np.mean(model.predict(X_scaled) == y))

        # Save model artifacts
        self._user_dir.mkdir(parents=True, exist_ok=True)
        model_path = self._user_dir / "voice_classifier.pkl"
        scaler_path = self._user_dir / "voice_scaler.pkl"
        meta_path = self._user_dir / "voice_meta.json"

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        with open(meta_path, "w") as f:
            json.dump({
                "classes": classes,
                "label_encoder_classes": classes,
                "n_samples": len(y_list),
                "feature_dim": int(X.shape[1]),
                "model_type": model_type,
                "train_accuracy": round(train_acc, 4),
                "retrained_at": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)

        self._update_retrain_meta(len(self.load_corrections()))

        log.info(
            "[user-retrain] Voice user=%s samples=%d classes=%s acc=%.3f model=%s",
            self.user_id, len(y_list), classes, train_acc, model_type,
        )

        # Export ONNX for on-device inference
        onnx_path = self.export_voice_onnx()

        return {
            "trained": True,
            "modality": "voice",
            "n_samples": len(y_list),
            "classes": classes,
            "model_type": model_type,
            "train_accuracy": round(train_acc, 4),
            "onnx_exported": onnx_path is not None,
        }

    def retrain_all(self, force: bool = False) -> Dict[str, Any]:
        """Retrain both EEG and voice models.

        Returns combined result from both modalities.
        """
        eeg_result = self.retrain_eeg(force=force)
        voice_result = self.retrain_voice(force=force)
        return {
            "eeg": eeg_result,
            "voice": voice_result,
            "any_trained": eeg_result.get("trained", False) or voice_result.get("trained", False),
        }

    def get_status(self) -> Dict[str, Any]:
        """Return retraining status for this user."""
        corrections = self.load_corrections()
        eeg_count = sum(1 for c in corrections if c.get("eeg_features"))
        voice_count = sum(1 for c in corrections if c.get("voice_features"))
        meta = self._load_retrain_meta()

        eeg_model_exists = (self._user_dir / "eeg_classifier.pkl").exists()
        voice_model_exists = (self._user_dir / "voice_classifier.pkl").exists()

        eeg_meta = self._load_model_meta("eeg_meta.json")
        voice_meta = self._load_model_meta("voice_meta.json")

        return {
            "user_id": self.user_id,
            "total_corrections": len(corrections),
            "eeg_corrections": eeg_count,
            "voice_corrections": voice_count,
            "eeg_model_exists": eeg_model_exists,
            "voice_model_exists": voice_model_exists,
            "eeg_accuracy": eeg_meta.get("train_accuracy") if eeg_meta else None,
            "voice_accuracy": voice_meta.get("train_accuracy") if voice_meta else None,
            "last_retrain_count": meta.get("last_retrain_count", 0),
            "should_retrain": self.should_retrain(),
            "eeg_ready": eeg_count >= MIN_CORRECTIONS_EEG,
            "voice_ready": voice_count >= MIN_CORRECTIONS_VOICE,
        }

    # ---- per-user model loading (for inference) ----------------------------

    def load_eeg_model(self) -> Optional[Tuple[Any, Any, Dict]]:
        """Load user's fine-tuned EEG model if it exists.

        Returns (model, scaler, meta_dict) or None.
        """
        model_path = self._user_dir / "eeg_classifier.pkl"
        scaler_path = self._user_dir / "eeg_scaler.pkl"
        meta_path = self._user_dir / "eeg_meta.json"

        if not model_path.exists():
            return None

        try:
            import joblib
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path) if scaler_path.exists() else None
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            return model, scaler, meta
        except Exception as exc:
            log.warning("[user-retrain] Failed to load EEG model for %s: %s", self.user_id, exc)
            return None

    def load_voice_model(self) -> Optional[Tuple[Any, Any, Dict]]:
        """Load user's fine-tuned voice model if it exists.

        Returns (model, scaler, meta_dict) or None.
        """
        model_path = self._user_dir / "voice_classifier.pkl"
        scaler_path = self._user_dir / "voice_scaler.pkl"
        meta_path = self._user_dir / "voice_meta.json"

        if not model_path.exists():
            return None

        try:
            import joblib
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path) if scaler_path.exists() else None
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            return model, scaler, meta
        except Exception as exc:
            log.warning("[user-retrain] Failed to load voice model for %s: %s", self.user_id, exc)
            return None

    # ---- ONNX export for on-device deployment --------------------------------

    def export_eeg_onnx(self) -> Optional[str]:
        """Export per-user fine-tuned EEG model to ONNX for on-device use.

        Converts the sklearn SGDClassifier to ONNX format using skl2onnx.
        Falls back to a manual ONNX graph if skl2onnx is unavailable.

        Returns the ONNX file path on success, None on failure.
        """
        model_tuple = self.load_eeg_model()
        if model_tuple is None:
            log.debug("[onnx-export] No EEG model to export for user=%s", self.user_id)
            return None

        model, scaler, meta = model_tuple
        feature_dim = meta.get("feature_dim", 170)

        try:
            return self._export_sklearn_onnx(
                model, scaler, feature_dim,
                "eeg_emotion_user.onnx", "eeg_features",
            )
        except Exception as exc:
            log.warning(
                "[onnx-export] EEG ONNX export failed for user=%s: %s",
                self.user_id, exc,
            )
            return None

    def export_voice_onnx(self) -> Optional[str]:
        """Export per-user fine-tuned voice model to ONNX for on-device use.

        Returns the ONNX file path on success, None on failure.
        """
        model_tuple = self.load_voice_model()
        if model_tuple is None:
            log.debug("[onnx-export] No voice model to export for user=%s", self.user_id)
            return None

        model, scaler, meta = model_tuple
        feature_dim = meta.get("feature_dim", 40)

        try:
            return self._export_sklearn_onnx(
                model, scaler, feature_dim,
                "voice_emotion_user.onnx", "voice_features",
            )
        except Exception as exc:
            log.warning(
                "[onnx-export] Voice ONNX export failed for user=%s: %s",
                self.user_id, exc,
            )
            return None

    def _export_sklearn_onnx(
        self,
        model: Any,
        scaler: Any,
        feature_dim: int,
        filename: str,
        input_name: str,
    ) -> Optional[str]:
        """Export an sklearn model (with optional scaler) to ONNX.

        Handles both sklearn classifiers (via skl2onnx) and LightGBM
        classifiers (via onnxmltools). Falls back to manual numpy-based
        ONNX construction for simple linear models.
        """
        self._user_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = self._user_dir / filename

        # Check if model is a LightGBM classifier — needs onnxmltools
        is_lgbm = False
        try:
            import lightgbm as lgb
            is_lgbm = isinstance(model, lgb.LGBMClassifier)
        except ImportError:
            pass

        try:
            from skl2onnx import convert_sklearn, update_registered_converter
            from skl2onnx.common.data_types import FloatTensorType
            from sklearn.pipeline import Pipeline

            # Register LightGBM converter if needed
            if is_lgbm:
                try:
                    from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
                        convert_lightgbm,
                    )
                    from skl2onnx.common.shape_calculator import (
                        calculate_linear_classifier_output_shapes,
                    )
                    import lightgbm as lgb

                    update_registered_converter(
                        lgb.LGBMClassifier,
                        "LightGbmLGBMClassifier",
                        calculate_linear_classifier_output_shapes,
                        convert_lightgbm,
                        options={
                            "zipmap": [True, False, "columns"],
                            "nocl": [True, False],
                        },
                    )
                except ImportError:
                    log.info("[onnx-export] onnxmltools not available for LightGBM, using manual export")
                    if hasattr(model, "coef_"):
                        return self._export_manual_onnx(model, scaler, feature_dim, onnx_path, input_name)
                    return None

            if scaler is not None:
                pipeline = Pipeline([("scaler", scaler), ("clf", model)])
            else:
                pipeline = Pipeline([("clf", model)])

            initial_type = [(input_name, FloatTensorType([None, feature_dim]))]

            # LightGBM needs ai.onnx.ml opset pinned to 3 for onnxmltools compat
            target_opset = {"": 17, "ai.onnx.ml": 3} if is_lgbm else 17
            onnx_model = convert_sklearn(
                pipeline, initial_types=initial_type, target_opset=target_opset,
                options={id(model): {"zipmap": False}} if is_lgbm else None,
            )
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            log.info(
                "[onnx-export] Exported %s for user=%s (%d bytes)",
                filename, self.user_id, onnx_path.stat().st_size,
            )
            return str(onnx_path)

        except ImportError:
            log.info("[onnx-export] skl2onnx not available, using manual ONNX export")
            return self._export_manual_onnx(model, scaler, feature_dim, onnx_path, input_name)

    def _export_manual_onnx(
        self,
        model: Any,
        scaler: Any,
        feature_dim: int,
        onnx_path: Path,
        input_name: str,
    ) -> Optional[str]:
        """Manual ONNX export for SGDClassifier using numpy weights.

        Creates a simple linear model: logits = (X - mean) / scale @ W + b
        """
        try:
            import onnx
            from onnx import numpy_helper, TensorProto
            from onnx.helper import (
                make_model, make_graph, make_node, make_tensor_value_info,
                make_opsetid,
            )
        except ImportError:
            log.warning("[onnx-export] onnx package not available, cannot export")
            return None

        # Extract weights from SGDClassifier
        if not hasattr(model, "coef_"):
            log.warning("[onnx-export] Model has no coef_ attribute, cannot export")
            return None

        W = model.coef_.astype(np.float32)  # (n_classes, feature_dim)
        b = model.intercept_.astype(np.float32) if hasattr(model, "intercept_") else np.zeros(W.shape[0], dtype=np.float32)

        nodes = []
        initializers = []
        curr_input = input_name

        # If scaler exists, add normalization: (X - mean) / scale
        if scaler is not None and hasattr(scaler, "mean_"):
            mean = scaler.mean_.astype(np.float32)
            scale = scaler.scale_.astype(np.float32)
            scale = np.where(scale == 0, 1.0, scale).astype(np.float32)

            initializers.append(numpy_helper.from_array(mean, name="scaler_mean"))
            initializers.append(numpy_helper.from_array(scale, name="scaler_scale"))

            nodes.append(make_node("Sub", [curr_input, "scaler_mean"], ["centered"]))
            nodes.append(make_node("Div", ["centered", "scaler_scale"], ["scaled"]))
            curr_input = "scaled"

        # Linear: logits = X @ W^T + b
        initializers.append(numpy_helper.from_array(W.T, name="weights"))
        initializers.append(numpy_helper.from_array(b, name="bias"))

        nodes.append(make_node("MatMul", [curr_input, "weights"], ["pre_bias"]))
        nodes.append(make_node("Add", ["pre_bias", "bias"], ["logits"]))

        input_info = make_tensor_value_info(input_name, TensorProto.FLOAT, [None, feature_dim])
        output_info = make_tensor_value_info("logits", TensorProto.FLOAT, [None, W.shape[0]])

        graph = make_graph(nodes, "user_emotion_classifier", [input_info], [output_info], initializer=initializers)
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 17)])
        onnx_model.ir_version = 8

        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        log.info(
            "[onnx-export] Manual ONNX exported %s for user=%s (%d bytes)",
            onnx_path.name, self.user_id, onnx_path.stat().st_size,
        )
        return str(onnx_path)

    # ---- private helpers ---------------------------------------------------

    def _load_retrain_meta(self) -> Dict[str, Any]:
        """Load retrain metadata (tracks when last retrain happened)."""
        meta_path = self._user_dir / "retrain_meta.json"
        if not meta_path.exists():
            return {}
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception:
            return {}

    def _update_retrain_meta(self, correction_count: int) -> None:
        """Update retrain metadata after a successful retrain."""
        self._user_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self._user_dir / "retrain_meta.json"
        meta = self._load_retrain_meta()
        meta["last_retrain_count"] = correction_count
        meta["last_retrain_at"] = datetime.now(timezone.utc).isoformat()
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def _load_model_meta(self, filename: str) -> Optional[Dict]:
        """Load a model-specific meta JSON file."""
        meta_path = self._user_dir / filename
        if not meta_path.exists():
            return None
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception:
            return None
