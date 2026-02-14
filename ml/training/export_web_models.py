"""Export trained models to ONNX format for browser-side inference.

Loads sklearn .pkl models, converts to ONNX with proper input/output shapes,
and copies to client/public/models/ for static serving via Vite.
"""

import sys
import shutil
from pathlib import Path
import numpy as np

MODEL_DIR = Path(__file__).parent.parent / "models" / "saved"
WEB_MODELS_DIR = Path(__file__).parent.parent.parent / "client" / "public" / "models"


def export_sklearn_to_onnx(pkl_path: Path, output_path: Path, n_features: int = 17):
    """Convert a sklearn model (.pkl) to ONNX format.

    Args:
        pkl_path: Path to the .pkl file (joblib format with model + feature_names).
        output_path: Path to write the .onnx file.
        n_features: Number of input features.
    """
    import joblib

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print(f"skl2onnx not installed. Install with: pip install skl2onnx")
        print(f"Copying .pkl as fallback for: {pkl_path.name}")
        return False

    data = joblib.load(str(pkl_path))
    model = data["model"]
    feature_names = data.get("feature_names", [f"f{i}" for i in range(n_features)])
    n_features = len(feature_names)

    initial_type = [("input", FloatTensorType([None, n_features]))]

    try:
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"Exported: {pkl_path.name} -> {output_path.name}")
        return True
    except Exception as e:
        print(f"Failed to convert {pkl_path.name}: {e}")
        return False


def validate_onnx_model(onnx_path: Path, pkl_path: Path):
    """Validate ONNX model outputs match sklearn model outputs."""
    import joblib
    import onnxruntime as ort

    data = joblib.load(str(pkl_path))
    model = data["model"]
    feature_names = data.get("feature_names", [])

    # Create random test input
    n_features = len(feature_names) if feature_names else 17
    test_input = np.random.randn(1, n_features).astype(np.float32)

    # Sklearn prediction
    try:
        sklearn_probs = model.predict_proba(test_input)[0]
    except Exception:
        print(f"Sklearn prediction failed for {pkl_path.name}")
        return False

    # ONNX prediction
    try:
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        onnx_output = session.run(None, {input_name: test_input})
        onnx_probs = onnx_output[1][0] if len(onnx_output) > 1 else onnx_output[0][0]
    except Exception as e:
        print(f"ONNX prediction failed for {onnx_path.name}: {e}")
        return False

    # Compare
    max_diff = np.max(np.abs(np.array(list(sklearn_probs)) - np.array(list(onnx_probs.values() if isinstance(onnx_probs, dict) else onnx_probs))))
    print(f"Validation {onnx_path.name}: max_diff = {max_diff:.6f}")
    return max_diff < 0.01


def export_all():
    """Export all trained models to ONNX and copy to web directory."""
    WEB_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_names = [
        "sleep_staging_model",
        "emotion_classifier_model",
        "dream_detector_model",
    ]

    exported = []
    for name in model_names:
        pkl_path = MODEL_DIR / f"{name}.pkl"
        onnx_src = MODEL_DIR / f"{name}.onnx"
        onnx_dst = WEB_MODELS_DIR / f"{name}.onnx"

        if onnx_src.exists():
            # Already have ONNX, just copy
            shutil.copy2(str(onnx_src), str(onnx_dst))
            print(f"Copied existing ONNX: {name}.onnx")
            exported.append(name)
        elif pkl_path.exists():
            # Convert from sklearn
            onnx_path = WEB_MODELS_DIR / f"{name}.onnx"
            if export_sklearn_to_onnx(pkl_path, onnx_path):
                exported.append(name)
        else:
            print(f"No model found for: {name}")

    print(f"\nExported {len(exported)}/{len(model_names)} models to {WEB_MODELS_DIR}")
    return exported


if __name__ == "__main__":
    export_all()
