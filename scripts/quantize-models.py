#!/usr/bin/env python3
"""
quantize-models.py — INT8 quantization for ONNX models (Issue #510).

Converts ONNX models from FP32 to INT8, reducing model size by ~4x
and improving inference speed on CPU/WASM by ~2-3x.

Usage:
    python scripts/quantize-models.py

Input:
    public/models/emotion_classifier_model.onnx  (2.2 MB, FP32)
    public/models/eegnet_emotion_4ch.onnx        (4 KB, FP32)

Output:
    public/models/emotion_classifier_model_int8.onnx  (~0.6 MB, INT8)
    public/models/eegnet_emotion_4ch_int8.onnx        (~1.5 KB, INT8)

Requirements:
    pip install onnxruntime onnx

References:
    https://onnxruntime.ai/docs/performance/quantization.html
"""

import sys
from pathlib import Path

try:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install onnxruntime onnx")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "public" / "models"

# Models to quantize
MODELS = [
    "emotion_classifier_model.onnx",
    "eegnet_emotion_4ch.onnx",
]


def quantize_model(input_path: Path, output_path: Path) -> None:
    """Apply INT8 dynamic quantization to an ONNX model."""
    if not input_path.exists():
        print(f"  SKIP: {input_path.name} not found")
        return

    input_size_kb = input_path.stat().st_size / 1024
    print(f"  Quantizing {input_path.name} ({input_size_kb:.1f} KB)...")

    try:
        # Validate the input model first
        model = onnx.load(str(input_path))
        onnx.checker.check_model(model)

        # Dynamic quantization: weights are quantized to INT8,
        # activations are quantized at runtime (no calibration data needed).
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
            # Only quantize MatMul and Gemm (fully connected layers).
            # Conv layers in tiny models like EEGNet may lose accuracy with INT8.
            op_types_to_quantize=["MatMul", "Gemm"],
        )

        output_size_kb = output_path.stat().st_size / 1024
        reduction = (1 - output_size_kb / input_size_kb) * 100
        print(f"  OK: {output_path.name} ({output_size_kb:.1f} KB, {reduction:.0f}% smaller)")

    except Exception as e:
        print(f"  ERROR: {e}")


def main() -> None:
    print("=== NeuralDreamWorkshop Model INT8 Quantization ===")
    print(f"Models directory: {MODELS_DIR}")
    print()

    if not MODELS_DIR.exists():
        print(f"ERROR: Models directory not found: {MODELS_DIR}")
        print("Make sure ONNX models are in public/models/")
        sys.exit(1)

    for model_name in MODELS:
        input_path = MODELS_DIR / model_name
        stem = input_path.stem
        output_name = f"{stem}_int8.onnx"
        output_path = MODELS_DIR / output_name
        quantize_model(input_path, output_path)

    print()
    print("Done. INT8 models are in public/models/")
    print()
    print("To use INT8 models in the app:")
    print("1. Update model paths in client/src/lib/ml-local.ts")
    print("2. Test inference accuracy — INT8 may reduce emotion accuracy by 1-3%")
    print("3. If accuracy drops too much, try quantizing only the larger model")


if __name__ == "__main__":
    main()
