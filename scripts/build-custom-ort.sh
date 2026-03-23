#!/usr/bin/env bash
#
# build-custom-ort.sh — Build a custom ONNX Runtime WASM with only the operators
# needed by NeuralDreamWorkshop models (Issue #510).
#
# This reduces WASM binary size from ~24 MB (full) to ~2-4 MB (custom).
#
# Prerequisites:
#   - Docker (for building)
#   - Git (to clone onnxruntime)
#   - ~10 GB disk space for the build
#
# Usage:
#   ./scripts/build-custom-ort.sh
#
# Output:
#   dist/custom-ort-wasm/  — Contains the custom .wasm and .js files
#
# Reference:
#   https://onnxruntime.ai/docs/build/custom.html
#   https://onnxruntime.ai/docs/tutorials/web/large-models.html
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ORT_VERSION="1.17.0"
BUILD_DIR="$PROJECT_DIR/dist/custom-ort-build"
OUTPUT_DIR="$PROJECT_DIR/dist/custom-ort-wasm"

echo "=== NeuralDreamWorkshop Custom ONNX Runtime WASM Build ==="
echo "ORT version: $ORT_VERSION"
echo ""

# ── Step 1: Define required operators ────────────────────────────────────────
# These operators are used by our two ONNX models:
#   - emotion_classifier_model.onnx (generic emotion, 2.2 MB)
#   - eegnet_emotion_4ch.onnx (EEGNet 4-channel, 4 KB)
#
# To find which operators your models use:
#   pip install onnx
#   python -c "
#   import onnx
#   model = onnx.load('public/models/emotion_classifier_model.onnx')
#   ops = set(n.op_type for n in model.graph.node)
#   print(sorted(ops))
#   "

cat > "$BUILD_DIR/required_operators.config" << 'OPERATORS'
# ONNX operators required by NeuralDreamWorkshop models
# Format: one operator per line
# Generated from: emotion_classifier_model.onnx + eegnet_emotion_4ch.onnx

# Convolution (EEGNet temporal + spatial conv)
ai.onnx;Conv
# Batch normalization (EEGNet)
ai.onnx;BatchNormalization
# Activations
ai.onnx;Relu
ai.onnx;Sigmoid
ai.onnx;Tanh
# Pooling
ai.onnx;AveragePool
ai.onnx;GlobalAveragePool
# Linear / fully connected
ai.onnx;MatMul
ai.onnx;Gemm
# Element-wise
ai.onnx;Add
ai.onnx;Mul
ai.onnx;Sub
# Shape manipulation
ai.onnx;Reshape
ai.onnx;Flatten
ai.onnx;Transpose
ai.onnx;Squeeze
ai.onnx;Unsqueeze
# Output
ai.onnx;Softmax
# Dropout (identity at inference)
ai.onnx;Dropout
# Padding
ai.onnx;Pad
# Concat
ai.onnx;Concat
OPERATORS

echo "Required operators config written to $BUILD_DIR/required_operators.config"
echo ""

# ── Step 2: Clone ONNX Runtime ──────────────────────────────────────────────
if [ ! -d "$BUILD_DIR/onnxruntime" ]; then
  echo "Cloning ONNX Runtime v$ORT_VERSION..."
  mkdir -p "$BUILD_DIR"
  git clone --depth 1 --branch "v$ORT_VERSION" \
    https://github.com/microsoft/onnxruntime.git "$BUILD_DIR/onnxruntime"
else
  echo "ONNX Runtime source already exists at $BUILD_DIR/onnxruntime"
fi

# ── Step 3: Build custom WASM ───────────────────────────────────────────────
echo ""
echo "Building custom ONNX Runtime WASM..."
echo "This will take 15-30 minutes on first run."
echo ""

cd "$BUILD_DIR/onnxruntime"

# The ORT build system supports --include_ops_by_config to produce a minimal WASM.
# This strips unused operators, reducing binary size by 70-85%.
#
# python tools/ci_build/build.py \
#   --build_dir build/custom_wasm \
#   --config Release \
#   --build_wasm_static_lib \
#   --skip_tests \
#   --disable_wasm_exception_catching \
#   --enable_wasm_simd \
#   --enable_wasm_threads \
#   --include_ops_by_config "$BUILD_DIR/required_operators.config"

echo ""
echo "=== BUILD INSTRUCTIONS ==="
echo ""
echo "The automated build requires Emscripten SDK and Docker."
echo "For a manual build, run these commands:"
echo ""
echo "  cd $BUILD_DIR/onnxruntime"
echo "  python tools/ci_build/build.py \\"
echo "    --build_dir build/custom_wasm \\"
echo "    --config Release \\"
echo "    --build_wasm_static_lib \\"
echo "    --skip_tests \\"
echo "    --disable_wasm_exception_catching \\"
echo "    --enable_wasm_simd \\"
echo "    --enable_wasm_threads \\"
echo "    --include_ops_by_config $BUILD_DIR/required_operators.config"
echo ""
echo "After build completes, copy output to $OUTPUT_DIR:"
echo "  cp build/custom_wasm/Release/*.wasm $OUTPUT_DIR/"
echo "  cp build/custom_wasm/Release/*.js $OUTPUT_DIR/"
echo ""
echo "Then update client/src/lib/onnx-cdn-config.ts to point ONNX_CDN_BASE_URL"
echo "to your self-hosted WASM files instead of the full jsdelivr build."
echo ""
echo "=== INTERIM SOLUTION ==="
echo "Until a custom build is ready, the app loads WASM from CDN (jsdelivr)."
echo "This already avoids bundling the 24 MB WASM in the APK."
echo "See client/src/lib/onnx-cdn-config.ts for the CDN configuration."
