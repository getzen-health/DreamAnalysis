import { sbGetSetting } from "./supabase-store";
/**
 * ONNX Runtime CDN configuration (Issue #510).
 *
 * Loads ONNX Runtime WASM binaries from a CDN instead of bundling them
 * in the APK. This reduces the app bundle size by ~24 MB (the threaded
 * SIMD variant alone is 24 MB).
 *
 * The CDN approach:
 *   - WASM files are loaded on-demand from jsdelivr when inference is first needed
 *   - Browser caches them after first load (HTTP cache headers)
 *   - No impact on APK/bundle size
 *   - Graceful fallback to bundled WASM if CDN is unreachable
 */

/** Version of onnxruntime-web matching package.json (^1.17.0) */
export const ONNX_RUNTIME_VERSION = "1.17.0";

/** CDN base URL for onnxruntime-web WASM files */
export const ONNX_CDN_BASE_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ONNX_RUNTIME_VERSION}/dist/`;

/**
 * Build a full CDN URL for a specific ONNX Runtime WASM file.
 *
 * @param filename - e.g. "ort-wasm-simd.wasm", "ort-wasm-simd-threaded.wasm"
 * @returns Full CDN URL
 */
export function getOrtWasmCdnUrl(filename: string): string {
  return `${ONNX_CDN_BASE_URL}${filename}`;
}

/**
 * Whether to load ONNX WASM from CDN instead of bundling.
 *
 * Returns true by default. Can be overridden via localStorage for debugging:
 *   localStorage.setItem("ndw_onnx_use_cdn", "false");
 */
export function shouldUseCdnWasm(): boolean {
  try {
    const override = sbGetSetting("ndw_onnx_use_cdn");
    if (override === "false") return false;
  } catch {
    // localStorage not available (SSR, etc.)
  }
  return true;
}

/**
 * Apply CDN configuration to an onnxruntime-web module.
 *
 * Must be called BEFORE any InferenceSession.create() call.
 * Sets env.wasm.wasmPaths to point at the CDN so ORT fetches
 * WASM binaries from there instead of from the app bundle.
 */
export function applyOrtCdnConfig(
  ortModule: typeof import("onnxruntime-web")
): void {
  if (!shouldUseCdnWasm()) return;
  ortModule.env.wasm.wasmPaths = ONNX_CDN_BASE_URL;
}

/**
 * List of ONNX operators required by our models.
 *
 * Used to document what a custom ONNX Runtime build needs to include.
 * See scripts/build-custom-ort.sh for the custom build process.
 *
 * Models:
 *   - emotion_classifier_model.onnx (generic emotion, 2.2 MB)
 *   - eegnet_emotion_4ch.onnx (EEGNet 4-channel, 4 KB)
 */
export const REQUIRED_ONNX_OPERATORS: string[] = [
  // Convolution layers (EEGNet temporal + spatial convolutions)
  "Conv",
  // Batch normalization (EEGNet)
  "BatchNormalization",
  // Activation functions
  "Relu",
  "Sigmoid",
  "Tanh",
  // Pooling
  "AveragePool",
  "GlobalAveragePool",
  // Fully connected layers
  "MatMul",
  "Gemm",
  // Element-wise operations
  "Add",
  "Mul",
  "Sub",
  // Shape operations
  "Reshape",
  "Flatten",
  "Transpose",
  "Squeeze",
  "Unsqueeze",
  // Output
  "Softmax",
  // Dropout (identity at inference time)
  "Dropout",
  // Padding
  "Pad",
  // Concatenation
  "Concat",
];
