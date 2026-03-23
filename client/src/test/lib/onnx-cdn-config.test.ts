import { describe, it, expect } from "vitest";

/**
 * Tests for ONNX Runtime CDN loading configuration (Issue #510).
 *
 * Verifies that the CDN URL builder produces correct URLs and that
 * the configuration logic properly detects when CDN loading should be used.
 */

import {
  getOrtWasmCdnUrl,
  ONNX_CDN_BASE_URL,
  ONNX_RUNTIME_VERSION,
  shouldUseCdnWasm,
  REQUIRED_ONNX_OPERATORS,
} from "@/lib/onnx-cdn-config";

describe("ONNX CDN configuration (Issue #510)", () => {
  describe("CDN URL construction", () => {
    it("builds a valid CDN URL for WASM file", () => {
      const url = getOrtWasmCdnUrl("ort-wasm-simd.wasm");
      expect(url).toContain("cdn.jsdelivr.net");
      expect(url).toContain("onnxruntime-web");
      expect(url).toContain(ONNX_RUNTIME_VERSION);
      expect(url.endsWith("ort-wasm-simd.wasm")).toBe(true);
    });

    it("builds URL for threaded SIMD variant", () => {
      const url = getOrtWasmCdnUrl("ort-wasm-simd-threaded.wasm");
      expect(url).toContain("ort-wasm-simd-threaded.wasm");
    });

    it("builds URL for plain WASM variant", () => {
      const url = getOrtWasmCdnUrl("ort-wasm.wasm");
      expect(url).toContain("ort-wasm.wasm");
    });

    it("includes the correct version number", () => {
      const url = getOrtWasmCdnUrl("ort-wasm-simd.wasm");
      expect(url).toContain(ONNX_RUNTIME_VERSION);
    });
  });

  describe("CDN base URL", () => {
    it("uses jsdelivr CDN", () => {
      expect(ONNX_CDN_BASE_URL).toContain("cdn.jsdelivr.net");
    });

    it("points to onnxruntime-web package", () => {
      expect(ONNX_CDN_BASE_URL).toContain("onnxruntime-web");
    });
  });

  describe("shouldUseCdnWasm", () => {
    it("returns true when CDN loading is enabled (default)", () => {
      expect(shouldUseCdnWasm()).toBe(true);
    });
  });

  describe("required ONNX operators list", () => {
    it("lists operators needed for emotion classification", () => {
      expect(REQUIRED_ONNX_OPERATORS).toContain("Conv");
      expect(REQUIRED_ONNX_OPERATORS).toContain("Relu");
      expect(REQUIRED_ONNX_OPERATORS).toContain("MatMul");
    });

    it("lists operators needed for EEGNet", () => {
      expect(REQUIRED_ONNX_OPERATORS).toContain("BatchNormalization");
      expect(REQUIRED_ONNX_OPERATORS).toContain("AveragePool");
    });

    it("includes softmax for output layer", () => {
      expect(REQUIRED_ONNX_OPERATORS).toContain("Softmax");
    });

    it("has no duplicate operators", () => {
      const unique = new Set(REQUIRED_ONNX_OPERATORS);
      expect(unique.size).toBe(REQUIRED_ONNX_OPERATORS.length);
    });
  });
});
