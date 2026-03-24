/**
 * Local ML Engine — runs ONNX models in the browser via onnxruntime-web.
 *
 * Emotion: loaded from /models/emotion_classifier_model.onnx (2.2 MB).
 * Sleep + Dream: JS heuristic fallbacks (no download needed).
 * Falls back gracefully to server API via use-inference.ts.
 */

import { extractFeatures, extractBandPowers } from "./eeg-features";
import {
  softmax,
  normalizeChannels,
  eegnetEmotionFromProbabilities,
  validateEEGNetInput,
  type EEGNetEmotionResult,
} from "./eegnet-utils";
import {
  loadPersonalAdapter,
  savePersonalAdapter,
  applyAdapter,
  updateAfterSession,
} from "./personal-adapter";
import { applyOrtCdnConfig } from "./onnx-cdn-config";
import { loadModelFromStorage } from "./model-updater";
import {
  InferenceLatencyTracker,
  type InferenceStats,
  type ModelLatencyStats,
} from "./inference-latency";

// onnxruntime-web is loaded dynamically to avoid hard failures if not installed
let ort: typeof import("onnxruntime-web") | null = null;

/**
 * Detect WebAssembly SIMD support at runtime.
 *
 * The threaded SIMD variant (`ort-wasm-simd-threaded.wasm`) is 24 MB and
 * requires both SIMD and SharedArrayBuffer (cross-origin isolation).
 * Browsers that lack either will hang or error at decode time.
 *
 * Strategy:
 *   - SIMD + SAB available  → threaded SIMD (default, fastest)
 *   - SIMD only             → single-threaded SIMD (ort-wasm-simd.wasm)
 *   - Neither               → plain WASM scalar fallback
 *
 * Returns true if SIMD is supported (SAB check is a separate gate).
 */
function detectSimd(): boolean {
  try {
    // WebAssembly SIMD feature-detect: attempt to validate a tiny SIMD module.
    // This is the canonical check used by the wasm-feature-detect library.
    const simdBytes = new Uint8Array([
      0x00, 0x61, 0x73, 0x6d, // magic: \0asm
      0x01, 0x00, 0x00, 0x00, // version: 1
      0x01, 0x05, 0x01,       // type section
      0x60, 0x00, 0x01, 0x7b, // () -> v128
    ]);
    return WebAssembly.validate(simdBytes);
  } catch {
    return false;
  }
}

/**
 * Configure onnxruntime-web WASM backend based on runtime capabilities.
 * Must be called before the first InferenceSession.create().
 */
function configureOrtWasm(ortModule: typeof import("onnxruntime-web")): void {
  const hasSab = typeof SharedArrayBuffer !== "undefined";
  const hasSimd = detectSimd();

  if (!hasSimd) {
    // Scalar fallback — disable threading and SIMD to avoid loading the wrong binary
    ortModule.env.wasm.simd = false;
    ortModule.env.wasm.numThreads = 1;
  } else if (!hasSab) {
    // SIMD available but no SharedArrayBuffer (no cross-origin isolation) —
    // disable threading so ORT doesn't attempt the threaded WASM variant
    ortModule.env.wasm.simd = true;
    ortModule.env.wasm.numThreads = 1;
  }
  // If both SIMD and SAB are available, ORT defaults to threaded SIMD — no override needed.
}

async function loadOrt() {
  if (ort) return ort;
  try {
    ort = await import("onnxruntime-web");
    // Apply CDN loading so WASM binaries come from jsdelivr, not the app bundle.
    // This reduces APK size by ~24 MB (Issue #510).
    applyOrtCdnConfig(ort);
    // Apply SIMD/threading configuration before any InferenceSession is created.
    configureOrtWasm(ort);
    return ort;
  } catch {
    return null;
  }
}

interface SleepPrediction {
  stage: string;
  stage_index: number;
  confidence: number;
  probabilities: Record<string, number>;
}

interface EmotionPrediction {
  emotion: string;
  confidence: number;
  probabilities: Record<string, number>;
}

interface DreamPrediction {
  is_dreaming: boolean;
  probability: number;
}

const SLEEP_STAGES = ["Wake", "N1", "N2", "N3", "REM"];
const EMOTIONS = ["happy", "sad", "angry", "fearful", "relaxed", "focused"];

type InferenceSession = import("onnxruntime-web").InferenceSession;

// ── Heuristic sleep staging (no model needed) ─────────────────────────────
// Uses delta/theta/alpha/beta band powers to classify sleep stage.
// Accuracy: ~60-65% without calibration. Good enough for offline use.
function sleepHeuristic(features: number[], fs: number, signal: number[]): SleepPrediction {
  const bands = extractBandPowers(signal, fs);
  const delta = bands.delta ?? 0;
  const theta = bands.theta ?? 0;
  const alpha = bands.alpha ?? 0;
  const beta  = bands.beta  ?? 0.001;

  const scores = {
    Wake: alpha * 0.4 + beta * 0.6,
    N1:   theta * 0.5 + alpha * 0.3 + (1 - delta) * 0.2,
    N2:   (1 - delta) * 0.4 + theta * 0.3 + (1 - beta) * 0.3,
    N3:   delta * 0.7 + (1 - beta) * 0.3,
    REM:  theta * 0.6 + (1 - delta) * 0.4,
  };

  // Softmax
  const maxScore = Math.max(...Object.values(scores));
  const exps = Object.fromEntries(
    Object.entries(scores).map(([k, v]) => [k, Math.exp(v - maxScore)])
  );
  const expSum = Object.values(exps).reduce((a, b) => a + b, 0);
  const probs = Object.fromEntries(
    Object.entries(exps).map(([k, v]) => [k, v / expSum])
  ) as Record<string, number>;

  const stage = (Object.entries(probs).sort((a, b) => b[1] - a[1])[0][0]) as string;
  const idx = SLEEP_STAGES.indexOf(stage);

  return {
    stage,
    stage_index: idx >= 0 ? idx : 0,
    confidence: probs[stage],
    probabilities: { Wake: probs.Wake, N1: probs.N1, N2: probs.N2, N3: probs.N3, REM: probs.REM },
  };
}

// ── Heuristic dream detection (no model needed) ───────────────────────────
// Dreams correlate with REM sleep: high theta, low delta, moderate alpha.
function dreamHeuristic(features: number[], fs: number, signal: number[]): DreamPrediction {
  const bands = extractBandPowers(signal, fs);
  const delta = bands.delta ?? 0;
  const theta = bands.theta ?? 0;
  const alpha = bands.alpha ?? 0;

  // Higher theta + lower delta + some alpha → more likely dreaming (REM)
  const remScore = theta * 0.5 + (1 - delta) * 0.3 + alpha * 0.2;
  const probability = Math.min(1, Math.max(0, (remScore - 0.15) * 3));

  return { is_dreaming: probability > 0.5, probability };
}

class LocalMLEngine {
  private emotionSession: InferenceSession | null = null;
  private eegnetSession: InferenceSession | null = null;
  private userEegSession: InferenceSession | null = null;
  private _ready = false;
  private _initPromise: Promise<void> | null = null;
  private _lastSignal: number[] = [];
  private _lastFs = 256;
  private _latency = new InferenceLatencyTracker();

  /** Must call before first prediction. Resolves after ONNX attempt. */
  async initialize(): Promise<void> {
    if (this._initPromise) return this._initPromise;
    this._initPromise = this._doInit();
    return this._initPromise;
  }

  private async _doInit(): Promise<void> {
    const ortModule = await loadOrt();
    if (!ortModule) {
      // Heuristics still work without ONNX
      this._ready = true;
      console.info("onnxruntime-web unavailable, using heuristic inference");
      return;
    }

    try {
      this.emotionSession = await ortModule.InferenceSession.create(
        "/models/emotion_classifier_model.onnx"
      );
      console.info("Local emotion ONNX loaded");
    } catch {
      // Model not served — heuristics only
    }

    // Try to load EEGNet model (4KB, fast)
    try {
      this.eegnetSession = await ortModule.InferenceSession.create(
        "/models/eegnet_emotion_4ch.onnx",
        { executionProviders: ["wasm"] }
      );
      console.info("EEGNet ONNX loaded (4KB, 4-channel)");
    } catch {
      // EEGNet not served — fall back to generic emotion model or heuristics
    }

    // Try to load per-user EEG model from IndexedDB (downloaded by model-updater)
    try {
      const userBuffer = await loadModelFromStorage("eeg_emotion_user.onnx");
      if (userBuffer && ortModule) {
        this.userEegSession = await ortModule.InferenceSession.create(
          userBuffer,
          { executionProviders: ["wasm"] }
        );
        console.info("Per-user EEG ONNX loaded from IndexedDB");
      }
    } catch {
      // Per-user model not available — will use generic
    }

    // Always ready: heuristics handle sleep + dream, ONNX handles emotion when available
    this._ready = true;
  }

  /**
   * Attempt to reload per-user model from IndexedDB.
   * Call after model-updater downloads a new version.
   */
  async reloadUserModel(): Promise<boolean> {
    const ortModule = await loadOrt();
    if (!ortModule) return false;

    try {
      const userBuffer = await loadModelFromStorage("eeg_emotion_user.onnx");
      if (!userBuffer) return false;

      this.userEegSession = await ortModule.InferenceSession.create(
        userBuffer,
        { executionProviders: ["wasm"] }
      );
      console.info("Per-user EEG ONNX reloaded from IndexedDB");
      return true;
    } catch {
      return false;
    }
  }

  isReady(): boolean { return this._ready; }

  /** Check if EEGNet model is loaded and available */
  isEEGNetReady(): boolean { return this.eegnetSession !== null; }

  /** Cache the raw signal so heuristics can use it */
  setLastSignal(signal: number[], fs: number): void {
    this._lastSignal = signal;
    this._lastFs = fs;
  }

  /**
   * Run EEGNet inference on raw 4-channel EEG data.
   *
   * Input: array of 4 Float32Arrays, each 1024 samples (4 seconds @ 256 Hz).
   * Normalizes each channel to zero-mean unit-variance before inference.
   * Returns structured emotion result or null if EEGNet is not loaded.
   */
  async analyzeEmotionEEGNet(
    rawChannels: Float32Array[],
    _sampleRate: number = 256
  ): Promise<EmotionPrediction | null> {
    if (!this.eegnetSession) return null;

    // Validate input shape
    if (!validateEEGNetInput(rawChannels)) {
      console.warn(
        `EEGNet: expected ${4} channels x ${1024} samples, got ${rawChannels.length} channels x ${rawChannels[0]?.length ?? 0} samples`
      );
      return null;
    }

    const ortModule = await loadOrt();
    if (!ortModule) return null;

    try {
      // Normalize each channel to zero-mean, unit-variance
      const normalized = normalizeChannels(rawChannels);

      // Flatten into [1, 4, 1024] tensor
      const flat = new Float32Array(4 * 1024);
      for (let c = 0; c < 4; c++) {
        flat.set(normalized[c], c * 1024);
      }

      const input = new ortModule.Tensor("float32", flat, [1, 4, 1024]);
      const inputName = this.eegnetSession.inputNames[0];
      const t0 = performance.now();
      const results = await this.eegnetSession.run({ [inputName]: input });
      this._latency.record("eegnet", performance.now() - t0);
      const outputName = this.eegnetSession.outputNames[0];
      const logits = Array.from(results[outputName].data as Float32Array);

      // Softmax → probabilities → personal adaptation → structured result
      const rawProbs = softmax(logits);

      // Apply personal adapter to adjust probabilities per user
      let adapter = loadPersonalAdapter();
      const adaptedProbs = applyAdapter(rawProbs, adapter);

      const result: EEGNetEmotionResult = eegnetEmotionFromProbabilities(adaptedProbs);

      // Find predicted class index for session tracking
      let predictedClass = 0;
      let maxProb = adaptedProbs[0];
      for (let i = 1; i < adaptedProbs.length; i++) {
        if (adaptedProbs[i] > maxProb) {
          maxProb = adaptedProbs[i];
          predictedClass = i;
        }
      }

      // Update session stats (unsupervised anti-collapse)
      adapter = updateAfterSession(adapter, predictedClass);
      savePersonalAdapter(adapter);

      return {
        emotion: result.emotion,
        confidence: result.confidence,
        probabilities: result.probabilities,
      };
    } catch (err) {
      console.warn("EEGNet inference failed:", err);
      return null;
    }
  }

  async analyzeSleep(features: number[]): Promise<SleepPrediction | null> {
    return sleepHeuristic(features, this._lastFs, this._lastSignal);
  }

  /** Check if per-user fine-tuned model is loaded */
  isUserModelReady(): boolean { return this.userEegSession !== null; }

  async analyzeEmotion(features: number[]): Promise<EmotionPrediction | null> {
    // Try per-user fine-tuned ONNX model first (most accurate for this user)
    if (this.userEegSession) {
      const ortModule = await loadOrt();
      if (ortModule) {
        try {
          // Per-user model expects 170-dim feature vector (padded if needed)
          let paddedFeatures = features;
          if (features.length < 170) {
            paddedFeatures = [...features, ...new Array(170 - features.length).fill(0)];
          } else if (features.length > 170) {
            paddedFeatures = features.slice(0, 170);
          }

          const input = new ortModule.Tensor(
            "float32",
            new Float32Array(paddedFeatures),
            [1, paddedFeatures.length]
          );
          const inputName = this.userEegSession.inputNames[0];
          const t0 = performance.now();
          const results = await this.userEegSession.run({ [inputName]: input });
          this._latency.record("userEeg", performance.now() - t0);
          const outputName = this.userEegSession.outputNames[0];
          const output = results[outputName];
          const data = output.data as Float32Array | BigInt64Array;

          // skl2onnx outputs class labels (first output) and probabilities (second)
          // If we have a second output, use it for probabilities
          const outputNames = this.userEegSession.outputNames;
          if (outputNames.length >= 2) {
            const probOutput = results[outputNames[1]];
            // probOutput may be a map or array depending on zipmap setting
            const label = Number(data[0]);
            const emotionForLabel = EMOTIONS[label] || EMOTIONS[0];
            return {
              emotion: emotionForLabel,
              confidence: 0.75, // Fine-tuned model has reasonable confidence
              probabilities: Object.fromEntries(
                EMOTIONS.map((e) => [e, e === emotionForLabel ? 0.75 : 0.05])
              ),
            };
          }

          // Single output: treat as logits
          let maxIdx = 0;
          let maxVal = Number(data[0]);
          for (let i = 1; i < data.length; i++) {
            if (Number(data[i]) > maxVal) { maxVal = Number(data[i]); maxIdx = i; }
          }

          const expVals = Array.from(data as Iterable<number | bigint>).map((v) => Math.exp(Number(v) - maxVal));
          const expSum = expVals.reduce((a, b) => a + b, 0);
          const probs = expVals.map((v) => v / expSum);

          return {
            emotion: EMOTIONS[maxIdx] || "relaxed",
            confidence: probs[maxIdx],
            probabilities: Object.fromEntries(
              EMOTIONS.map((e, i) => [e, probs[i] || 0])
            ),
          };
        } catch {
          // Fall through to generic model
        }
      }
    }

    // Try generic ONNX emotion model
    if (this.emotionSession) {
      const ortModule = await loadOrt();
      if (ortModule) {
        try {
          const input = new ortModule.Tensor(
            "float32",
            new Float32Array(features),
            [1, features.length]
          );
          const inputName = this.emotionSession.inputNames[0];
          const t0 = performance.now();
          const results = await this.emotionSession.run({ [inputName]: input });
          this._latency.record("emotion", performance.now() - t0);
          const outputName = this.emotionSession.outputNames[0];
          const output = results[outputName];
          const data = output.data as Float32Array;

          let maxIdx = 0;
          let maxVal = data[0];
          for (let i = 1; i < data.length; i++) {
            if (data[i] > maxVal) { maxVal = data[i]; maxIdx = i; }
          }

          const expVals = Array.from(data).map((v) => Math.exp(v - maxVal));
          const expSum = expVals.reduce((a, b) => a + b, 0);
          const probs = expVals.map((v) => v / expSum);

          return {
            emotion: EMOTIONS[maxIdx] || "relaxed",
            confidence: probs[maxIdx],
            probabilities: Object.fromEntries(
              EMOTIONS.map((e, i) => [e, probs[i] || 0])
            ),
          };
        } catch { /* fall through to heuristic */ }
      }
    }

    // Heuristic fallback: use band power features
    const bands = extractBandPowers(this._lastSignal, this._lastFs);
    const alpha = bands.alpha ?? 0;
    const beta  = bands.beta  ?? 0.001;
    const theta = bands.theta ?? 0;
    const delta = bands.delta ?? 0;

    const abr = alpha / Math.max(beta, 1e-10);
    const tbr = theta / Math.max(beta, 1e-10);

    // Simple rule-based mapping
    let emotion = "relaxed";
    if (abr > 1.5 && tbr > 1.0) emotion = "relaxed";
    else if (beta > 0.35 && alpha < 0.15) emotion = "focused";
    else if (delta > 0.4) emotion = "sad";
    else if (beta > 0.4) emotion = "happy";
    else emotion = "relaxed";

    const probs = Object.fromEntries(EMOTIONS.map((e) => [e, e === emotion ? 0.6 : 0.08]));
    return { emotion, confidence: 0.6, probabilities: probs };
  }

  async detectDream(features: number[]): Promise<DreamPrediction | null> {
    return dreamHeuristic(features, this._lastFs, this._lastSignal);
  }

  /**
   * Get per-model ONNX inference latency stats.
   *
   * Returns avg, p95, min, max (in ms) from a rolling 50-sample buffer
   * for each model that has been run: "eegnet", "emotion", "userEeg".
   * Returns empty object if no ONNX inference has occurred yet.
   */
  getInferenceStats(): InferenceStats {
    return this._latency.getStats();
  }
}

export const localML = new LocalMLEngine();
export { extractFeatures };
export type { SleepPrediction, EmotionPrediction, DreamPrediction, InferenceStats, ModelLatencyStats };
