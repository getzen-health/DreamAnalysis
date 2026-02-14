/**
 * Local ML Engine — runs ONNX models in the browser via onnxruntime-web.
 *
 * Loads ONNX models from /models/ (served statically by Vite) and provides
 * the same prediction interface as the server API. Falls back gracefully
 * if models are not available.
 */

import { extractFeatures } from "./eeg-features";

// onnxruntime-web is loaded dynamically to avoid hard failures if not installed
let ort: typeof import("onnxruntime-web") | null = null;

async function loadOrt() {
  if (ort) return ort;
  try {
    ort = await import("onnxruntime-web");
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

type InferenceSession = InstanceType<typeof import("onnxruntime-web").InferenceSession>;

class LocalMLEngine {
  private sleepSession: InferenceSession | null = null;
  private emotionSession: InferenceSession | null = null;
  private dreamSession: InferenceSession | null = null;
  private _ready = false;
  private _initPromise: Promise<void> | null = null;

  async initialize(): Promise<void> {
    if (this._initPromise) return this._initPromise;

    this._initPromise = this._doInit();
    return this._initPromise;
  }

  private async _doInit(): Promise<void> {
    const ortModule = await loadOrt();
    if (!ortModule) {
      console.warn("onnxruntime-web not available, local inference disabled");
      return;
    }

    const modelPaths = [
      { name: "sleep", path: "/models/sleep_staging_model.onnx" },
      { name: "emotion", path: "/models/emotion_classifier_model.onnx" },
      { name: "dream", path: "/models/dream_detector_model.onnx" },
    ];

    for (const { name, path } of modelPaths) {
      try {
        const session = await ortModule.InferenceSession.create(path);
        if (name === "sleep") this.sleepSession = session;
        else if (name === "emotion") this.emotionSession = session;
        else if (name === "dream") this.dreamSession = session;
      } catch {
        // Model not available, that's fine
      }
    }

    this._ready =
      this.sleepSession !== null ||
      this.emotionSession !== null ||
      this.dreamSession !== null;
  }

  isReady(): boolean {
    return this._ready;
  }

  async analyzeSleep(features: number[]): Promise<SleepPrediction | null> {
    if (!this.sleepSession) return null;
    const ortModule = await loadOrt();
    if (!ortModule) return null;

    try {
      const input = new ortModule.Tensor("float32", new Float32Array(features), [1, features.length]);
      const inputName = this.sleepSession.inputNames[0];
      const results = await this.sleepSession.run({ [inputName]: input });
      const outputName = this.sleepSession.outputNames[0];
      const output = results[outputName];
      const data = output.data as Float32Array;

      // Find argmax
      let maxIdx = 0;
      let maxVal = data[0];
      for (let i = 1; i < data.length; i++) {
        if (data[i] > maxVal) {
          maxVal = data[i];
          maxIdx = i;
        }
      }

      // Softmax normalization
      const expVals = Array.from(data).map((v) => Math.exp(v - maxVal));
      const expSum = expVals.reduce((a, b) => a + b, 0);
      const probs = expVals.map((v) => v / expSum);

      return {
        stage: SLEEP_STAGES[maxIdx] || "Wake",
        stage_index: maxIdx,
        confidence: probs[maxIdx],
        probabilities: Object.fromEntries(
          SLEEP_STAGES.map((s, i) => [s, probs[i] || 0])
        ),
      };
    } catch {
      return null;
    }
  }

  async analyzeEmotion(features: number[]): Promise<EmotionPrediction | null> {
    if (!this.emotionSession) return null;
    const ortModule = await loadOrt();
    if (!ortModule) return null;

    try {
      const input = new ortModule.Tensor("float32", new Float32Array(features), [1, features.length]);
      const inputName = this.emotionSession.inputNames[0];
      const results = await this.emotionSession.run({ [inputName]: input });
      const outputName = this.emotionSession.outputNames[0];
      const output = results[outputName];
      const data = output.data as Float32Array;

      let maxIdx = 0;
      let maxVal = data[0];
      for (let i = 1; i < data.length; i++) {
        if (data[i] > maxVal) {
          maxVal = data[i];
          maxIdx = i;
        }
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
    } catch {
      return null;
    }
  }

  async detectDream(features: number[]): Promise<DreamPrediction | null> {
    if (!this.dreamSession) return null;
    const ortModule = await loadOrt();
    if (!ortModule) return null;

    try {
      const input = new ortModule.Tensor("float32", new Float32Array(features), [1, features.length]);
      const inputName = this.dreamSession.inputNames[0];
      const results = await this.dreamSession.run({ [inputName]: input });
      const outputName = this.dreamSession.outputNames[0];
      const output = results[outputName];
      const data = output.data as Float32Array;

      // Binary classification: [not_dreaming, dreaming]
      const prob = data.length >= 2 ? data[1] : data[0];

      return {
        is_dreaming: prob > 0.5,
        probability: prob,
      };
    } catch {
      return null;
    }
  }
}

export const localML = new LocalMLEngine();
export { extractFeatures };
export type { SleepPrediction, EmotionPrediction, DreamPrediction };
