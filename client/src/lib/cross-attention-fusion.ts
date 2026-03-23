/**
 * Cross-attention fusion for EEG + Voice emotion (client-side ONNX).
 *
 * Loads the cross_attention_fusion.onnx model and runs inference using
 * onnxruntime-web. Falls back to weighted averaging when ONNX is unavailable.
 *
 * Research basis: MMHA-FNN (2025) showed multi-head attention fusion
 * achieved 81.14% vs concatenation 71.02% -- a +10% gain.
 *
 * Usage:
 *   import { fuseCrossAttention, isCrossAttentionReady } from "./cross-attention-fusion";
 *
 *   const result = await fuseCrossAttention(eegProbs, voiceProbs);
 *   if (result) { ... }
 */

import * as ort from "onnxruntime-web";

// ── Types ──────────────────────────────────────────────────────────────────

export interface CrossAttentionResult {
  emotion: string;
  probabilities: Record<string, number>;
  confidence: number;
  model_type: "cross_attention_onnx" | "cross_attention_fallback";
}

// ── Constants ──────────────────────────────────────────────────────────────

const EMOTIONS_6 = ["happy", "sad", "angry", "fear", "surprise", "neutral"] as const;
const VOICE_5 = ["happy", "sad", "angry", "fear", "surprise"] as const;

const ONNX_MODEL_PATH = "/models/cross_attention_fusion.onnx";

// ── ONNX Session Management ───────────────────────────────────────────────

let _session: ort.InferenceSession | null = null;
let _loadAttempted = false;
let _loadFailed = false;

/**
 * Load the cross-attention ONNX model. Cached after first successful load.
 * Returns null if the model file is unavailable or ONNX Runtime fails.
 */
async function getSession(): Promise<ort.InferenceSession | null> {
  if (_session) return _session;
  if (_loadFailed) return null;
  if (_loadAttempted) return null;

  _loadAttempted = true;
  try {
    _session = await ort.InferenceSession.create(ONNX_MODEL_PATH, {
      executionProviders: ["wasm"],
    });
    return _session;
  } catch {
    _loadFailed = true;
    return null;
  }
}

/** Check whether the ONNX model has been loaded successfully. */
export function isCrossAttentionReady(): boolean {
  return _session !== null;
}

/** Preload the ONNX model (call during app initialization). */
export async function preloadCrossAttention(): Promise<boolean> {
  const session = await getSession();
  return session !== null;
}

// ── Entropy-based confidence ──────────────────────────────────────────────

function entropyConfidence(probs: number[]): number {
  const eps = 1e-10;
  const p = probs.map((v) => Math.max(v, eps));
  const sum = p.reduce((a, b) => a + b, 0);
  const normalized = p.map((v) => v / sum);
  const entropy = -normalized.reduce((acc, pi) => acc + pi * Math.log(pi), 0);
  const maxEntropy = Math.log(p.length);
  if (maxEntropy < eps) return 1.0;
  return Math.max(0, Math.min(1, 1 - entropy / maxEntropy));
}

// ── Weighted average fallback ─────────────────────────────────────────────

function fallbackFusion(
  eegProbs: Record<string, number>,
  voiceProbs: Record<string, number>,
): CrossAttentionResult {
  const eegConf = Math.max(...Object.values(eegProbs));
  const voiceConf = Math.max(...Object.values(voiceProbs));
  const total = eegConf + voiceConf || 1;
  const eegW = eegConf / total;
  const voiceW = voiceConf / total;

  const fused: Record<string, number> = {};
  for (const emo of EMOTIONS_6) {
    fused[emo] = eegW * (eegProbs[emo] ?? 0) + voiceW * (voiceProbs[emo] ?? 0);
  }

  // Normalize
  const fusedTotal = Object.values(fused).reduce((a, b) => a + b, 0) || 1;
  for (const emo of EMOTIONS_6) {
    fused[emo] = Math.round((fused[emo] / fusedTotal) * 10000) / 10000;
  }

  const probArray = EMOTIONS_6.map((e) => fused[e]);
  const bestIdx = probArray.indexOf(Math.max(...probArray));
  const confidence = entropyConfidence(probArray);

  return {
    emotion: EMOTIONS_6[bestIdx],
    probabilities: fused,
    confidence: Math.round(confidence * 10000) / 10000,
    model_type: "cross_attention_fallback",
  };
}

// ── Main fusion function ──────────────────────────────────────────────────

/**
 * Fuse EEG and voice emotion probabilities using cross-attention ONNX model.
 *
 * Falls back to confidence-weighted averaging when the ONNX model is not
 * available (first load, network error, etc.).
 *
 * @param eegProbs - 6-class EEG emotion probabilities
 * @param voiceProbs - 5-class voice emotion probabilities
 * @returns Fused emotion result
 */
export async function fuseCrossAttention(
  eegProbs: Record<string, number>,
  voiceProbs: Record<string, number>,
): Promise<CrossAttentionResult> {
  const session = await getSession();

  if (!session) {
    return fallbackFusion(eegProbs, voiceProbs);
  }

  try {
    // Build input tensors in canonical order
    const eegData = new Float32Array(EMOTIONS_6.map((e) => eegProbs[e] ?? 0));
    const voiceData = new Float32Array(VOICE_5.map((e) => voiceProbs[e] ?? 0));

    const eegTensor = new ort.Tensor("float32", eegData, [1, 6]);
    const voiceTensor = new ort.Tensor("float32", voiceData, [1, 5]);

    const results = await session.run({
      eeg_probs: eegTensor,
      voice_probs: voiceTensor,
    });

    const logits = results.fused_logits.data as Float32Array;

    // Softmax
    const maxLogit = Math.max(...logits);
    const expLogits = Array.from(logits).map((v) => Math.exp(v - maxLogit));
    const expSum = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map((v) => v / expSum);

    const probDict: Record<string, number> = {};
    for (let i = 0; i < EMOTIONS_6.length; i++) {
      probDict[EMOTIONS_6[i]] = Math.round(probs[i] * 10000) / 10000;
    }

    const bestIdx = probs.indexOf(Math.max(...probs));
    const confidence = entropyConfidence(probs);

    return {
      emotion: EMOTIONS_6[bestIdx],
      probabilities: probDict,
      confidence: Math.round(confidence * 10000) / 10000,
      model_type: "cross_attention_onnx",
    };
  } catch {
    // ONNX inference failed -- fall back
    return fallbackFusion(eegProbs, voiceProbs);
  }
}
