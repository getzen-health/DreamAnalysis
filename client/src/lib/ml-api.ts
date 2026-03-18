import { Capacitor } from "@capacitor/core";
import { getParticipantId } from "@/lib/participant";

const ML_API_URL_DEFAULT =
  import.meta.env.VITE_ML_API_URL ||
  "http://localhost:8080";

// Clear any stale ngrok/localhost URLs saved in localStorage when on production.
if (typeof window !== "undefined" && window.location.hostname !== "localhost") {
  try {
    const stored = localStorage.getItem("ml_backend_url");
    if (stored && (stored.includes("ngrok") || stored.includes("localhost"))) {
      localStorage.removeItem("ml_backend_url");
    }
  } catch { /* ignore */ }
}

/** Reads the ML backend URL, handling web, native, and user overrides. */
export function getMLApiUrl(): string {
  // 1. User override from Settings (always wins)
  try {
    const stored = localStorage.getItem("ml_backend_url");
    if (stored?.trim()) return stored.trim().replace(/\/$/, "");
  } catch { /* SSR / private browsing */ }

  // 2. Native app: always use the build-time env var (Railway URL).
  //    Never fall back to localhost — it won't reach the dev machine.
  if (typeof window !== "undefined" && Capacitor.isNativePlatform()) {
    return ML_API_URL_DEFAULT;
  }

  // 3. Web localhost: use localhost:8080 for direct local dev
  if (typeof window !== "undefined" && window.location.hostname === "localhost") {
    return "http://localhost:8080";
  }

  // 4. Web production (Vercel): use env var
  return ML_API_URL_DEFAULT;
}

/** Extra headers needed when routing through ngrok (bypasses browser interstitial). */
function ngrokHeaders(): Record<string, string> {
  return getMLApiUrl().includes("ngrok") ? { "ngrok-skip-browser-warning": "true" } : {};
}
const EXPRESS_URL = import.meta.env.VITE_EXPRESS_URL || "";

// ─── Express API helpers ─────────────────────────────────────────────────

async function expressFetch<T>(path: string): Promise<T> {
  const response = await fetch(`${EXPRESS_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
  });
  if (!response.ok) {
    throw new Error(`Express API error: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

// ─── Brain history types ─────────────────────────────────────────────────

export interface StoredEmotionReading {
  id: string;
  userId: string | null;
  sessionId: string | null;
  stress: number;
  happiness: number;
  focus: number;
  energy: number;
  dominantEmotion: string;
  valence: number | null;
  arousal: number | null;
  timestamp: string;
}

export interface TodayTotals {
  userId: string;
  count: number;
  avgStress: number | null;
  avgFocus: number | null;
  avgHappiness: number | null;
  avgEnergy: number | null;
  avgValence: number | null;
  avgArousal: number | null;
  dominantEmotion: string | null;
}

export interface YesterdayComparison {
  userId: string;
  count: number;
  windowStart: string;
  windowEnd: string;
  avgStress: number | null;
  avgFocus: number | null;
  avgHappiness: number | null;
  avgEnergy: number | null;
  avgValence: number | null;
}

// ─── Brain history fetch functions ──────────────────────────────────────

export async function getEmotionHistory(userId: string, days: number = 1): Promise<StoredEmotionReading[]> {
  return expressFetch<StoredEmotionReading[]>(`/api/brain/history/${encodeURIComponent(userId)}?days=${days}`);
}

export async function getTodayTotals(userId: string): Promise<TodayTotals> {
  return expressFetch<TodayTotals>(`/api/brain/today-totals/${encodeURIComponent(userId)}`);
}

export async function getAtThisTimeYesterday(userId: string): Promise<YesterdayComparison> {
  return expressFetch<YesterdayComparison>(`/api/brain/at-this-time-yesterday/${encodeURIComponent(userId)}`);
}

interface CrossChannelMetrics {
  n_channels: number;
  coherence_alpha?: number;
  plv_alpha?: number;
}

interface SignalQuality {
  sqi: number;
  artifacts_detected: string[];
  clean_ratio: number;
  rejected_epochs: number[];
  channel_quality: number[];
}

interface AnomalyResult {
  is_anomaly: boolean;
  anomaly_score: number;
  spikes_detected: number;
  seizure_probability: number;
  alert_level: "normal" | "watch" | "warning" | "critical";
}

interface PersonalModelResult {
  has_personal: boolean;
  personal_prediction?: string;
  personal_confidence?: number;
  personal_probabilities?: Record<string, number>;
}

interface EEGAnalysisResult {
  sleep_stage: {
    stage: string;
    stage_index: number;
    confidence: number;
    probabilities: Record<string, number>;
  };
  emotions: {
    emotion: string;
    confidence: number;
    probabilities: Record<string, number>;
    valence: number;
    arousal: number;
    stress_index: number;
    focus_index: number;
    relaxation_index: number;
    band_powers: Record<string, number>;
  };
  dream_detection: {
    is_dreaming: boolean;
    probability: number;
    rem_likelihood: number;
    dream_intensity: number;
    lucidity_estimate: number;
  };
  features: Record<string, number>;
  band_powers: Record<string, number>;
  cross_channel?: CrossChannelMetrics;
  signal_quality?: SignalQuality;
  anomaly?: AnomalyResult;
  personal?: PersonalModelResult;
  // Simple signal quality fields for dashboard badge
  signal_quality_score?: number;   // 0-100
  artifact_detected?: boolean;
  artifact_type?: "clean" | "blink" | "muscle" | "electrode_pop";
  /** True when >= 4 seconds of EEG have been buffered; emotion accuracy is degraded below this. */
  epoch_ready?: boolean;
}

interface SimulationResult {
  signals: number[][];
  fs: number;
  state: string;
  duration: number;
  timestamps: number[];
  analysis: {
    sleep_stage: EEGAnalysisResult["sleep_stage"];
    emotions: EEGAnalysisResult["emotions"];
    dream_detection: EEGAnalysisResult["dream_detection"];
  };
}

interface ModelsStatus {
  sleep_staging: { loaded: boolean; type: string };
  emotion_classifier: { loaded: boolean; type: string };
  dream_detector: { loaded: boolean; type: string };
  available_states: string[];
}

interface DeviceInfo {
  type: string;
  name: string;
  channels: number;
  sample_rate: number;
  available: boolean;
}

interface DeviceListResponse {
  brainflow_available: boolean;
  devices: DeviceInfo[];
  connected?: boolean;
  message?: string;
}

interface DeviceStatusResponse {
  connected: boolean;
  streaming: boolean;
  device_type: string | null;
  n_channels: number;
  sample_rate: number;
  brainflow_available: boolean;
}

interface BenchmarkResult {
  model_name: string;
  dataset: string;
  accuracy: number;
  f1_macro: number;
  per_class: Record<string, { precision: number; recall: number; f1: number; support: number }>;
  confusion_matrix: number[][];
  inference_time_ms?: number;
  n_test_samples?: number;
}

// Wavelet analysis types
interface WaveletResult {
  spectrogram: {
    coefficients: number[][];
    frequencies: number[];
    times: number[];
  };
  dwt_energies: Record<string, number>;
  events: {
    sleep_spindles: Array<{ start: number; end: number; amplitude: number }>;
    k_complexes: Array<{ time: number; amplitude: number }>;
  };
}

// Neurofeedback types
interface NeurofeedbackProtocol {
  name: string;
  description: string;
}

interface NeurofeedbackEvalResult {
  status: string;
  score?: number;
  reward?: boolean;
  feedback_value?: number;
  streak?: number;
  progress?: number;
  baseline?: number;
}

interface NeurofeedbackStopResult {
  status: string;
  stats: {
    total_rewards: number;
    reward_rate: number;
    avg_score: number;
    max_streak: number;
    total_evaluations: number;
  };
}

// Session types
interface SessionSummary {
  session_id: string;
  user_id: string;
  session_type: string;
  start_time: number;
  status: string;
  summary: {
    duration_sec?: number;
    n_frames?: number;
    n_channels?: number;
    n_samples?: number;
    avg_stress?: number;
    avg_focus?: number;
    avg_relaxation?: number;
    avg_flow?: number;
    avg_creativity?: number;
    avg_valence?: number;
    avg_arousal?: number;
    dominant_emotion?: string;
  };
}

// Connectivity types
interface ConnectivityResult {
  connectivity_matrix: number[][];
  graph_metrics: {
    clustering_coefficient: number;
    avg_path_length: number;
    small_world_index: number;
    hub_nodes: number[];
    modularity: number;
    degree_centrality?: number[];
  };
  directed_flow: {
    granger: {
      matrix: number[][];
      significant_pairs: Array<{ from: number; to: number; strength: number }>;
    };
    dtf_matrix: number[][];
    dominant_direction: string;
  };
}

// Calibration types
interface CalibrationStatus {
  calibrated?: boolean;
  n_samples?: number;
  personal_accuracy?: number;
  classes?: string[];
  personal_model_active?: boolean;
  total_sessions?: number;
  total_labeled_epochs?: number;
  head_accuracy_pct?: number;
  estimated_global_accuracy_pct?: number;
  accuracy_improvement_pct?: number;
  personalization_progress_pct?: number;
  activation_threshold_sessions?: number;
  personal_blend_weight_pct?: number;
  baseline_ready?: boolean;
  baseline_frames?: number;
  feature_priors?: {
    alpha_mean: number;
    beta_mean: number;
    theta_mean: number;
  };
  class_counts?: Record<string, number>;
  next_milestone?: number;
  message?: string;
}

/** Exponential backoff delays (ms) between successive retries. */
const RETRY_DELAYS = [1_000, 3_000, 9_000] as const;

/** Resolves after `ms` milliseconds (uses setTimeout so fake timers can control it). */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Sentinel to signal that an error must not be retried (4xx client errors). */
class NonRetryableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "NonRetryableError";
  }
}

async function mlFetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
  let lastError: Error = new Error("mlFetch: no attempts made");

  for (let attempt = 0; attempt <= RETRY_DELAYS.length; attempt++) {
    // Fresh AbortController for every attempt — 30-second hard timeout.
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30_000);

    try {
      const response = await fetch(`${getMLApiUrl()}/api${endpoint}`, {
        ...options,
        signal: controller.signal,
        headers: {
          "Content-Type": "application/json",
          ...ngrokHeaders(),
          ...options?.headers,
        },
      });

      if (!response.ok) {
        if (response.status >= 400 && response.status < 500) {
          // 4xx — client error, extract detail and throw immediately without retry
          let errorMsg = `Request failed (${response.status})`;
          try {
            const body = await response.clone().json();
            const detail = body?.detail;
            if (typeof detail === "string") errorMsg = detail;
            else if (Array.isArray(detail)) {
              errorMsg = detail.map((d: { msg?: string }) => d.msg).join("; ");
            }
          } catch { /* ignore parse errors, use default message */ }
          throw new NonRetryableError(errorMsg);
        }
        // 5xx — extract detail message if available (e.g. 503 "Bluetooth not available")
        // then store as lastError for retry logic; 503 with a detail message means
        // the server understood the request but can't fulfill it — no point retrying.
        let serverMsg: string | null = null;
        try {
          const body = await response.clone().json();
          const detail = body?.detail;
          if (typeof detail === "string") serverMsg = detail;
          else if (Array.isArray(detail)) serverMsg = detail.map((d: { msg?: string }) => d.msg).join("; ");
        } catch { /* ignore */ }
        if (serverMsg) {
          // A descriptive server error — surface immediately, don't retry.
          throw new NonRetryableError(serverMsg);
        }
        lastError = new Error(`Request failed (${response.status})`);
      } else {
        return response.json() as Promise<T>;
      }
    } catch (err) {
      if (err instanceof NonRetryableError) {
        // 4xx: propagate immediately as a plain Error
        throw new Error(err.message);
      }
      lastError = err instanceof Error ? err : new Error("Unknown fetch error");
    } finally {
      clearTimeout(timeoutId);
    }

    // Wait before the next retry (skip wait after the final attempt)
    if (attempt < RETRY_DELAYS.length) {
      await sleep(RETRY_DELAYS[attempt]);
    }
  }

  throw lastError;
}

async function mlFetchRaw(endpoint: string, options?: RequestInit): Promise<string> {
  const response = await fetch(`${getMLApiUrl()}/api${endpoint}`, {
    ...options,
    headers: { ...ngrokHeaders(), ...((options?.headers as Record<string, string>) ?? {}) },
  });
  if (!response.ok) {
    throw new Error(`ML API error: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

// ─── Core Analysis ───────────────────────────────────────────────────────

export async function analyzeEEG(
  signals: number[][],
  fs: number = 256
): Promise<EEGAnalysisResult> {
  return mlFetch<EEGAnalysisResult>("/analyze-eeg", {
    method: "POST",
    body: JSON.stringify({ signals, fs }),
  });
}

export async function simulateEEG(
  state: string = "rest",
  duration: number = 30,
  fs: number = 256,
  nChannels: number = 1
): Promise<SimulationResult> {
  return mlFetch<SimulationResult>("/simulate-eeg", {
    method: "POST",
    body: JSON.stringify({
      state,
      duration,
      fs,
      n_channels: nChannels,
    }),
  });
}

export async function getModelsStatus(): Promise<ModelsStatus> {
  return mlFetch<ModelsStatus>("/models/status");
}

export async function getModelsBenchmarks(): Promise<Record<string, BenchmarkResult>> {
  return mlFetch<Record<string, BenchmarkResult>>("/models/benchmarks");
}

// ─── Wavelet Analysis (Phase 5) ─────────────────────────────────────────

export async function analyzeWavelet(
  signals: number[][],
  fs: number = 256
): Promise<WaveletResult> {
  return mlFetch<WaveletResult>("/analyze-wavelet", {
    method: "POST",
    body: JSON.stringify({ signals, fs }),
  });
}

// ─── Signal Quality (Phase 6) ───────────────────────────────────────────

export async function cleanSignal(
  signals: number[][],
  fs: number = 256
): Promise<{
  cleaned_signals: number[][];
  removed_components: number[];
  before_sqi: number[];
  after_sqi: number[];
  improvement: number;
}> {
  return mlFetch("/clean-signal", {
    method: "POST",
    body: JSON.stringify({ signals, fs }),
  });
}

// ─── Neurofeedback (Phase 7) ────────────────────────────────────────────

export async function getNeurofeedbackProtocols(): Promise<Record<string, NeurofeedbackProtocol>> {
  return mlFetch<Record<string, NeurofeedbackProtocol>>("/neurofeedback/protocols");
}

export async function startNeurofeedback(
  protocolType: string = "alpha_up",
  calibrate: boolean = true,
  threshold?: number
): Promise<{ status: string; protocol: string }> {
  return mlFetch("/neurofeedback/start", {
    method: "POST",
    body: JSON.stringify({
      protocol_type: protocolType,
      calibrate,
      threshold,
    }),
  });
}

export async function evaluateNeurofeedback(
  bandPowers: Record<string, number>,
  channelPowers?: Record<string, number>[]
): Promise<NeurofeedbackEvalResult> {
  return mlFetch<NeurofeedbackEvalResult>("/neurofeedback/evaluate", {
    method: "POST",
    body: JSON.stringify({
      band_powers: bandPowers,
      channel_powers: channelPowers,
    }),
  });
}

export async function stopNeurofeedback(): Promise<NeurofeedbackStopResult> {
  return mlFetch<NeurofeedbackStopResult>("/neurofeedback/stop", { method: "POST" });
}

// ─── Session Recording (Phase 8) ────────────────────────────────────────

export async function startSession(
  sessionType: string = "general",
  userId: string
): Promise<{ status: string; session_id: string }> {
  return mlFetch("/sessions/start", {
    method: "POST",
    body: JSON.stringify({ user_id: userId, session_type: sessionType }),
  });
}

export async function stopSession(userId: string): Promise<Record<string, unknown>> {
  return mlFetch(`/sessions/stop?user_id=${encodeURIComponent(userId)}`, { method: "POST" });
}

export async function listSessions(
  userId?: string,
  sessionType?: string
): Promise<SessionSummary[]> {
  const resolvedUserId = userId ?? getParticipantId();
  const params = new URLSearchParams();
  params.set("user_id", resolvedUserId);
  if (sessionType) params.set("session_type", sessionType);
  const qs = params.toString();
  return mlFetch<SessionSummary[]>(`/sessions${qs ? `?${qs}` : ""}`);
}

export async function getSession(sessionId: string, userId?: string): Promise<Record<string, unknown>> {
  const resolvedUserId = userId ?? getParticipantId();
  return mlFetch(`/sessions/${sessionId}?user_id=${encodeURIComponent(resolvedUserId)}`);
}

export async function deleteSession(sessionId: string, userId?: string): Promise<{ status: string }> {
  const resolvedUserId = userId ?? getParticipantId();
  return mlFetch(`/sessions/${sessionId}?user_id=${encodeURIComponent(resolvedUserId)}`, { method: "DELETE" });
}

export async function exportSession(sessionId: string, format: string = "csv", userId?: string): Promise<string> {
  const resolvedUserId = userId ?? getParticipantId();
  return mlFetchRaw(`/sessions/${sessionId}/export?format=${format}&user_id=${encodeURIComponent(resolvedUserId)}`);
}

// ─── Baseline Calibration (resting-state normalisation) ─────────────────

export interface BaselineFrameResult {
  status: string;
  n_frames: number;
  ready: boolean;
  message: string;
}

export interface BaselineStatusResult {
  n_frames: number;
  ready: boolean;
  n_features: number;
}

/** Send one second of raw EEG to the baseline calibrator. */
export async function addBaselineFrame(
  signals: number[][],
  userId?: string,
  fs: number = 256
): Promise<BaselineFrameResult> {
  const resolvedUserId = userId ?? getParticipantId();
  return mlFetch<BaselineFrameResult>("/calibration/baseline/add-frame", {
    method: "POST",
    body: JSON.stringify({ signals, fs, user_id: resolvedUserId }),
  });
}

export async function getBaselineStatus(
  userId?: string
): Promise<BaselineStatusResult> {
  const resolvedUserId = userId ?? getParticipantId();
  return mlFetch<BaselineStatusResult>(
    `/calibration/baseline/status?user_id=${encodeURIComponent(resolvedUserId)}`
  );
}

export async function resetBaselineCalibration(
  userId?: string
): Promise<{ status: string; message: string }> {
  const resolvedUserId = userId ?? getParticipantId();
  return mlFetch("/calibration/baseline/reset", {
    method: "POST",
    body: JSON.stringify({ user_id: resolvedUserId }),
  });
}

// ─── Calibration & Personal Models (Phase 9) ────────────────────────────

export async function startCalibration(): Promise<{
  status: string;
  steps: Array<{ step: number; instruction: string; label: string; duration_sec: number }>;
}> {
  return mlFetch("/calibration/start", { method: "POST" });
}

export async function submitCalibration(
  signalsList: number[][][],
  labels: string[],
  fs: number = 256
): Promise<{ calibrated: boolean; n_samples: number; personal_accuracy: number }> {
  return mlFetch("/calibration/submit", {
    method: "POST",
    body: JSON.stringify({ signals_list: signalsList, labels, fs }),
  });
}

export async function submitFeedback(
  signals: number[][] | null | undefined,
  predictedLabel: string,
  correctLabel: string,
  userId?: string
): Promise<{ updated: boolean }> {
  const resolvedUserId = userId ?? getParticipantId();
  // Send signals only when a non-empty array is provided; null tells the backend
  // this is a label-only correction (no raw EEG attached). The backend still
  // counts label-only corrections toward the 5-session fine-tuning threshold.
  const signalsPayload = signals && signals.length > 0 ? signals : null;
  return mlFetch("/feedback", {
    method: "POST",
    body: JSON.stringify({
      user_id: resolvedUserId,
      signals: signalsPayload,
      predicted_label: predictedLabel,
      correct_label: correctLabel,
    }),
  });
}

export async function getCalibrationStatus(
  userId?: string
): Promise<CalibrationStatus> {
  const resolvedUserId = userId ?? getParticipantId();
  return mlFetch<CalibrationStatus>(`/calibration/status?user_id=${resolvedUserId}`);
}

// ─── Connectivity (Phase 10) ────────────────────────────────────────────

export async function analyzeConnectivity(
  signals: number[][],
  fs: number = 256
): Promise<ConnectivityResult> {
  return mlFetch<ConnectivityResult>("/analyze-connectivity", {
    method: "POST",
    body: JSON.stringify({ signals, fs }),
  });
}

// ─── Anomaly Detection (Phase 11) ───────────────────────────────────────

export async function setAnomalyBaseline(
  featuresList: Record<string, number>[]
): Promise<{ fitted: boolean; n_samples: number }> {
  return mlFetch("/anomaly/set-baseline", {
    method: "POST",
    body: JSON.stringify({ features_list: featuresList }),
  });
}

// ─── Device Management ──────────────────────────────────────────────────

export async function listDevices(): Promise<DeviceListResponse> {
  return mlFetch<DeviceListResponse>("/devices");
}

export async function connectDevice(
  deviceType: string,
  params?: Record<string, string>
): Promise<Record<string, unknown>> {
  return mlFetch("/devices/connect", {
    method: "POST",
    body: JSON.stringify({ device_type: deviceType, params }),
  });
}

export async function disconnectDevice(): Promise<Record<string, string>> {
  return mlFetch("/devices/disconnect", { method: "POST" });
}

export async function getDeviceStatus(): Promise<DeviceStatusResponse> {
  return mlFetch<DeviceStatusResponse>("/devices/status");
}

export async function startDeviceStream(): Promise<Record<string, unknown>> {
  return mlFetch("/devices/start-stream", { method: "POST" });
}

export async function stopDeviceStream(): Promise<Record<string, string>> {
  return mlFetch("/devices/stop-stream", { method: "POST" });
}

export function getWebSocketUrl(): string {
  const url = getMLApiUrl();
  const wsProtocol = url.startsWith("https") ? "wss" : "ws";
  const host = url.replace(/^https?:\/\//, "");
  const base = `${wsProtocol}://${host}/ws/eeg-stream`;
  // ngrok requires this query param to skip the browser-warning interstitial
  return url.includes("ngrok") ? `${base}?ngrok-skip-browser-warning=true` : base;
}

/** Pings ML backend /health with a timeout. Returns true if reachable. */
export async function pingBackend(timeoutMs = 12_000): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const res = await fetch(`${getMLApiUrl()}/health`, {
      signal: controller.signal,
      headers: { ...ngrokHeaders() },
    });
    clearTimeout(timer);
    return res.ok;
  } catch {
    return false;
  }
}

// ─── Health Integration ─────────────────────────────────────────────────

interface HealthSample {
  metric: string;
  value: number;
  timestamp: string;
  source: string;
  metadata?: Record<string, unknown>;
}

interface HealthDailySummary {
  date: string;
  health: Record<string, { avg: number; min: number; max: number; count: number }>;
  brain: Record<string, number>;
}

interface HealthInsight {
  insight_type: string;
  title: string;
  description: string;
  correlation_strength: number;
  evidence_count: number;
  brain_metric: string;
  health_metric: string;
}

interface HealthTrend {
  date: string;
  flow_score: number | null;
  creativity_score: number | null;
  encoding_score: number | null;
  valence: number | null;
  arousal: number | null;
}

export async function ingestHealthData(
  userId: string,
  source: "apple_health" | "google_fit" | "health_connect",
  data: Record<string, unknown>
): Promise<{ stored: number; metrics: string[] }> {
  return mlFetch("/health/ingest", {
    method: "POST",
    body: JSON.stringify({ user_id: userId, source, data }),
  });
}

export async function getHealthDailySummary(
  userId: string,
  date?: string
): Promise<HealthDailySummary> {
  const params = date ? `?date=${date}` : "";
  return mlFetch<HealthDailySummary>(`/health/daily-summary/${userId}${params}`);
}

export async function getHealthInsights(
  userId: string
): Promise<HealthInsight[]> {
  const resp = await mlFetch<{ insights: HealthInsight[] }>(`/health/insights/${userId}`);
  return resp.insights ?? [];
}

export async function getHealthTrends(
  userId: string,
  days: number = 30
): Promise<HealthTrend[]> {
  const resp = await mlFetch<{ trends: HealthTrend[] }>(`/health/trends/${userId}?days=${days}`);
  return resp.trends ?? [];
}

export async function getSupportedHealthMetrics(): Promise<Record<string, string[]>> {
  return mlFetch<Record<string, string[]>>("/health/supported-metrics");
}

// ─── Weekly Report & Session Trends ─────────────────────────────────────

interface WeeklyReport {
  user_id: string;
  period_start: string;
  period_end: string;
  total_sessions: number;
  avg_stress: number;
  avg_focus: number;
  avg_flow: number;
  avg_relaxation: number;
  avg_creativity: number;
  stress_change: number;
  focus_change: number;
  flow_change: number;
  relaxation_change: number;
  creativity_change: number;
}

interface SessionTrend {
  session_id: string;
  date: string;
  avg_stress: number;
  avg_focus: number;
  avg_flow: number;
  avg_relaxation: number;
  avg_creativity: number;
}

interface SessionTrends {
  user_id: string;
  trends: SessionTrend[];
}

export async function getWeeklyReport(userId: string): Promise<WeeklyReport> {
  return mlFetch<WeeklyReport>(`/sessions/weekly-report?user_id=${encodeURIComponent(userId)}`);
}

export async function getSessionTrends(
  userId: string,
  lastN?: number
): Promise<SessionTrends> {
  const params = new URLSearchParams({ user_id: userId });
  if (lastN !== undefined) params.set("last_n", String(lastN));
  return mlFetch<SessionTrends>(`/sessions/trends?${params.toString()}`);
}

export async function exportToHealthKit(
  userId: string
): Promise<{ exported_records: number }> {
  return mlFetch("/health/export-to-healthkit/" + userId, { method: "POST" });
}

// ─── Brain Readiness Score ───────────────────────────────────────────────

export interface ReadinessFactors {
  sleep_quality: number | null;
  stress_avg: number | null;
  hrv_trend: number | null;
  voice_emotion: number | null;
}

export interface ReadinessHistoryPoint {
  date: string;
  score: number | null;
}

export interface ReadinessScoreResult {
  user_id: string;
  score: number;
  factors: ReadinessFactors;
  history: ReadinessHistoryPoint[];
  color: "red" | "yellow" | "green";
  label: string;
}

export async function getReadinessScore(
  userId: string
): Promise<ReadinessScoreResult> {
  return mlFetch<ReadinessScoreResult>(
    `/brain-report/readiness-score/${encodeURIComponent(userId)}`
  );
}

// ─── Habit Streak ────────────────────────────────────────────────────────

export interface StreakResult {
  user_id: string;
  current_streak: number;
  best_streak: number;
  today_checked_in: boolean;
  milestones: number[];
  next_milestone: number | null;
  total_checkins: number;
}

export async function getBrainStreak(userId: string): Promise<StreakResult> {
  return mlFetch<StreakResult>(
    `/brain-report/streak/${encodeURIComponent(userId)}`
  );
}

export async function recordStreakCheckin(
  userId: string
): Promise<{ status: string; current_streak: number; best_streak: number }> {
  return mlFetch(
    `/brain-report/streak/${encodeURIComponent(userId)}/checkin`,
    { method: "POST" }
  );
}

// ---------------------------------------------------------------------------
// Food Emotion
// ---------------------------------------------------------------------------

interface FoodEmotionComponents {
  faa: number;
  high_beta: number;
  prefrontal_theta: number;
  delta: number;
}

interface FoodRecommendations {
  avoid: string[];
  prefer: string[];
  strategy: string;
  mindfulness_tip: string;
}

export interface FoodEmotionResult {
  food_state: string;
  confidence: number;
  state_probabilities: Record<string, number>;
  recommendations: FoodRecommendations;
  components: FoodEmotionComponents;
  band_powers: Record<string, number>;
  faa: number;
  is_calibrated: boolean;
  calibration_progress: number;
  simulation_mode?: boolean;
}

export async function predictFoodEmotion(
  eegData?: number[][]
): Promise<FoodEmotionResult> {
  const body = eegData
    ? { signals: eegData, fs: 256.0 }
    : { simulate: true, state: "rest", fs: 256.0 };
  return mlFetch<FoodEmotionResult>("/predict-food-emotion", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function calibrateFoodEmotion(
  eegData?: number[][]
): Promise<{ calibrated: boolean; simulation_mode?: boolean }> {
  const body = eegData
    ? { signals: eegData, fs: 256.0 }
    : { simulate: true, state: "rest", fs: 256.0 };
  return mlFetch<{ calibrated: boolean; simulation_mode?: boolean }>("/food-emotion/calibrate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function getFoodRecommendations(
  foodState: string
): Promise<FoodRecommendations> {
  return mlFetch<FoodRecommendations>(
    `/food-emotion/recommendations/${encodeURIComponent(foodState)}`
  );
}

// ── Multimodal Fusion ────────────────────────────────────────────────────────

export interface MultimodalStatus {
  eeg_model_loaded: boolean;
  audio_model_loaded: boolean;
  video_model_loaded: boolean;
  n_modalities: number;
  fusion_weights: { eeg: number; audio: number; video: number };
  ready: boolean;
}

export async function getMultimodalStatus(): Promise<MultimodalStatus> {
  return mlFetch<MultimodalStatus>("/multimodal/status");
}

export type {
  EEGAnalysisResult,
  SimulationResult,
  ModelsStatus,
  CrossChannelMetrics,
  SignalQuality,
  AnomalyResult,
  PersonalModelResult,
  DeviceInfo,
  DeviceListResponse,
  DeviceStatusResponse,
  BenchmarkResult,
  WaveletResult,
  NeurofeedbackProtocol,
  NeurofeedbackEvalResult,
  NeurofeedbackStopResult,
  SessionSummary,
  ConnectivityResult,
  CalibrationStatus,
  HealthSample,
  HealthDailySummary,
  HealthInsight,
  HealthTrend,
  WeeklyReport,
  SessionTrend,
  SessionTrends,
};

// ─── Voice + Watch emotion analysis ─────────────────────────────────────────

export interface WatchBiometrics {
  hr?: number;    // heart rate bpm
  hrv?: number;   // HRV SDNN ms
  spo2?: number;  // SpO2 percentage
}

export interface VoiceWatchEmotionResult {
  emotion: string;
  probabilities: Record<string, number>;
  valence: number;
  arousal: number;
  confidence: number;
  model_type: string;
  stress_from_watch: number | null;
  stress_index?: number;
  focus_index?: number;
  biomarkers?: Record<string, number>;
}

export interface VoiceWatchStatus {
  audio_model_loaded: boolean;
  librosa_available: boolean;
  ready: boolean;
  fusion_weights: { audio: number; watch: number };
}

export async function analyzeVoiceWatch(
  audioBase64: string,
  watch: WatchBiometrics = {},
  userId?: string
): Promise<VoiceWatchEmotionResult> {
  return mlFetch<VoiceWatchEmotionResult>("/voice-watch/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      audio_b64: audioBase64,
      hr:   watch.hr,
      hrv:  watch.hrv,
      spo2: watch.spo2,
      ...(userId ? { user_id: userId } : {}),
    }),
  });
}

export async function getVoiceWatchStatus(): Promise<VoiceWatchStatus> {
  return mlFetch<VoiceWatchStatus>("/voice-watch/status");
}

// ─── EEG Session Data Storage (Training Pipeline) ────────────────────────

export interface SaveEEGResult {
  saved: string;
  session_id: string;
  user_id: string;
  total_epochs: number;
  shape: number[];
}

export async function saveEEGEpoch(params: {
  signals: number[][];
  user_id?: string;
  session_id?: string;
  sample_rate?: number;
  device_type?: string;
  predicted_emotion?: string;
  band_powers?: Record<string, number>;
  frontal_asymmetry?: number;
  valence?: number;
  arousal?: number;
  signal_quality?: number;
}): Promise<SaveEEGResult> {
  return mlFetch<SaveEEGResult>("/sessions/save-eeg", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export interface TrainingDataStats {
  total_epochs: number;
  users: Record<string, {
    total_epochs: number;
    labeled_epochs: number;
    n_sessions: number;
    emotion_distribution: Record<string, number>;
  }>;
  ready_to_train: boolean;
}

export async function getTrainingDataStats(
  userId?: string
): Promise<TrainingDataStats> {
  const qs = userId ? `?user_id=${encodeURIComponent(userId)}` : "";
  return mlFetch<TrainingDataStats>(`/sessions/training-data/stats${qs}`);
}

// ── Supplement Tracker ─────────────────────────────────────────────────────

export interface SupplementLogEntry {
  id: string;
  name: string;
  type: string;
  dosage: number;
  unit: string;
  timestamp: number;
  notes: string;
}

export interface ActiveSupplement {
  name: string;
  type: string;
  dosage: number;
  unit: string;
  taken_at: number;
  hours_ago: number;
}

export interface SupplementVerdict {
  name: string;
  verdict: "positive" | "negative" | "neutral" | "insufficient_data";
  n_exposures: number;
  valence_shift: number;
  stress_shift: number;
  focus_shift: number;
  alpha_beta_shift: number;
}

export interface SupplementReport {
  user_id: string;
  n_supplements: number;
  supplements: SupplementVerdict[];
}

export async function logSupplement(params: {
  user_id: string;
  name: string;
  type: string;
  dosage: number;
  unit: string;
  notes?: string;
}): Promise<{ entry_id: string; logged_at: number }> {
  return mlFetch("/supplements/log", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export async function getSupplementLog(
  userId: string,
  lastN = 50
): Promise<{ user_id: string; count: number; entries: SupplementLogEntry[] }> {
  return mlFetch(`/supplements/log/${encodeURIComponent(userId)}?last_n=${lastN}`);
}

export async function getSupplementReport(
  userId: string
): Promise<SupplementReport> {
  return mlFetch(`/supplements/report/${encodeURIComponent(userId)}`);
}

export async function getActiveSupplements(
  userId: string,
  hours = 24
): Promise<{ user_id: string; hours: number; count: number; supplements: ActiveSupplement[] }> {
  return mlFetch(`/supplements/active/${encodeURIComponent(userId)}?hours=${hours}`);
}

// ── EI Composite ──────────────────────────────────────────────────────────────

export interface EIQDimensions {
  self_perception: number;
  self_expression: number;
  interpersonal: number;
  decision_making: number;
  stress_management: number;
}

export interface EIQResult {
  eiq_score: number;
  eiq_grade: string;
  dimensions: EIQDimensions;
  strengths: string[];
  growth_areas: string[];
  has_baseline: boolean;
  processed_at?: number;
}

export interface EIQSessionStats {
  n_assessments: number;
  mean_eiq: number | null;
  trend: "improving" | "declining" | "stable" | null;
}

export async function getEIQSessionStats(userId: string): Promise<EIQSessionStats> {
  return mlFetch<EIQSessionStats>(`/ei-composite/session-stats/${encodeURIComponent(userId)}`);
}

export async function getEIQHistory(
  userId: string,
  limit = 30
): Promise<{ user_id: string; count: number; history: EIQResult[] }> {
  return mlFetch(`/ei-composite/history/${encodeURIComponent(userId)}?limit=${limit}`);
}

// ─── Voice Watch Check-In ─────────────────────────────────────────────────────

export interface VoiceWatchCheckinResult {
  checkin_id: string;
  checkin_type: "morning" | "noon" | "evening";
  emotion: string;
  valence: number;
  arousal: number;
  confidence: number;
  stress_index: number;
  focus_index: number;
  model_type: string;
  timestamp: number;
  biomarkers?: Record<string, number>;
}

/** @deprecated Use VoiceWatchCheckinResult instead. */
export type CheckInResult = VoiceWatchCheckinResult;

/** Submit a voice analysis via the canonical voice-watch pipeline. */
export async function submitVoiceWatch(
  audioBase64: string,
  userId?: string
): Promise<VoiceWatchEmotionResult> {
  const resolvedUserId = userId ?? getParticipantId();
  return mlFetch<VoiceWatchEmotionResult>("/voice-watch/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ audio_b64: audioBase64, user_id: resolvedUserId }),
  });
}

// ─── Sleep-to-Mood Predictor ─────────────────────────────────────────────────

export interface SleepMoodPrediction {
  predicted_valence: number;       // -1 to 1
  predicted_arousal: number;       // 0 to 1
  predicted_stress_risk: number;   // 0 to 1
  predicted_focus_score: number;   // 0 to 1
  predicted_focus_window: string;  // e.g. "9:00am–12:00pm"
  confidence: number;
  key_factor: string;
  mood_label: "positive" | "neutral" | "challenging";
  sleep_score: number;             // 0-100
  timestamp: number;
}

export async function predictSleepMood(sleepData: {
  total_sleep_hours?: number;
  deep_sleep_pct?: number;
  rem_pct?: number;
  sleep_efficiency?: number;
  waso_minutes?: number;
  sleep_onset_latency?: number;
  hrv_ms?: number;
  user_id?: string;
}): Promise<SleepMoodPrediction> {
  return mlFetch<SleepMoodPrediction>("/sleep-mood/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(sleepData),
  });
}

export async function getSleepMoodHistory(userId: string, lastN = 14): Promise<{ predictions: SleepMoodPrediction[] }> {
  return mlFetch<{ predictions: SleepMoodPrediction[] }>(`/sleep-mood/history/${encodeURIComponent(userId)}?last_n=${lastN}`);
}

// ─── EI Growth Tracker ───────────────────────────────────────────────────────

export interface DailyEISnapshot {
  date: string;
  eiq_score: number;
  dimension_scores: Record<string, number>;
  data_sources: string[];
  confidence: number;
  timestamp: number;
}

export interface EIGrowthTrend {
  weekly_averages: number[];
  slope_per_week: number;
  trend: "improving" | "declining" | "stable";
  p_significant: boolean;
  weeks_of_data: number;
}

export interface EIMilestone {
  type: string;
  label: string;
  achieved_at: string;
  description: string;
}

export interface EITrajectory {
  predicted_weeks: number | null;
  confidence: number;
  current_eiq: number;
  target_eiq: number;
  trajectory_points: number[];
}

export async function addEISnapshot(params: {
  user_id: string;
  eiq_score: number;
  dimension_scores: Record<string, number>;
  data_sources: string[];
  confidence: number;
}): Promise<{ added: boolean; snapshot: DailyEISnapshot }> {
  return mlFetch("/ei/growth/snapshot", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
}

export async function getEIGrowthTrend(userId: string): Promise<EIGrowthTrend> {
  return mlFetch<EIGrowthTrend>(`/ei/growth/trend/${encodeURIComponent(userId)}`);
}

export async function getEIMilestones(userId: string): Promise<{ milestones: EIMilestone[] }> {
  return mlFetch<{ milestones: EIMilestone[] }>(`/ei/growth/milestones/${encodeURIComponent(userId)}`);
}

export async function getEITrajectory(userId: string, target = 80): Promise<EITrajectory> {
  return mlFetch<EITrajectory>(`/ei/growth/trajectory/${encodeURIComponent(userId)}?target=${target}`);
}

export async function getEIDimensionReport(
  userId: string
): Promise<{ dimensions: Record<string, { label: string; current_score: number | null; slope_per_week: number; trend: string; changeability: string; d_value: number; description: string }> }> {
  return mlFetch(`/ei/growth/dimensions/${encodeURIComponent(userId)}`);
}

export async function getEIGrowthHistory(
  userId: string,
  lastN = 30
): Promise<{ count: number; snapshots: DailyEISnapshot[] }> {
  return mlFetch(`/ei/growth/history/${encodeURIComponent(userId)}?last_n=${lastN}`);
}

// ── Personal Model Personalization (#203) ────────────────────────────────────

export interface PersonalModelStatus {
  user_id: string;
  personal_model_active: boolean;
  total_sessions: number;
  total_labeled_epochs: number;
  buffer_size: number;
  head_accuracy_pct: number;
  baseline_ready: boolean;
  baseline_frames: number;
  class_counts: Record<string, number>;
  next_milestone: number;
  message: string;
}

export async function getPersonalStatus(
  userId: string
): Promise<PersonalModelStatus> {
  return mlFetch<PersonalModelStatus>(
    `/personal/status?user_id=${encodeURIComponent(userId)}`
  );
}

export async function triggerPersonalFineTune(
  userId: string
): Promise<{ status: string; val_accuracy_pct: number; buffer_size: number; personal_model_active: boolean; message: string }> {
  return mlFetch(`/personal/fine-tune`, {
    method: "POST",
    body: JSON.stringify({ user_id: userId }),
  });
}

// ── Dream narrative analysis (#287) ──────────────────────────────────────────

export interface DreamNarrativeAnalysis {
  emotional_valence: number;
  emotional_arousal: number;
  emotional_intensity: number;
  nightmare_score: number;
  is_nightmare: boolean;
  archetypes: string[];
  emotions_detected: { word: string; polarity: string }[];
  lucid_probability: number;
  word_count: number;
  irt_recommended: boolean;
  irt_protocol: {
    description: string;
    steps: string[];
    reference: string;
  } | null;
  insights: string[];
  morning_mood_prediction: string;
  theme_analysis: Record<string, unknown> | null;
}

export async function analyzeDreamNarrative(
  text: string,
  userId?: string
): Promise<{ status: string; analysis: DreamNarrativeAnalysis }> {
  const resolvedUserId = userId ?? getParticipantId();
  return mlFetch("/analyze-dream-narrative", {
    method: "POST",
    body: JSON.stringify({ text, user_id: resolvedUserId }),
  });
}

// ── Food Image Analysis (#351) ───────────────────────────────────────────────

export interface FoodItem {
  name: string;
  portion: string;
  calories: number;
  protein_g: number;
  carbs_g: number;
  fat_g: number;
  fiber_g: number;
}

export interface FoodImageAnalysisResult {
  food_items: FoodItem[];
  total_calories: number;
  total_protein_g: number;
  total_carbs_g: number;
  total_fat_g: number;
  total_fiber_g: number;
  dominant_macro: string;
  glycemic_impact: string;
  confidence: number;
  analysis_method: string;
  summary: string;
  error?: string;
}

export async function analyzeFoodImage(
  base64: string,
  textDescription?: string,
  mealType?: string
): Promise<FoodImageAnalysisResult> {
  return mlFetch<FoodImageAnalysisResult>("/food/analyze-image", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_base64: base64,
      text_description: textDescription,
      meal_type: mealType ?? "meal",
    }),
  });
}

// ── Adaptive music — ISO principle (#284) ────────────────────────────────────

export interface MusicTherapyPrescription {
  current_state: string;
  target_state: string;
  iso_phase: "match" | "transition" | "target";
  recommended_tempo_bpm: number;
  recommended_key: string;
  recommended_mode: string;
  search_query: string;
  session_duration_min: number;
  evidence_grade: string;
}

export async function getMusicTherapyPrescription(
  valence: number,
  arousal: number,
  targetValence?: number
): Promise<MusicTherapyPrescription> {
  return mlFetch("/music-therapy/prescribe", {
    method: "POST",
    body: JSON.stringify({
      valence,
      arousal,
      target_valence: targetValence ?? 0.5,
    }),
  });
}

// ─── Health Emotion Estimation ───────────────────────────────────────────────

export interface HealthEmotionRequest {
  hr_bpm: number;
  hrv_rmssd_ms?: number;
  respiratory_rate?: number;
  steps_last_hour?: number;
  sleep_hours?: number;
  timestamp?: string;
}

export interface HealthEmotionResult {
  emotion: string;
  valence: number;
  arousal: number;
  stress: number;
  confidence: number;
  source: "health";
  watch_says: string;
  explanation: string;
}

export async function estimateHealthEmotion(
  payload: HealthEmotionRequest
): Promise<HealthEmotionResult> {
  return mlFetch("/health-emotion/estimate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
