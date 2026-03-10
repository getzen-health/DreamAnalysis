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

/** Reads the ML backend URL from localStorage so the user can override it in Settings. */
export function getMLApiUrl(): string {
  if (typeof window !== "undefined" && window.location.hostname === "localhost") {
    try {
      const stored = localStorage.getItem("ml_backend_url");
      if (stored?.trim()) return stored.trim().replace(/\/$/, "");
    } catch { /* ignore */ }
    return "http://localhost:8080";
  }
  try {
    const stored = localStorage.getItem("ml_backend_url");
    if (stored?.trim()) return stored.trim().replace(/\/$/, "");
  } catch { /* SSR / private browsing fallback */ }
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

/** Extracts the FastAPI error detail from a non-ok response and throws. */
async function throwFromResponse(response: Response): Promise<never> {
  try {
    const body = await response.clone().json();
    const detail = body?.detail;
    if (typeof detail === "string") throw new Error(detail);
    if (Array.isArray(detail)) {
      throw new Error(detail.map((d: { msg?: string }) => d.msg).join("; "));
    }
  } catch (parseErr) {
    if (parseErr instanceof Error && parseErr.message !== "") throw parseErr;
  }
  throw new Error(`Request failed (${response.status})`);
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
  userId: string = "default"
): Promise<{ status: string; session_id: string }> {
  return mlFetch("/sessions/start", {
    method: "POST",
    body: JSON.stringify({ user_id: userId, session_type: sessionType }),
  });
}

export async function stopSession(): Promise<Record<string, unknown>> {
  return mlFetch("/sessions/stop", { method: "POST" });
}

export async function listSessions(
  userId?: string,
  sessionType?: string
): Promise<SessionSummary[]> {
  const params = new URLSearchParams();
  if (userId) params.set("user_id", userId);
  if (sessionType) params.set("session_type", sessionType);
  const qs = params.toString();
  return mlFetch<SessionSummary[]>(`/sessions${qs ? `?${qs}` : ""}`);
}

export async function getSession(sessionId: string): Promise<Record<string, unknown>> {
  return mlFetch(`/sessions/${sessionId}`);
}

export async function deleteSession(sessionId: string): Promise<{ status: string }> {
  return mlFetch(`/sessions/${sessionId}`, { method: "DELETE" });
}

export async function exportSession(sessionId: string, format: string = "csv"): Promise<string> {
  return mlFetchRaw(`/sessions/${sessionId}/export?format=${format}`);
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
  userId: string = "default",
  fs: number = 256
): Promise<BaselineFrameResult> {
  return mlFetch<BaselineFrameResult>("/calibration/baseline/add-frame", {
    method: "POST",
    body: JSON.stringify({ signals, fs, user_id: userId }),
  });
}

export async function getBaselineStatus(
  userId: string = "default"
): Promise<BaselineStatusResult> {
  return mlFetch<BaselineStatusResult>(
    `/calibration/baseline/status?user_id=${encodeURIComponent(userId)}`
  );
}

export async function resetBaselineCalibration(
  userId: string = "default"
): Promise<{ status: string; message: string }> {
  return mlFetch("/calibration/baseline/reset", {
    method: "POST",
    body: JSON.stringify({ user_id: userId }),
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
  signals: number[][],
  predictedLabel: string,
  correctLabel: string,
  userId: string = "default"
): Promise<{ updated: boolean }> {
  return mlFetch("/feedback", {
    method: "POST",
    body: JSON.stringify({
      user_id: userId,
      signals,
      predicted_label: predictedLabel,
      correct_label: correctLabel,
    }),
  });
}

export async function getCalibrationStatus(
  userId: string = "default"
): Promise<CalibrationStatus> {
  return mlFetch<CalibrationStatus>(`/calibration/status?user_id=${userId}`);
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
}

export interface VoiceWatchStatus {
  audio_model_loaded: boolean;
  librosa_available: boolean;
  ready: boolean;
  fusion_weights: { audio: number; watch: number };
}

export async function analyzeVoiceWatch(
  audioBase64: string,
  watch: WatchBiometrics = {}
): Promise<VoiceWatchEmotionResult> {
  return mlFetch<VoiceWatchEmotionResult>("/voice-watch/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      audio_b64: audioBase64,
      hr:   watch.hr,
      hrv:  watch.hrv,
      spo2: watch.spo2,
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

// ─── Voice Check-In ─────────────────────────────────────────────────────────

export interface CheckInResult {
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

export interface DailySummary {
  morning: CheckInResult | null;
  noon: CheckInResult | null;
  evening: CheckInResult | null;
  average_valence: number;
  average_arousal: number;
  dominant_emotion: string;
}

export async function submitVoiceCheckin(
  audioBase64: string,
  userId: string,
  checkinType: "morning" | "noon" | "evening"
): Promise<CheckInResult> {
  return mlFetch<CheckInResult>("/voice-checkin/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      audio_b64: audioBase64,
      user_id: userId,
      checkin_type: checkinType,
    }),
  });
}

export async function getCheckinHistory(userId: string, lastN = 30): Promise<{ checkins: CheckInResult[] }> {
  return mlFetch<{ checkins: CheckInResult[] }>(`/voice-checkin/history/${encodeURIComponent(userId)}?last_n=${lastN}`);
}

export async function getDailyCheckinSummary(userId: string): Promise<DailySummary> {
  return mlFetch<DailySummary>(`/voice-checkin/daily-summary/${encodeURIComponent(userId)}`);
}
