const ML_API_URL_DEFAULT = import.meta.env.VITE_ML_API_URL || "http://localhost:8000";

/** Reads the ML backend URL from localStorage so the user can override it in Settings. */
function getMLApiUrl(): string {
  try {
    const stored = localStorage.getItem("ml_backend_url");
    if (stored?.trim()) return stored.trim().replace(/\/$/, "");
  } catch { /* SSR / private browsing fallback */ }
  return ML_API_URL_DEFAULT;
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
  calibrated: boolean;
  n_samples: number;
  personal_accuracy: number;
  classes: string[];
}

async function mlFetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${getMLApiUrl()}/api${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!response.ok) {
    throw new Error(`ML API error: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function mlFetchRaw(endpoint: string, options?: RequestInit): Promise<string> {
  const response = await fetch(`${getMLApiUrl()}/api${endpoint}`, options);
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
  return `${wsProtocol}://${host}/ws/eeg-stream`;
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
