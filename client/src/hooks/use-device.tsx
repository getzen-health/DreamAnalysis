import {
  useState,
  useCallback,
  useEffect,
  useRef,
  createContext,
  useContext,
  type ReactNode,
} from "react";
import {
  listDevices,
  connectDevice,
  disconnectDevice,
  getDeviceStatus,
  startDeviceStream,
  stopDeviceStream,
  getWebSocketUrl,
  startSession,
  stopSession,
  saveEEGEpoch,
  type DeviceInfo,
  type DeviceStatusResponse,
} from "@/lib/ml-api";
import { getParticipantId } from "@/lib/participant";
import { museBle, museFrameToEegStreamFrame } from "@/lib/muse-ble";
import {
  BLE_RECONNECT_MAX_ATTEMPTS,
  computeBleBackoffDelay,
  type BleReconnectState,
} from "@/lib/ble-reconnect";
import { saveEmotionHistory } from "@/lib/supabase-store";

type DeviceState = "disconnected" | "connecting" | "connected" | "streaming";

/* ── Full frame type matching the 12-model WebSocket output ────────── */

interface EEGStreamFrame {
  signals: number[][];
  analysis: {
    band_powers: Record<string, number>;
    features: Record<string, number>;
    sleep_staging?: {
      stage: string;
      stage_index: number;
      confidence: number;
      probabilities: Record<string, number>;
    };
    emotions?: {
      emotion: string | null;  // null while buffering the first 30s
      confidence: number;
      valence: number;
      arousal: number;
      stress_index: number;
      focus_index: number;
      relaxation_index: number;
      probabilities?: Record<string, number>;
      band_powers?: Record<string, number>;
      // 30-second window metadata
      ready?: boolean;         // true once first 30s window is complete
      buffered_sec?: number;   // seconds of EEG buffered so far
      window_sec?: number;     // target window length (30)
    };
    // Epoch-ready flag from /analyze-eeg REST endpoint (true when >= 4s buffered)
    epoch_ready?: boolean;
    dream_detection?: {
      is_dreaming: boolean;
      probability: number;
      rem_likelihood: number;
      dream_intensity: number;
      lucidity_estimate: number;
    };
    flow_state?: {
      in_flow: boolean;
      flow_score: number;
      confidence: number;
    };
    creativity?: {
      creativity_score: number;
      state: string;
      confidence: number;
    };
    memory_encoding?: {
      encoding_active: boolean;
      encoding_score: number;
      state: string;
      confidence: number;
    };
    drowsiness?: {
      state: string;
      drowsiness_index: number;
      confidence: number;
    };
    cognitive_load?: {
      level: string;
      load_index: number;
      confidence: number;
    };
    attention?: {
      state: string;
      attention_score: number;
      confidence: number;
    };
    stress?: {
      level: string;
      stress_index: number;
      confidence: number;
    };
    lucid_dream?: {
      state: string;
      lucidity_score: number;
      confidence: number;
    };
    meditation?: {
      depth: string;
      meditation_score: number;
      confidence: number;
    };
  };
  quality?: {
    sqi: number;
    artifacts_detected: string[];
    clean_ratio: number;
    channel_quality: number[];
  };
  // Simple amplitude-threshold signal quality for dashboard badge
  signal_quality_score?: number;   // 0-100
  artifact_detected?: boolean;
  artifact_type?: "clean" | "blink" | "muscle" | "electrode_pop";
  smoothed_states?: Record<string, unknown>;
  confidence_summary?: Record<string, unknown>;
  coherence?: Record<string, unknown>;
  emotion_shift?: {
    shift_detected: boolean;
    shift_type: string;
    description: string;
    body_feeling: string;
    guidance: string;
    reason: string;
    magnitude: number;
    confidence: number;
    previous_emotion: string;
    current_emotion: string;
    indicators: Record<string, number>;
    trends: Record<string, unknown>;
  };
  spiritual?: Record<string, unknown>;
  timestamp: number;
  n_channels: number;
  sample_rate: number;
}

/* ── Hook return type ─────────────────────────────────────────────── */

interface UseDeviceReturn {
  state: DeviceState;
  devices: DeviceInfo[];
  selectedDevice: string | null;
  deviceStatus: DeviceStatusResponse | null;
  latestFrame: EEGStreamFrame | null;
  /** True once >= 4 seconds of EEG has been buffered; emotion accuracy is degraded below this. */
  epochReady: boolean;
  error: string | null;
  brainflowAvailable: boolean;
  devicesLoaded: boolean;
  reconnectCount: number;
  /** BLE-specific reconnection state with exponential backoff (Issue #512) */
  bleReconnect: BleReconnectState;
  refreshDevices: () => Promise<void>;
  connect: (deviceType: string, params?: Record<string, string>) => Promise<void>;
  disconnect: () => Promise<void>;
  startStream: () => Promise<void>;
  stopStream: () => Promise<void>;
}

/* ── Context ──────────────────────────────────────────────────────── */

const DeviceContext = createContext<UseDeviceReturn | null>(null);

/* ── Provider (single instance for the whole app) ─────────────────── */

export function DeviceProvider({ children }: { children: ReactNode }) {
  const value = useDeviceInternal();
  return (
    <DeviceContext.Provider value={value}>{children}</DeviceContext.Provider>
  );
}

/* ── Consumer hook ────────────────────────────────────────────────── */

export function useDevice(): UseDeviceReturn {
  const ctx = useContext(DeviceContext);
  if (!ctx) {
    throw new Error("useDevice must be used inside <DeviceProvider>");
  }
  return ctx;
}

/* ── Internal implementation ──────────────────────────────────────── */

/* ── Client-side synthetic EEG frame generator ──────────────────────── */
// Used as fallback when ML backend is unreachable and user picks "Synthetic".
// Produces plausible-looking EEG data with band powers, emotions, etc.

function generateSyntheticFrame(): EEGStreamFrame {
  const t = Date.now() / 1000;
  const fs = 256;
  const nChannels = 4;
  const nSamples = 256; // 1 second of data

  // Generate 4 channels of synthetic signals (mix of sine waves at EEG band frequencies)
  const signals: number[][] = [];
  for (let ch = 0; ch < nChannels; ch++) {
    const channel: number[] = [];
    const phase = ch * 0.7; // offset per channel
    for (let i = 0; i < nSamples; i++) {
      const time = t + i / fs;
      // Mix of delta(2Hz), theta(6Hz), alpha(10Hz), beta(20Hz), gamma(40Hz) + noise
      const sample =
        15 * Math.sin(2 * Math.PI * 2 * time + phase) +    // delta
        8 * Math.sin(2 * Math.PI * 6 * time + phase) +     // theta
        12 * Math.sin(2 * Math.PI * 10 * time + phase) +   // alpha
        5 * Math.sin(2 * Math.PI * 20 * time + phase) +    // beta
        2 * Math.sin(2 * Math.PI * 40 * time + phase) +    // gamma
        3 * (Math.random() - 0.5);                          // noise
      channel.push(sample);
    }
    signals.push(channel);
  }

  // Varying band powers (oscillate slowly for visual interest)
  const slowOsc = (freq: number, offset: number) =>
    0.5 + 0.3 * Math.sin(2 * Math.PI * freq * t + offset);

  const delta = slowOsc(0.02, 0) * 0.25;
  const theta = slowOsc(0.03, 1) * 0.15;
  const alpha = slowOsc(0.015, 2) * 0.30;
  const beta = slowOsc(0.025, 3) * 0.20;
  const gamma = slowOsc(0.04, 4) * 0.10;
  const total = delta + theta + alpha + beta + gamma;

  const emotions = ["happy", "neutral", "calm", "focused", "relaxed"];
  const emotion = emotions[Math.floor(t / 10) % emotions.length];

  const stressVal = 0.3 + 0.2 * Math.sin(t * 0.05);
  const focusVal = 0.5 + 0.3 * Math.sin(t * 0.03 + 1);
  const relaxVal = 1 - stressVal;

  return {
    signals,
    analysis: {
      band_powers: {
        delta: delta / total,
        theta: theta / total,
        alpha: alpha / total,
        beta: beta / total,
        gamma: gamma / total,
      },
      features: {},
      sleep_staging: { stage: "Wake", stage_index: 0, confidence: 0.95, probabilities: { wake: 0.95, n1: 0.03, n2: 0.01, n3: 0.005, rem: 0.005 } },
      emotions: {
        emotion,
        confidence: 0.75,
        valence: 0.3 + 0.2 * Math.sin(t * 0.04),
        arousal: 0.4 + 0.2 * Math.sin(t * 0.06),
        stress_index: stressVal,
        focus_index: focusVal,
        relaxation_index: relaxVal,
        ready: true,
        buffered_sec: 30,
        window_sec: 30,
      },
      epoch_ready: true,
      flow_state: { in_flow: focusVal > 0.7, flow_score: focusVal * 0.8, confidence: 0.6 },
      creativity: { creativity_score: 0.4 + 0.1 * Math.sin(t * 0.02), state: "normal", confidence: 0.5 },
      drowsiness: { state: "alert", drowsiness_index: 0.1, confidence: 0.8 },
      cognitive_load: { level: "medium", load_index: 0.5, confidence: 0.6 },
      attention: { state: "focused", attention_score: focusVal, confidence: 0.65 },
      stress: { level: stressVal > 0.5 ? "moderate" : "relaxed", stress_index: stressVal, confidence: 0.7 },
      meditation: { depth: "light", meditation_score: relaxVal * 0.6, confidence: 0.5 },
      memory_encoding: { encoding_active: false, encoding_score: 0.3, state: "normal", confidence: 0.4 },
      dream_detection: { is_dreaming: false, probability: 0.05, rem_likelihood: 0.02, dream_intensity: 0.1, lucidity_estimate: 0.0 },
      lucid_dream: { state: "awake", lucidity_score: 0.0, confidence: 0.9 },
    },
    quality: { sqi: 85, artifacts_detected: [], clean_ratio: 0.95, channel_quality: [90, 88, 87, 91] },
    signal_quality_score: 85,
    artifact_detected: false,
    artifact_type: "clean",
    timestamp: t,
    n_channels: nChannels,
    sample_rate: fs,
  };
}

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_DELAY_MS = 8_000;   // cap at 8 s — 30 s was too long
const STALE_FRAME_TIMEOUT_MS = 15_000;  // reconnect WS if no frame arrives for 15 s
// On production (non-localhost) the ML backend is remote — Muse BLE can't reach it.
// Cap reconnects to 2 attempts then stop, to avoid infinite retry noise.
const IS_REMOTE_BACKEND = typeof window !== "undefined" && window.location.hostname !== "localhost";
const RECONNECT_MAX_ATTEMPTS = IS_REMOTE_BACKEND ? 2 : Infinity;

function useDeviceInternal(): UseDeviceReturn {
  const userIdRef = useRef(getParticipantId());
  const [state, setState] = useState<DeviceState>("disconnected");
  const [devices, setDevices] = useState<DeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null);
  const [deviceStatus, setDeviceStatus] = useState<DeviceStatusResponse | null>(null);
  const [latestFrame, setLatestFrame] = useState<EEGStreamFrame | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [brainflowAvailable, setBrainflowAvailable] = useState(false);
  const [devicesLoaded, setDevicesLoaded] = useState(false);
  const [reconnectCount, setReconnectCount] = useState(0);
  const [bleReconnect, setBleReconnect] = useState<BleReconnectState>({
    attempt: 0,
    isReconnecting: false,
    lastError: null,
    gaveUp: false,
  });
  const bleReconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const staleTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isStreamingRef = useRef(false);
  const intentionalDisconnectRef = useRef(false);   // true = user clicked Disconnect
  const lastFrameTimeRef = useRef(0);
  const pendingFrameRef = useRef<EEGStreamFrame | null>(null);
  const throttleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const FRAME_THROTTLE_MS = 1500; // update display every 1.5s

  /* WebSocket connection with auto-reconnect */
  const openWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    const ws = new WebSocket(getWebSocketUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      reconnectRef.current = 0;
      setReconnectCount(0);
      setError(null);
      ws.send(JSON.stringify({ command: "set_user", user_id: userIdRef.current }));
      // 30-second keepalive ping to prevent server-side idle timeout
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = setInterval(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: "ping" }));
        }
      }, 30000);
    };

    ws.onmessage = (event) => {
      try {
        const frame: EEGStreamFrame = JSON.parse(event.data);
        const now = Date.now();

        // Dispatch raw signals immediately (unthrottled) so EEGWaveformCanvas
        // can fill its circular buffer at full 4Hz without waiting for UI throttle.
        if (frame.signals) {
          window.dispatchEvent(
            new CustomEvent("eeg-signals", { detail: frame.signals })
          );
        }

        // Throttle UI updates to every FRAME_THROTTLE_MS so humans can read values
        if (now - lastFrameTimeRef.current >= FRAME_THROTTLE_MS) {
          lastFrameTimeRef.current = now;
          setLatestFrame(frame);
        } else {
          // Store pending frame, flush when throttle window expires
          pendingFrameRef.current = frame;
          if (!throttleTimerRef.current) {
            throttleTimerRef.current = setTimeout(() => {
              if (pendingFrameRef.current) {
                lastFrameTimeRef.current = Date.now();
                setLatestFrame(pendingFrameRef.current);
                pendingFrameRef.current = null;
              }
              throttleTimerRef.current = null;
            }, FRAME_THROTTLE_MS - (now - lastFrameTimeRef.current));
          }
        }
      } catch {
        // ignore parse errors
      }
    };

    ws.onerror = () => {
      if (pingIntervalRef.current) { clearInterval(pingIntervalRef.current); pingIntervalRef.current = null; }
      if (IS_REMOTE_BACKEND) {
        // Friendly message — user is on production without local hardware
        setError(null);
      } else {
        setError("WebSocket connection error — is the ML backend running? Try: cd ~/NeuralDreamWorkshop/ml && ./start.sh");
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
      if (pingIntervalRef.current) { clearInterval(pingIntervalRef.current); pingIntervalRef.current = null; }
      if (isStreamingRef.current && reconnectRef.current < RECONNECT_MAX_ATTEMPTS) {
        const delay = Math.min(RECONNECT_BASE_MS * Math.pow(2, reconnectRef.current), RECONNECT_MAX_DELAY_MS);
        reconnectRef.current += 1;
        setReconnectCount(reconnectRef.current);
        reconnectTimerRef.current = setTimeout(() => {
          if (isStreamingRef.current) openWebSocket();
        }, delay);
      } else if (reconnectRef.current >= RECONNECT_MAX_ATTEMPTS) {
        // Gave up on WebSocket — stay in current state, don't show scary error
        isStreamingRef.current = false;
        setState("disconnected");
        setError(null);
      }
    };
  }, []);

  // All BrainFlow-supported EEG devices — always shown so the user can pick any
  // headband they own. Muse devices connect via Web Bluetooth even without BrainFlow.
  const ALL_EEG_DEVICES: DeviceInfo[] = [
    // Muse
    { type: "muse_2", name: "Muse 2 EEG Headband", channels: 4, sample_rate: 256, available: true },
    { type: "muse_s", name: "Muse S EEG Headband", channels: 4, sample_rate: 256, available: true },
    // OpenBCI
    { type: "openbci_cyton", name: "OpenBCI Cyton", channels: 8, sample_rate: 250, available: true },
    { type: "openbci_ganglion", name: "OpenBCI Ganglion", channels: 4, sample_rate: 200, available: true },
    { type: "openbci_cyton_daisy", name: "OpenBCI Cyton+Daisy", channels: 16, sample_rate: 125, available: true },
    // Emotiv
    { type: "emotiv_epoc_x", name: "Emotiv EPOC X", channels: 14, sample_rate: 256, available: true },
    { type: "emotiv_insight", name: "Emotiv Insight", channels: 5, sample_rate: 128, available: true },
    { type: "emotiv_epoc_flex", name: "Emotiv EPOC Flex", channels: 32, sample_rate: 256, available: true },
    // NeuroSky
    { type: "neurosky_mindwave", name: "NeuroSky MindWave", channels: 1, sample_rate: 512, available: true },
    // BrainBit
    { type: "brainbit", name: "BrainBit", channels: 4, sample_rate: 250, available: true },
    // Crown by Neurosity
    { type: "neurosity_crown", name: "Crown (Neurosity)", channels: 8, sample_rate: 256, available: true },
    // ANT Neuro
    { type: "ant_neuro", name: "ANT Neuro eego", channels: 32, sample_rate: 2048, available: true },
    // G.tec
    { type: "gtec_unicorn", name: "g.tec Unicorn", channels: 8, sample_rate: 250, available: true },
    // Enobio
    { type: "enobio", name: "Enobio", channels: 8, sample_rate: 500, available: true },
    // Synthetic (demo) — only on desktop, not on native APK
    ...(!museBle.isNative ? [{ type: "synthetic", name: "Synthetic (Demo)", channels: 16, sample_rate: 256, available: true }] : []),
  ];

  const refreshDevices = useCallback(async () => {
    try {
      const result = await listDevices();
      // Merge backend-reported devices with our full static list.
      // Backend devices that overlap with ALL_EEG_DEVICES are already in the list.
      const knownTypes = new Set(ALL_EEG_DEVICES.map((d) => d.type));
      const extraBackendDevices = result.brainflow_available
        ? (result.devices ?? []).filter((d) => !knownTypes.has(d.type))
        : [];
      const merged = [...ALL_EEG_DEVICES, ...extraBackendDevices];
      setDevices(merged);
      setBrainflowAvailable(result.brainflow_available);
      setError(null);
    } catch {
      // Backend unreachable — show all devices so user can still pick one
      setError(IS_REMOTE_BACKEND ? null : "unreachable");
      setBrainflowAvailable(false);
      setDevices([...ALL_EEG_DEVICES]);
    } finally {
      setDevicesLoaded(true);
    }
  }, []);

  const connect = useCallback(async (deviceType: string, params?: Record<string, string>) => {
    setError(null);
    setState("connecting");

    // ── BLE path: native Capacitor (iOS/Android) OR Web Bluetooth (Chrome desktop) ──
    if (museBle.isAvailable && (deviceType === "muse_2" || deviceType === "muse_s")) {
      // Pre-flight diagnostic — skip on native (just try connecting directly)
      if (!museBle.isNative) {
        try {
          const diag = await museBle.checkBluetoothReady();
          if (diag.hints.length > 0) {
            setError(diag.hints.join(" "));
            setState("disconnected");
            return;
          }
        } catch { /* diagnostic failed — proceed anyway */ }
      }

      // Suppress BLE status errors during initial connection
      museBle.onStatus(() => {});
      try {
        let eegEmotionThrottle = 0;
        museBle.onEegFrame((frame) => {
          try {
            const eegFrame = museFrameToEegStreamFrame(frame);
            window.dispatchEvent(
              new CustomEvent("eeg-signals", { detail: eegFrame.signals })
            );
            setLatestFrame(eegFrame as EEGStreamFrame);

            // Propagate EEG emotion data to localStorage so Today page + all other pages update
            const now = Date.now();
            if (now - eegEmotionThrottle > 3000) { // throttle to every 3s
              eegEmotionThrottle = now;
              const emotions = eegFrame.analysis?.emotions as Record<string, unknown> | undefined;
              // Don't overwrite manual feelings (user logged a feeling recently)
              const manualUntil = parseInt(localStorage.getItem("ndw_manual_emotion_until") ?? "0", 10);
              if (emotions?.emotion && now > manualUntil) {
                try {
                  localStorage.setItem("ndw_last_emotion", JSON.stringify({
                    result: {
                      emotion: emotions.emotion,
                      valence: emotions.valence ?? 0,
                      arousal: emotions.arousal ?? 0.5,
                      stress_index: emotions.stress_index ?? 0.5,
                      focus_index: emotions.focus_index ?? 0.5,
                      confidence: emotions.confidence ?? 0.5,
                      model_type: "eeg",
                      timestamp: now / 1000,
                    },
                    timestamp: now,
                  }));
                  window.dispatchEvent(new CustomEvent("ndw-emotion-update"));
                  // Fire-and-forget: sync EEG emotion → localStorage + Supabase + Express DB
                  // valence is -1 to 1; mood is 0-1 — convert by shifting and scaling
                  const rawValence = Number(emotions.valence ?? 0);
                  saveEmotionHistory(userIdRef.current, {
                    stress: Number(emotions.stress_index ?? 0.5),
                    focus: Number(emotions.focus_index ?? 0.5),
                    mood: (rawValence + 1) / 2,
                    energy: Number(emotions.arousal ?? 0.5),
                    valence: rawValence,
                    arousal: Number(emotions.arousal ?? 0.5),
                    source: "eeg",
                    dominantEmotion: String(emotions.emotion),
                  }).catch(() => {});
                } catch { /* storage quota */ }
              }
            }
          } catch (frameErr) {
            console.error("EEG frame conversion failed:", frameErr);
            // Still set raw signals so waveform renders
            setLatestFrame({
              signals: frame.signals,
              analysis: { band_powers: frame.bandPowers || {}, features: {} },
              quality: { sqi: frame.signalQuality ?? 0, artifacts_detected: [], clean_ratio: 1, channel_quality: [] },
              timestamp: Date.now() / 1000,
              n_channels: frame.signals?.length ?? 4,
              sample_rate: 256,
            } as EEGStreamFrame);
            setError(`EEG processing: ${frameErr instanceof Error ? frameErr.message : String(frameErr)}`);
          }
        });
        await museBle.connect();
        // BLE succeeded — NOW register the status callback for disconnect handling
        // Issue #512: exponential backoff reconnection (1s, 2s, 4s, 8s, 16s, max 5 attempts)
        museBle.onStatus((status) => {
          if (status.state === "idle" || status.state === "error") {
            if (status.error) setError(status.error);
            if (!intentionalDisconnectRef.current) {
              setBleReconnect((prev) => {
                const attempt = prev.attempt;
                if (attempt >= BLE_RECONNECT_MAX_ATTEMPTS) {
                  // Exhausted all attempts — give up
                  isStreamingRef.current = false;
                  setState("disconnected");
                  setError("Muse connection lost after " + BLE_RECONNECT_MAX_ATTEMPTS + " attempts. Tap Connect to retry.");
                  return { attempt, isReconnecting: false, lastError: "Max attempts reached", gaveUp: true };
                }
                const delay = computeBleBackoffDelay(attempt);
                const nextState: BleReconnectState = {
                  attempt,
                  isReconnecting: true,
                  lastError: status.error ?? "Connection lost",
                  gaveUp: false,
                };
                // Schedule reconnection with exponential backoff
                if (bleReconnectTimerRef.current) clearTimeout(bleReconnectTimerRef.current);
                bleReconnectTimerRef.current = setTimeout(async () => {
                  if (intentionalDisconnectRef.current) return;
                  try {
                    await museBle.reconnect();
                    // Success — reset counter
                    setError(null);
                    setBleReconnect({ attempt: 0, isReconnecting: false, lastError: null, gaveUp: false });
                  } catch (e) {
                    const errMsg = e instanceof Error ? e.message : "Reconnection failed";
                    setBleReconnect((s) => {
                      const next = s.attempt + 1;
                      if (next >= BLE_RECONNECT_MAX_ATTEMPTS) {
                        isStreamingRef.current = false;
                        setState("disconnected");
                        setError("Muse connection lost after " + BLE_RECONNECT_MAX_ATTEMPTS + " attempts. Tap Connect to retry.");
                        return { attempt: next, isReconnecting: false, lastError: errMsg, gaveUp: true };
                      }
                      return { attempt: next, isReconnecting: true, lastError: errMsg, gaveUp: false };
                    });
                  }
                }, delay);
                return nextState;
              });
            } else {
              isStreamingRef.current = false;
              setState("disconnected");
              setBleReconnect({ attempt: 0, isReconnecting: false, lastError: null, gaveUp: false });
            }
          }
        });
        setSelectedDevice(deviceType);
        setDeviceStatus({
          connected: true,
          streaming: true,
          device_type: deviceType,
          n_channels: 4,
          sample_rate: 256,
          brainflow_available: false,
        });
        setState("streaming");
        isStreamingRef.current = true;
        startSession("general", userIdRef.current).catch(() => {});
        return; // BLE succeeded, done
      } catch (e) {
        const bleErr = e instanceof Error ? e.message : "BLE connection failed";
        console.error("BLE connection error:", bleErr, e);
        // BLE failed — only fall through to BrainFlow if running locally
        // (BrainFlow on Railway/cloud has no Bluetooth hardware)
        const isLocal = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
        if (!museBle.isNative && isLocal) {
          console.warn("Web Bluetooth GATT failed, falling back to BrainFlow:", e);
          museBle.onStatus(() => {}); // clear any stale callback
          setState("connecting");
          setError(null);
          // Fall through to BrainFlow path below
        } else {
          setError(bleErr || "No Muse headband detected. Make sure your Muse is turned on and Bluetooth is enabled.");
          setState("disconnected");
          return;
        }
      }
    }

    // ── Desktop path (BrainFlow via ML backend WebSocket) ──────────────────
    try {
      await connectDevice(deviceType, params);
      setSelectedDevice(deviceType);
      setState("connected");
      const status = await getDeviceStatus();
      setDeviceStatus(status);
      // Auto-start streaming immediately — no extra button click needed
      await startDeviceStream();
      setState("streaming");
      isStreamingRef.current = true;
      reconnectRef.current = 0;
      openWebSocket();
      startSession("general", userIdRef.current).catch(() => {});
    } catch (e) {
      // ── Synthetic fallback: generate client-side fake EEG when ML backend is unreachable ──
      if (deviceType === "synthetic") {
        setSelectedDevice("synthetic");
        setDeviceStatus({
          connected: true,
          streaming: true,
          device_type: "synthetic",
          n_channels: 4,
          sample_rate: 256,
          brainflow_available: false,
        });
        setState("streaming");
        isStreamingRef.current = true;
        setError(null);

        // Start a local interval that generates synthetic EEG frames
        const syntheticInterval = setInterval(() => {
          if (!isStreamingRef.current) {
            clearInterval(syntheticInterval);
            return;
          }
          const frame = generateSyntheticFrame();
          window.dispatchEvent(
            new CustomEvent("eeg-signals", { detail: frame.signals })
          );
          setLatestFrame(frame as EEGStreamFrame);
          // Compute brain age from synthetic band powers
          if (frame.analysis?.band_powers) {
            import("@/lib/brain-age").then(({ computeAndCacheBrainAge }) => {
              computeAndCacheBrainAge(frame.analysis.band_powers);
            }).catch(() => {});
          }
        }, 1500);

        // Store interval ID so disconnect can clear it
        (window as unknown as Record<string, unknown>).__ndw_synthetic_interval = syntheticInterval;
        return;
      }

      const msg = e instanceof Error ? e.message : "Connection failed";
      if (IS_REMOTE_BACKEND && deviceType !== "synthetic") {
        setError("Could not connect to device. Make sure your Muse is turned on and Bluetooth is enabled.");
      } else {
        setError(msg);
      }
      setState("disconnected");
    }
  }, [openWebSocket]);

  const disconnect = useCallback(async () => {
    intentionalDisconnectRef.current = true;
    isStreamingRef.current = false;
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (bleReconnectTimerRef.current) {
      clearTimeout(bleReconnectTimerRef.current);
      bleReconnectTimerRef.current = null;
    }
    setBleReconnect({ attempt: 0, isReconnecting: false, lastError: null, gaveUp: false });
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    // Clear client-side synthetic data interval if active
    const synInterval = (window as unknown as Record<string, unknown>).__ndw_synthetic_interval;
    if (synInterval) {
      clearInterval(synInterval as ReturnType<typeof setInterval>);
      (window as unknown as Record<string, unknown>).__ndw_synthetic_interval = undefined;
    }
    stopSession(userIdRef.current).catch(() => {}); // save recording if one was active

    // Disconnect BLE if active (mobile path)
    if (museBle.getStatus().state === "streaming" || museBle.getStatus().state === "connected") {
      await museBle.disconnect().catch(() => {});
    } else {
      try {
        await disconnectDevice();
      } catch {
        // ignore
      }
    }

    setState("disconnected");
    setSelectedDevice(null);
    setDeviceStatus(null);
    setLatestFrame(null);
    intentionalDisconnectRef.current = false; // reset for future connects
  }, []);

  const startStream = useCallback(async () => {
    setError(null);
    try {
      await startDeviceStream();
      setState("streaming");
      isStreamingRef.current = true;
      reconnectRef.current = 0;
      openWebSocket();
      startSession("general", userIdRef.current).catch(() => {}); // auto-record while streaming
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start stream");
    }
  }, [openWebSocket]);

  const stopStream = useCallback(async () => {
    isStreamingRef.current = false;
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    stopSession(userIdRef.current).catch(() => {}); // save the recording
    try {
      await stopDeviceStream();
    } catch {
      // ignore
    }
    setState("connected");
    setLatestFrame(null);
  }, []);

  // Auto-restart a new session every 30 minutes while streaming
  // so each saved chunk is manageable and "Today" chart updates regularly
  useEffect(() => {
    if (!isStreamingRef.current) return;
    const interval = setInterval(() => {
      if (isStreamingRef.current) {
        stopSession(userIdRef.current)
          .catch(() => {})
          .finally(() => {
            if (isStreamingRef.current) {
              startSession("general", userIdRef.current).catch(() => {});
            }
          });
      }
    }, 30 * 60 * 1000); // every 30 minutes
    return () => clearInterval(interval);
  }, [state]);

  // On mount: check if backend device is still connected/streaming (survives refresh)
  // and immediately set brainflowAvailable so the dialog doesn't flash "not installed"
  useEffect(() => {
    let cancelled = false;
    let retryTimeout: ReturnType<typeof setTimeout> | null = null;

    const applyStatus = (status: Awaited<ReturnType<typeof getDeviceStatus>>) => {
      // Always update brainflowAvailable — eliminates the false "not installed" banner
      setBrainflowAvailable(status.brainflow_available ?? false);
      setDevicesLoaded(true);
      if (status.streaming) {
        setDeviceStatus(status);
        setSelectedDevice(status.device_type);
        setState("streaming");
        isStreamingRef.current = true;
        reconnectRef.current = 0;
        openWebSocket();
        startSession("general", userIdRef.current).catch(() => {}); // resume auto-recording after page reload
      } else if (status.connected) {
        setDeviceStatus(status);
        setSelectedDevice(status.device_type);
        setState("connected");
      }
    };

    (async () => {
      try {
        const status = await getDeviceStatus();
        if (cancelled) return;
        applyStatus(status);
        // If first call returned disconnected (not streaming, not connected), retry once
        // after 3 seconds to handle race conditions on page reload
        if (!status.streaming && !status.connected) {
          retryTimeout = setTimeout(async () => {
            if (cancelled) return;
            try {
              const retryStatus = await getDeviceStatus();
              if (cancelled) return;
              applyStatus(retryStatus);
            } catch {
              // Stay disconnected — user needs to connect their Muse headband
              if (IS_REMOTE_BACKEND) setDevicesLoaded(true);
            }
          }, 3000);
        }
      } catch {
        // ML service not available on first attempt — retry after 3 seconds
        if (IS_REMOTE_BACKEND) setDevicesLoaded(true);
        retryTimeout = setTimeout(async () => {
          if (cancelled) return;
          try {
            const retryStatus = await getDeviceStatus();
            if (cancelled) return;
            applyStatus(retryStatus);
          } catch {
            // ML service still not available — user needs local setup for EEG
            setDevicesLoaded(true);
          }
        }, 3000);
      }
    })();
    return () => {
      cancelled = true;
      if (retryTimeout) clearTimeout(retryTimeout);
    };
  }, [openWebSocket]);

  // Poll device status when connected
  useEffect(() => {
    if (state !== "connected" && state !== "streaming") return;
    const interval = setInterval(async () => {
      try {
        const status = await getDeviceStatus();
        setDeviceStatus(status);
      } catch {
        // ignore polling errors
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [state]);

  // Reconnect WebSocket immediately when the tab/app becomes visible again.
  // Without this, a WebSocket that died in the background stays dead until the
  // next reconnect timer fires (up to 8 s) or the user manually reconnects.
  useEffect(() => {
    const onVisible = () => {
      if (document.hidden || !isStreamingRef.current || museBle.isAvailable) return;
      reconnectRef.current = 0; // reset backoff so next attempt is instant
      const ws = wsRef.current;
      if (!ws || ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING) {
        openWebSocket();
      }
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => document.removeEventListener("visibilitychange", onVisible);
  }, [openWebSocket]);

  // Stale-frames watchdog: if the WebSocket is "open" but no frame has arrived
  // for STALE_FRAME_TIMEOUT_MS, reconnect — handles silent-death scenarios.
  useEffect(() => {
    if (state !== "streaming" || museBle.isAvailable) {
      if (staleTimerRef.current) { clearInterval(staleTimerRef.current); staleTimerRef.current = null; }
      return;
    }
    staleTimerRef.current = setInterval(() => {
      if (!isStreamingRef.current) return;
      const age = Date.now() - lastFrameTimeRef.current;
      if (age > STALE_FRAME_TIMEOUT_MS) {
        reconnectRef.current = 0;
        openWebSocket();
      }
    }, 5_000);
    return () => { if (staleTimerRef.current) { clearInterval(staleTimerRef.current); staleTimerRef.current = null; } };
  }, [state, openWebSocket]);

  // ── Auto-save EEG epochs for training pipeline ──────────────────────
  // Every SAVE_INTERVAL_MS, send the latest frame's signals + analysis
  // to POST /sessions/save-eeg so user data accumulates for model training.
  const SAVE_INTERVAL_MS = 4_000; // 4 seconds = ~1024 samples at 256 Hz
  const lastSaveRef = useRef(0);

  useEffect(() => {
    if (state !== "streaming" || !latestFrame?.signals) return;

    const now = Date.now();
    if (now - lastSaveRef.current < SAVE_INTERVAL_MS) return;
    lastSaveRef.current = now;

    const emotions = latestFrame.analysis?.emotions;
    saveEEGEpoch({
      signals: latestFrame.signals,
      device_type: selectedDevice || "muse_2",
      predicted_emotion: emotions?.emotion ?? undefined,
      band_powers: emotions?.band_powers ?? latestFrame.analysis?.band_powers ?? undefined,
      frontal_asymmetry: (latestFrame.analysis as Record<string, unknown>)?.frontal_asymmetry as number | undefined,
      valence: emotions?.valence,
      arousal: emotions?.arousal,
    }).catch(() => {
      // Non-critical — don't disrupt streaming if save fails
    });
  }, [state, latestFrame, selectedDevice]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isStreamingRef.current = false;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (bleReconnectTimerRef.current) clearTimeout(bleReconnectTimerRef.current);
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
      if (staleTimerRef.current) clearInterval(staleTimerRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // epochReady: true once the 4-second buffer has filled (from epoch_ready in WS analysis,
  // or from analysis.emotions.ready which tracks the longer 30s window).
  // Defaults to false while buffering; once true it stays true for the session.
  const epochReady =
    (latestFrame?.analysis as Record<string, unknown> | undefined)?.epoch_ready === true ||
    (latestFrame?.analysis?.emotions as Record<string, unknown> | undefined)?.ready === true ||
    // BLE path: if we have band_powers with real values, data is flowing
    (state === "streaming" && latestFrame?.analysis?.band_powers != null &&
      Object.values(latestFrame.analysis.band_powers as Record<string, number>).some(v => v > 0));

  return {
    state,
    devices,
    selectedDevice,
    deviceStatus,
    latestFrame,
    epochReady,
    error,
    brainflowAvailable,
    devicesLoaded,
    reconnectCount,
    bleReconnect,
    refreshDevices,
    connect,
    disconnect,
    startStream,
    stopStream,
  };
}

export type { DeviceState, EEGStreamFrame, UseDeviceReturn };
