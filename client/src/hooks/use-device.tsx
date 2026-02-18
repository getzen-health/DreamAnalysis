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
  type DeviceInfo,
  type DeviceStatusResponse,
} from "@/lib/ml-api";

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
      emotion: string;
      confidence: number;
      valence: number;
      arousal: number;
      stress_index: number;
      focus_index: number;
      relaxation_index: number;
      probabilities?: Record<string, number>;
      band_powers?: Record<string, number>;
    };
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
  error: string | null;
  brainflowAvailable: boolean;
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

const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_BASE_MS = 1000;

function useDeviceInternal(): UseDeviceReturn {
  const [state, setState] = useState<DeviceState>("disconnected");
  const [devices, setDevices] = useState<DeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null);
  const [deviceStatus, setDeviceStatus] = useState<DeviceStatusResponse | null>(null);
  const [latestFrame, setLatestFrame] = useState<EEGStreamFrame | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [brainflowAvailable, setBrainflowAvailable] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isStreamingRef = useRef(false);
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
      reconnectRef.current = 0; // reset on success
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const frame: EEGStreamFrame = JSON.parse(event.data);
        const now = Date.now();

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
      setError("WebSocket connection error");
    };

    ws.onclose = () => {
      wsRef.current = null;
      // Auto-reconnect if still supposed to be streaming
      if (isStreamingRef.current && reconnectRef.current < MAX_RECONNECT_ATTEMPTS) {
        const delay = RECONNECT_BASE_MS * Math.pow(2, reconnectRef.current);
        reconnectRef.current += 1;
        reconnectTimerRef.current = setTimeout(() => {
          if (isStreamingRef.current) openWebSocket();
        }, delay);
      }
    };
  }, []);

  const refreshDevices = useCallback(async () => {
    try {
      const result = await listDevices();
      setDevices(result.devices);
      setBrainflowAvailable(result.brainflow_available);
      setError(null);
    } catch {
      setError("Failed to connect to ML service");
      setDevices([]);
    }
  }, []);

  const connect = useCallback(async (deviceType: string, params?: Record<string, string>) => {
    setError(null);
    setState("connecting");
    try {
      await connectDevice(deviceType, params);
      setSelectedDevice(deviceType);
      setState("connected");
      const status = await getDeviceStatus();
      setDeviceStatus(status);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Connection failed");
      setState("disconnected");
    }
  }, []);

  const disconnect = useCallback(async () => {
    isStreamingRef.current = false;
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    try {
      await disconnectDevice();
    } catch {
      // ignore
    }
    setState("disconnected");
    setSelectedDevice(null);
    setDeviceStatus(null);
    setLatestFrame(null);
  }, []);

  const startStream = useCallback(async () => {
    setError(null);
    try {
      await startDeviceStream();
      setState("streaming");
      isStreamingRef.current = true;
      reconnectRef.current = 0;
      openWebSocket();
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
    try {
      await stopDeviceStream();
    } catch {
      // ignore
    }
    setState("connected");
    setLatestFrame(null);
  }, []);

  // On mount: check if backend device is still connected/streaming (survives refresh)
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const status = await getDeviceStatus();
        if (cancelled) return;
        if (status.streaming) {
          setDeviceStatus(status);
          setSelectedDevice(status.device_type);
          setBrainflowAvailable(status.brainflow_available);
          setState("streaming");
          isStreamingRef.current = true;
          reconnectRef.current = 0;
          openWebSocket();
        } else if (status.connected) {
          setDeviceStatus(status);
          setSelectedDevice(status.device_type);
          setBrainflowAvailable(status.brainflow_available);
          setState("connected");
        }
      } catch {
        // ML service not available
      }
    })();
    return () => { cancelled = true; };
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

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isStreamingRef.current = false;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return {
    state,
    devices,
    selectedDevice,
    deviceStatus,
    latestFrame,
    error,
    brainflowAvailable,
    refreshDevices,
    connect,
    disconnect,
    startStream,
    stopStream,
  };
}

export type { DeviceState, EEGStreamFrame, UseDeviceReturn };
