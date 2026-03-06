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
  type DeviceInfo,
  type DeviceStatusResponse,
} from "@/lib/ml-api";
import { museBle, museFrameToEegStreamFrame } from "@/lib/muse-ble";

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
  devicesLoaded: boolean;
  reconnectCount: number;
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

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_DELAY_MS = 8_000;   // cap at 8 s — 30 s was too long
const STALE_FRAME_TIMEOUT_MS = 15_000;  // reconnect WS if no frame arrives for 15 s
// On production (non-localhost) the ML backend is remote — Muse BLE can't reach it.
// Cap reconnects to 2 attempts then stop, to avoid infinite retry noise.
const IS_REMOTE_BACKEND = typeof window !== "undefined" && window.location.hostname !== "localhost";
const RECONNECT_MAX_ATTEMPTS = IS_REMOTE_BACKEND ? 2 : Infinity;

function useDeviceInternal(): UseDeviceReturn {
  const [state, setState] = useState<DeviceState>("disconnected");
  const [devices, setDevices] = useState<DeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null);
  const [deviceStatus, setDeviceStatus] = useState<DeviceStatusResponse | null>(null);
  const [latestFrame, setLatestFrame] = useState<EEGStreamFrame | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [brainflowAvailable, setBrainflowAvailable] = useState(false);
  const [devicesLoaded, setDevicesLoaded] = useState(false);
  const [reconnectCount, setReconnectCount] = useState(0);
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
        setError("Live EEG streaming requires a local ML backend. Simulation mode is active — EEG data is simulated.");
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
        // Gave up — switch to simulation automatically on remote backend
        isStreamingRef.current = false;
        setState("disconnected");
      }
    };
  }, []);

  // Minimum static device list — always shown when backend returns empty or is unreachable
  const STATIC_DEVICES: DeviceInfo[] = [
    { type: "muse_2",    name: "Muse 2",           channels: 4,  sample_rate: 256, available: true },
    { type: "muse_s",    name: "Muse S",            channels: 4,  sample_rate: 256, available: true },
    { type: "synthetic", name: "Synthetic (demo)",  channels: 16, sample_rate: 256, available: true },
  ];

  const refreshDevices = useCallback(async () => {
    try {
      const result = await listDevices();
      // Never leave the list empty — merge backend list with static fallbacks
      const backendDevices = result.devices ?? [];
      const merged = backendDevices.length > 0
        ? backendDevices
        : STATIC_DEVICES;
      setDevices(merged);
      setBrainflowAvailable(result.brainflow_available);
      setError(null);
    } catch {
      // Backend unreachable — always show static list so user can still connect
      setError("unreachable");
      setBrainflowAvailable(false);
      setDevices(STATIC_DEVICES);
    } finally {
      setDevicesLoaded(true);
    }
  }, []);

  const connect = useCallback(async (deviceType: string, params?: Record<string, string>) => {
    setError(null);
    setState("connecting");

    // ── BLE path: native Capacitor (iOS/Android) OR Web Bluetooth (Chrome desktop) ──
    if (museBle.isAvailable && (deviceType === "muse_2" || deviceType === "muse_s")) {
      try {
        museBle.onEegFrame((frame) => {
          const eegFrame = museFrameToEegStreamFrame(frame);
          // Dispatch raw signals for waveform canvas
          window.dispatchEvent(
            new CustomEvent("eeg-signals", { detail: eegFrame.signals })
          );
          setLatestFrame(eegFrame as EEGStreamFrame);
        });
        museBle.onStatus((status) => {
          if (status.state === "idle" || status.state === "error") {
            if (status.error) setError(status.error);
            if (!intentionalDisconnectRef.current) {
              // Unexpected BLE drop — silently reconnect using saved deviceId (no picker)
              reconnectTimerRef.current = setTimeout(async () => {
                if (intentionalDisconnectRef.current) return;
                try {
                  await museBle.reconnect();
                  setError(null);
                } catch {
                  // Reconnect failed (Muse may be off/out of range)
                  isStreamingRef.current = false;
                  setState("disconnected");
                  setError("Muse connection lost. Tap Connect to reconnect.");
                }
              }, 2000);
            } else {
              isStreamingRef.current = false;
              setState("disconnected");
            }
          }
        });
        await museBle.connect();
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
        startSession("general").catch(() => {});
      } catch (e) {
        // BLE failed — on desktop, fall through to BrainFlow path instead of giving up
        if (!museBle.isNative) {
          console.warn("Web Bluetooth GATT failed, falling back to BrainFlow:", e);
          setState("connecting");
          setError(null);
          // Fall through to BrainFlow path below
        } else {
          setError(e instanceof Error ? e.message : "BLE connection failed");
          setState("disconnected");
          return;
        }
      }
      if (museBle.isNative) return; // native BLE succeeded above
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
      startSession("general").catch(() => {});
    } catch (e) {
      setError(e instanceof Error ? e.message : "Connection failed");
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
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    stopSession().catch(() => {}); // save recording if one was active

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
      startSession("general").catch(() => {}); // auto-record while streaming
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
    stopSession().catch(() => {}); // save the recording
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
        stopSession()
          .catch(() => {})
          .finally(() => {
            if (isStreamingRef.current) {
              startSession("general").catch(() => {});
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
        startSession("general").catch(() => {}); // resume auto-recording after page reload
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
              // Stay disconnected
            }
          }, 3000);
        }
      } catch {
        // ML service not available on first attempt — retry after 3 seconds
        retryTimeout = setTimeout(async () => {
          if (cancelled) return;
          try {
            const retryStatus = await getDeviceStatus();
            if (cancelled) return;
            applyStatus(retryStatus);
          } catch {
            // ML service still not available — refreshDevices will set proper state when dialog opens
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

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isStreamingRef.current = false;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
      if (staleTimerRef.current) clearInterval(staleTimerRef.current);
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
    devicesLoaded,
    reconnectCount,
    refreshDevices,
    connect,
    disconnect,
    startStream,
    stopStream,
  };
}

export type { DeviceState, EEGStreamFrame, UseDeviceReturn };
