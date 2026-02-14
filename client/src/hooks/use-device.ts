import { useState, useCallback, useEffect, useRef } from "react";
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

interface EEGStreamFrame {
  signals: number[][];
  analysis: {
    band_powers: Record<string, number>;
    features: Record<string, number>;
  };
  timestamp: number;
  n_channels: number;
  sample_rate: number;
}

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

export function useDevice(): UseDeviceReturn {
  const [state, setState] = useState<DeviceState>("disconnected");
  const [devices, setDevices] = useState<DeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null);
  const [deviceStatus, setDeviceStatus] = useState<DeviceStatusResponse | null>(null);
  const [latestFrame, setLatestFrame] = useState<EEGStreamFrame | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [brainflowAvailable, setBrainflowAvailable] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const refreshDevices = useCallback(async () => {
    try {
      const result = await listDevices();
      setDevices(result.devices);
      setBrainflowAvailable(result.brainflow_available);
      setError(null);
    } catch (e) {
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
    // Close WebSocket if open
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

      // Open WebSocket for live data
      const ws = new WebSocket(getWebSocketUrl());
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const frame: EEGStreamFrame = JSON.parse(event.data);
          setLatestFrame(frame);
        } catch {
          // ignore parse errors
        }
      };

      ws.onerror = () => {
        setError("WebSocket connection error");
      };

      ws.onclose = () => {
        wsRef.current = null;
      };
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start stream");
    }
  }, []);

  const stopStream = useCallback(async () => {
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
      if (wsRef.current) {
        wsRef.current.close();
      }
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
