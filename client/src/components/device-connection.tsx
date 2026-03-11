import { useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Wifi, WifiOff, Radio, Activity, Terminal, Wand2 } from "lucide-react";
import { type DeviceState, type UseDeviceReturn } from "@/hooks/use-device";
import { Link, useLocation } from "wouter";

interface DeviceConnectionProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  device: UseDeviceReturn;
}

function StatusBadge({ state }: { state: DeviceState }) {
  const config: Record<DeviceState, { label: string; color: string }> = {
    disconnected: { label: "Disconnected", color: "text-foreground/50" },
    connecting: { label: "Connecting...", color: "text-warning" },
    connected: { label: "Connected", color: "text-success" },
    streaming: { label: "Streaming", color: "text-primary" },
  };

  const { label, color } = config[state];

  return (
    <span className={`text-sm font-mono ${color}`}>
      {state === "streaming" && (
        <Activity className="inline h-3 w-3 mr-1 animate-pulse" />
      )}
      {label}
    </span>
  );
}

export function DeviceConnection({ open, onOpenChange, device }: DeviceConnectionProps) {
  const [, navigate] = useLocation();
  const {
    state,
    devices,
    selectedDevice,
    deviceStatus,
    error,
    refreshDevices,
    connect,
    disconnect,
    startStream,
    stopStream,
  } = device;

  useEffect(() => {
    if (open) {
      refreshDevices();
    }
  }, [open, refreshDevices]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="glass-card border-primary/20 max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Radio className="h-5 w-5 text-primary" />
            EEG Device Manager
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          {/* Status */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-foreground/70">Status:</span>
            <StatusBadge state={state} />
          </div>

          {/* ── Backend unreachable ── */}
          {(error === "unreachable" || error === "Failed to fetch") && (
            <div className="p-3 rounded-lg bg-warning/10 border border-warning/30 text-sm space-y-2">
              <p className="font-medium text-warning flex items-center gap-1.5">
                <Terminal className="h-3.5 w-3.5 shrink-0" />
                ML backend not reachable
              </p>
              <p className="text-xs text-muted-foreground">
                Run this one command in your terminal — it starts everything and auto-fills Settings:
              </p>
              <code className="block text-[11px] bg-black/30 text-green-400 px-3 py-2 rounded font-mono">
                cd ~/NeuralDreamWorkshop/ml &amp;&amp; ./start.sh
              </code>
              <p className="text-xs text-muted-foreground">
                The script starts uvicorn + ngrok and opens{" "}
                <Link href="/settings" onClick={() => onOpenChange(false)} className="underline text-primary">
                  Settings
                </Link>{" "}
                with the URL already filled in. Then come back here and connect.
              </p>
            </div>
          )}

          {/* ── BrainFlow not installed (only shown if user has non-Muse devices) ── */}
          {/* Muse connects via Web Bluetooth directly — BrainFlow is irrelevant for it */}

          {/* ── Other errors ── */}
          {error && error !== "unreachable" && error !== "Failed to fetch" && (
            <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/30 text-sm text-destructive">
              {error}
            </div>
          )}

          {/* Device Info (when connected) */}
          {deviceStatus && state !== "disconnected" && (
            <Card className="p-4 bg-card/50 border-primary/20">
              <div className="grid grid-cols-2 gap-2 text-sm">
                <span className="text-foreground/60">Device:</span>
                <span className="font-mono">{deviceStatus.device_type || selectedDevice || "Unknown"}</span>
                <span className="text-foreground/60">Channels:</span>
                <span className="font-mono">{deviceStatus.n_channels}</span>
                <span className="text-foreground/60">Sample Rate:</span>
                <span className="font-mono">{deviceStatus.sample_rate} Hz</span>
              </div>
            </Card>
          )}

          {/* Device List (when disconnected) */}
          {state === "disconnected" && (
            <div className="space-y-2">
              <p className="text-sm text-foreground/70">Available Devices:</p>
              {devices.length === 0 ? (
                <p className="text-sm text-foreground/50 italic">
                  No devices found. Check ML service connection.
                </p>
              ) : (
                devices.map((dev) => (
                  <Card
                    key={dev.type}
                    className="p-3 bg-card/50 border-primary/10 hover:border-primary/30 cursor-pointer transition-colors"
                    onClick={() => connect(dev.type)}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="font-medium text-sm">{dev.name}</p>
                        <p className="text-xs text-foreground/50">
                          {dev.channels} channels | {dev.sample_rate} Hz
                        </p>
                      </div>
                      <Wifi className="h-4 w-4 text-primary/50" />
                    </div>
                  </Card>
                ))
              )}
            </div>
          )}

          {/* Guided setup wizard link (disconnected only) */}
          {state === "disconnected" && (
            <button
              onClick={() => { onOpenChange(false); navigate("/device-setup"); }}
              className="w-full flex items-center justify-center gap-1.5 text-xs text-muted-foreground hover:text-primary transition-colors py-1"
            >
              <Wand2 className="h-3.5 w-3.5" />
              First time? Use the guided setup wizard →
            </button>
          )}

          {/* Actions */}
          <div className="flex gap-2 pt-2">
            {state === "disconnected" && (
              <Button
                variant="outline"
                className="flex-1"
                onClick={refreshDevices}
              >
                Refresh Devices
              </Button>
            )}

            {state === "connected" && (
              <>
                <Button
                  className="flex-1 bg-primary/20 border-primary/30 text-primary hover:bg-primary/30"
                  onClick={startStream}
                >
                  <Activity className="mr-2 h-4 w-4" />
                  Start Streaming
                </Button>
                <Button
                  variant="outline"
                  onClick={disconnect}
                >
                  <WifiOff className="mr-2 h-4 w-4" />
                  Disconnect
                </Button>
              </>
            )}

            {state === "streaming" && (
              <>
                <Button
                  variant="outline"
                  className="flex-1 border-warning/30 text-warning hover:bg-warning/10"
                  onClick={stopStream}
                >
                  Stop Streaming
                </Button>
                <Button
                  variant="outline"
                  onClick={disconnect}
                >
                  <WifiOff className="mr-2 h-4 w-4" />
                  Disconnect
                </Button>
              </>
            )}

            {state === "connecting" && (
              <Button variant="outline" className="flex-1" disabled>
                Connecting...
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
