import { useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Wifi, WifiOff, Radio, Activity } from "lucide-react";
import { useDevice, type DeviceState } from "@/hooks/use-device";

interface DeviceConnectionProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  device: ReturnType<typeof useDevice>;
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
  const {
    state,
    devices,
    selectedDevice,
    deviceStatus,
    error,
    brainflowAvailable,
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
          <DialogTitle className="font-futuristic flex items-center gap-2">
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

          {!brainflowAvailable && (
            <div className="p-3 rounded-lg bg-warning/10 border border-warning/30 text-sm text-warning">
              BrainFlow not installed on ML server. Only simulation mode available.
            </div>
          )}

          {error && (
            <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/30 text-sm text-destructive">
              {error}
            </div>
          )}

          {/* Device Info (when connected) */}
          {deviceStatus && state !== "disconnected" && (
            <Card className="p-4 bg-card/50 border-primary/20">
              <div className="grid grid-cols-2 gap-2 text-sm">
                <span className="text-foreground/60">Device:</span>
                <span className="font-mono">{deviceStatus.device_type || "Unknown"}</span>
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
