import { AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Capacitor } from "@capacitor/core";
import { useMLConnection } from "@/hooks/use-ml-connection";

export function SimulationModeBanner() {
  const { status, reconnect } = useMLConnection();
  // On native (iOS/Android), BLE handles EEG directly — no ML backend needed
  if (Capacitor.isNativePlatform()) return null;
  if (status !== "error") return null;
  return (
    <div className="w-full flex items-center gap-3 px-4 py-2 bg-amber-500/10 border border-amber-500/30 rounded-lg mb-4 text-sm">
      <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0" />
      <span className="text-amber-200 flex-1">
        ML backend unreachable — running in simulation mode
      </span>
      <Button variant="ghost" size="sm" className="text-amber-400 hover:text-amber-300 h-7 px-2" onClick={reconnect}>
        <RefreshCw className="w-3 h-3 mr-1" />
        Reconnect
      </Button>
    </div>
  );
}
