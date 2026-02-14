import { useState } from "react";
import { AlertTriangle, X, ShieldAlert, Eye } from "lucide-react";

export type AlertLevel = "normal" | "watch" | "warning" | "critical";

interface AlertBannerProps {
  level: AlertLevel;
  anomalyScore?: number;
  seizureProbability?: number;
  spikesDetected?: number;
  onDismiss?: () => void;
}

export function AlertBanner({
  level,
  anomalyScore,
  seizureProbability,
  spikesDetected,
  onDismiss,
}: AlertBannerProps) {
  const [dismissed, setDismissed] = useState(false);

  if (level === "normal" || dismissed) return null;

  const config = {
    watch: {
      bg: "bg-yellow-500/10 border-yellow-500/30",
      text: "text-yellow-400",
      icon: Eye,
      label: "Watch",
      pulse: false,
    },
    warning: {
      bg: "bg-orange-500/10 border-orange-500/30",
      text: "text-orange-400",
      icon: AlertTriangle,
      label: "Warning",
      pulse: true,
    },
    critical: {
      bg: "bg-red-500/15 border-red-500/40",
      text: "text-red-400",
      icon: ShieldAlert,
      label: "Critical",
      pulse: true,
    },
  }[level];

  const Icon = config.icon;

  const handleDismiss = () => {
    setDismissed(true);
    onDismiss?.();
  };

  return (
    <div
      className={`flex items-center justify-between px-4 py-2 rounded-lg border ${config.bg} ${config.pulse ? "animate-pulse" : ""}`}
      role="alert"
    >
      <div className="flex items-center gap-3">
        <Icon className={`h-5 w-5 ${config.text}`} />
        <div>
          <span className={`text-sm font-medium ${config.text}`}>
            {config.label}
          </span>
          <span className="text-xs text-foreground/60 ml-2">
            {spikesDetected !== undefined && spikesDetected > 0 && (
              <span>{spikesDetected} spike{spikesDetected !== 1 ? "s" : ""} detected. </span>
            )}
            {seizureProbability !== undefined && seizureProbability > 0.2 && (
              <span>Seizure probability: {(seizureProbability * 100).toFixed(0)}%. </span>
            )}
            {anomalyScore !== undefined && anomalyScore < -0.3 && (
              <span>Anomaly score: {anomalyScore.toFixed(2)}. </span>
            )}
          </span>
        </div>
      </div>
      <button
        onClick={handleDismiss}
        className="p-1 rounded hover:bg-foreground/10 transition-colors"
        aria-label="Dismiss alert"
      >
        <X className="h-4 w-4 text-foreground/50" />
      </button>
    </div>
  );
}
