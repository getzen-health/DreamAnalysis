import { Eye, Zap, Radio } from "lucide-react";

interface SignalQualityBadgeProps {
  sqi: number;
  artifacts?: string[];
  compact?: boolean;
}

export function SignalQualityBadge({ sqi, artifacts = [], compact = false }: SignalQualityBadgeProps) {
  const color =
    sqi >= 80
      ? "text-success border-success/30 bg-success/10"
      : sqi >= 60
        ? "text-warning border-warning/30 bg-warning/10"
        : "text-destructive border-destructive/30 bg-destructive/10";

  const label = sqi >= 80 ? "Good" : sqi >= 60 ? "Fair" : "Poor";

  if (compact) {
    return (
      <span
        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-xs font-mono ${color}`}
        title={`Signal Quality: ${sqi.toFixed(0)}%`}
      >
        <Radio className="h-3 w-3" />
        {sqi.toFixed(0)}
      </span>
    );
  }

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border ${color}`}>
      <div className="flex items-center gap-1">
        <Radio className="h-4 w-4" />
        <span className="text-sm font-mono font-medium">SQI: {sqi.toFixed(0)}</span>
      </div>
      <span className="text-xs opacity-70">{label}</span>
      {artifacts.length > 0 && (
        <div className="flex items-center gap-1 ml-1">
          {artifacts.includes("eye_blink") && (
            <Eye className="h-3 w-3 opacity-70" title="Eye blink detected" />
          )}
          {artifacts.includes("muscle") && (
            <Zap className="h-3 w-3 opacity-70" title="Muscle artifact detected" />
          )}
          {artifacts.includes("electrode_pop") && (
            <Radio className="h-3 w-3 opacity-70" title="Electrode pop detected" />
          )}
        </div>
      )}
    </div>
  );
}
