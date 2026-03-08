import { Eye, Zap, Radio } from "lucide-react";

interface SignalQualityBadgeProps {
  sqi?: number;
  artifacts?: string[];
  compact?: boolean;
  // Simple amplitude-threshold fields from /analyze-eeg
  artifactDetected?: boolean;
  artifactType?: "clean" | "blink" | "muscle" | "electrode_pop";
}

const ARTIFACT_LABELS: Record<string, string> = {
  blink:         "Eye blink detected",
  muscle:        "Muscle artifact — relax jaw",
  electrode_pop: "Electrode contact lost",
  clean:         "No artifact",
};

export function SignalQualityBadge({
  sqi: rawSqi,
  artifacts = [],
  compact = false,
  artifactDetected = false,
  artifactType = "clean",
}: SignalQualityBadgeProps) {
  const sqi = rawSqi ?? 0;
  const color =
    sqi >= 80
      ? "text-success border-success/30 bg-success/10"
      : sqi >= 60
        ? "text-warning border-warning/30 bg-warning/10"
        : "text-destructive border-destructive/30 bg-destructive/10";

  const label = sqi >= 80 ? "Good" : sqi >= 60 ? "Fair" : "Poor";

  const tooltipText = artifactDetected
    ? ARTIFACT_LABELS[artifactType] ?? artifactType
    : undefined;

  if (compact) {
    return (
      <span
        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-xs font-mono ${color}`}
        title={tooltipText ?? `Signal Quality: ${sqi.toFixed(0)}%`}
      >
        <Radio className="h-3 w-3" />
        {sqi.toFixed(0)}
        {artifactDetected && (
          <span className="ml-0.5 opacity-70">!</span>
        )}
      </span>
    );
  }

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border ${color}`} title={tooltipText}>
      <div className="flex items-center gap-1">
        <Radio className="h-4 w-4" />
        <span className="text-sm font-mono font-medium">SQI: {sqi.toFixed(0)}</span>
      </div>
      <span className="text-xs opacity-70">{label}</span>
      {(artifacts.length > 0 || artifactDetected) && (
        <div className="flex items-center gap-1 ml-1">
          {(artifacts.includes("eye_blink") || artifactType === "blink") && (
            <Eye className="h-3 w-3 opacity-70" aria-label="Eye blink detected" />
          )}
          {(artifacts.includes("muscle") || artifactType === "muscle") && (
            <Zap className="h-3 w-3 opacity-70" aria-label="Muscle artifact detected" />
          )}
          {(artifacts.includes("electrode_pop") || artifactType === "electrode_pop") && (
            <Radio className="h-3 w-3 opacity-70" aria-label="Electrode pop detected" />
          )}
        </div>
      )}
    </div>
  );
}
