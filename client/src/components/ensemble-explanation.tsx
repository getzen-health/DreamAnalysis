/**
 * EnsembleExplanation -- shows users how the EEG emotion prediction was made.
 *
 * When the ensemble model is active, the backend returns `eegnet_contribution`
 * (how much the CNN relied on raw brain waves) and `heuristic_contribution`
 * (how much the neuroscience-based feature rules contributed).  This component
 * renders that as a compact, human-readable bar + label.
 *
 * Only renders when both contribution values are present (i.e. the ensemble
 * path was used). Returns null otherwise -- safe to drop anywhere.
 */

interface EnsembleExplanationProps {
  /** EEGNet CNN contribution (0-1) */
  eegnetContribution?: number;
  /** Feature-based heuristic contribution (0-1) */
  heuristicContribution?: number;
  /** Epoch signal quality (0-1), if available */
  epochQuality?: number;
}

export function EnsembleExplanation({
  eegnetContribution,
  heuristicContribution,
  epochQuality,
}: EnsembleExplanationProps) {
  // Only show when ensemble is active (both values present)
  if (eegnetContribution == null || heuristicContribution == null) return null;

  const eegPct = Math.round(eegnetContribution * 100);
  const heuPct = Math.round(heuristicContribution * 100);

  // Quality label
  let qualityNote: string | null = null;
  if (epochQuality != null) {
    if (epochQuality < 0.3) qualityNote = "noisy signal -- leaning on neuroscience patterns";
    else if (epochQuality < 0.5) qualityNote = "moderate signal quality";
  }

  return (
    <div
      className="space-y-1.5 pt-1"
      data-testid="ensemble-explanation"
    >
      <p className="text-[10px] text-muted-foreground leading-relaxed">
        This reading is {eegPct}% based on brain wave patterns
        {" "}and {heuPct}% based on neuroscience rules.
      </p>
      {/* Stacked contribution bar */}
      <div className="flex h-1.5 rounded-full overflow-hidden bg-muted/40">
        <div
          data-testid="eegnet-bar"
          className="h-full bg-violet-500 transition-all duration-300"
          style={{ width: `${eegPct}%` }}
          title={`Brain waves: ${eegPct}%`}
        />
        <div
          data-testid="heuristic-bar"
          className="h-full bg-cyan-500 transition-all duration-300"
          style={{ width: `${heuPct}%` }}
          title={`Neuroscience rules: ${heuPct}%`}
        />
      </div>
      <div className="flex items-center justify-between text-[9px] text-muted-foreground/70">
        <span className="flex items-center gap-1">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-violet-500" />
          Brain waves {eegPct}%
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-cyan-500" />
          Neuro rules {heuPct}%
        </span>
      </div>
      {qualityNote && (
        <p
          className="text-[9px] text-amber-400/70 italic"
          data-testid="quality-note"
        >
          {qualityNote}
        </p>
      )}
    </div>
  );
}
