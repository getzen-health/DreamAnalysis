import { type ChannelQuality } from "@/lib/signal-quality";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface SignalQualityIndicatorProps {
  channels: ChannelQuality[];
}

const STATUS_COLORS: Record<ChannelQuality["status"], string> = {
  good: "bg-green-500",
  fair: "bg-amber-500",
  poor: "bg-red-500",
  disconnected: "bg-gray-400",
};

const STATUS_LABELS: Record<ChannelQuality["status"], string> = {
  good: "Good signal",
  fair: "Fair signal — adjust headband",
  poor: "Poor signal — check electrode contact",
  disconnected: "No signal — electrode not touching skin",
};

/**
 * Displays 4 small colored dots representing per-electrode signal quality.
 *
 * Layout: horizontal row of dots with electrode labels beneath:
 *   [TP9] [AF7] [AF8] [TP10]
 *
 * Colors:
 *   Green = good
 *   Amber = fair
 *   Red = poor
 *   Gray = disconnected
 *
 * Tapping/hovering shows a tooltip with channel name, status, and recommendation.
 */
export function SignalQualityIndicator({ channels }: SignalQualityIndicatorProps) {
  if (channels.length === 0) return null;

  return (
    <TooltipProvider delayDuration={200}>
      <div className="flex items-center gap-3">
        {channels.map((ch) => (
          <Tooltip key={ch.channel}>
            <TooltipTrigger asChild>
              <div className="flex flex-col items-center gap-0.5 cursor-default">
                <div
                  className={`h-3 w-3 rounded-full ${STATUS_COLORS[ch.status]} transition-colors`}
                  aria-label={`${ch.channel}: ${ch.status}`}
                />
                <span className="text-[10px] text-muted-foreground font-mono leading-none">
                  {ch.channel}
                </span>
              </div>
            </TooltipTrigger>
            <TooltipContent side="bottom" className="max-w-[200px]">
              <p className="font-medium">{ch.channel}</p>
              <p className="text-xs text-muted-foreground">
                {STATUS_LABELS[ch.status]}
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                Amplitude: {ch.amplitudeUv.toFixed(1)} uV
              </p>
              {ch.status !== "disconnected" && (
                <p className="text-xs text-muted-foreground">
                  Spectral flatness: {(ch.spectralFlatness * 100).toFixed(0)}%
                </p>
              )}
            </TooltipContent>
          </Tooltip>
        ))}
      </div>
    </TooltipProvider>
  );
}
