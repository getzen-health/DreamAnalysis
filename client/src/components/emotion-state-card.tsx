import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
  Smile, CloudSun, AlertTriangle, Battery, Brain, TreePine, Zap, Minus, Lightbulb,
  type LucideIcon,
} from "lucide-react";

export interface EmotionStateCardProps {
  emotion: string;       // "happy" | "sad" | "angry" | "fear" | "surprise" | "neutral"
  valence: number;       // -1 to 1
  arousal: number;       // 0 to 1
  stressIndex?: number;  // 0 to 1
  focusIndex?: number;   // 0 to 1
  confidence?: number;   // 0 to 1
  source?: string;       // "voice" | "eeg" | "health"
}

function getStateLabel(valence: number, arousal: number): { icon: LucideIcon; iconColor: string; label: string } {
  if (valence > 0.3 && arousal > 0.6)  return { icon: Smile, iconColor: "#4ade80", label: "You're energized" };
  if (valence > 0.3 && arousal <= 0.6) return { icon: CloudSun, iconColor: "#0891b2", label: "You're in a calm state" };
  if (valence <= -0.3 && arousal > 0.6) return { icon: AlertTriangle, iconColor: "#e879a8", label: "You're feeling stressed" };
  if (valence <= -0.3 && arousal <= 0.6) return { icon: Battery, iconColor: "#6366f1", label: "You're low on energy" };
  if (valence > 0 && arousal > 0.5)    return { icon: Brain, iconColor: "#6366f1", label: "You're alert and focused" };
  if (valence > 0)                     return { icon: TreePine, iconColor: "#4ade80", label: "You're relaxed" };
  if (arousal > 0.6)                   return { icon: Zap, iconColor: "#d4a017", label: "You're tense" };
  return { icon: Minus, iconColor: "#94a3b8", label: "You're in neutral mode" };
}

function getSuggestion(valence: number, arousal: number, stressIndex?: number): string {
  const stress = stressIndex ?? (arousal > 0.6 && valence < 0 ? 0.7 : 0);
  const energy = arousal;
  const calm = 1 - (stressIndex ?? arousal * 0.5);
  const focus = arousal * 0.6 + valence * 0.4;

  if (stress > 0.55) return "Try 4-7-8 breathing to reset";
  if (energy > 0.6 && focus > 0.5) return "Good time for deep work";
  if (energy > 0.6 && focus <= 0.5) return "Try a short walk first";
  if (energy <= 0.4 && calm > 0.5)  return "Good time for creative or reflective work";
  if (energy <= 0.4 && calm <= 0.5) return "Rest or light activity recommended";
  return "Check in again in an hour";
}

function deriveEnergy(arousal: number): number {
  return Math.round(Math.max(0, Math.min(1, arousal)) * 100);
}

function deriveCalm(arousal: number, stressIndex?: number): number {
  const raw = stressIndex !== undefined ? 1 - stressIndex : 1 - arousal;
  return Math.round(Math.max(0, Math.min(1, raw)) * 100);
}

function deriveFocus(arousal: number, valence: number, focusIndex?: number): number {
  if (focusIndex !== undefined) return Math.round(Math.max(0, Math.min(1, focusIndex)) * 100);
  return Math.round(Math.max(0, Math.min(1, arousal * 0.6 + valence * 0.4)) * 100);
}

const SOURCE_LABELS: Record<string, string> = {
  voice: "via voice",
  eeg: "via EEG",
  health: "via health",
};

export default function EmotionStateCard({
  valence,
  arousal,
  stressIndex,
  focusIndex,
  confidence,
  source,
}: EmotionStateCardProps) {
  const { icon: StateIcon, iconColor, label } = getStateLabel(valence, arousal);
  const suggestion = getSuggestion(valence, arousal, stressIndex);

  const energy = deriveEnergy(arousal);
  const calm   = deriveCalm(arousal, stressIndex);
  const focus  = deriveFocus(arousal, valence, focusIndex);

  return (
    <Card className="p-5 space-y-4 bg-gray-900 border-gray-800">
      {/* State header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <StateIcon className="w-9 h-9" style={{ color: iconColor }} />
          <span className="text-base font-semibold text-gray-100">{label}</span>
        </div>
        <div className="flex items-center gap-2">
          {confidence !== undefined && (
            <span className="text-xs text-gray-500">
              {Math.round(confidence * 100)}% conf
            </span>
          )}
          {source && SOURCE_LABELS[source] && (
            <Badge
              variant="outline"
              className="text-[10px] border-gray-700 text-gray-400 bg-gray-800/50"
            >
              {SOURCE_LABELS[source]}
            </Badge>
          )}
        </div>
      </div>

      {/* Progress bars */}
      <div className="space-y-3">
        <div className="space-y-1.5">
          <span className="text-xs text-gray-400">Energy</span>
          <Progress value={energy} className="h-2 bg-gray-800" />
        </div>
        <div className="space-y-1.5">
          <span className="text-xs text-gray-400">Calm</span>
          <Progress value={calm} className="h-2 bg-gray-800" />
        </div>
        <div className="space-y-1.5">
          <span className="text-xs text-gray-400">Focus</span>
          <Progress value={focus} className="h-2 bg-gray-800" />
        </div>
      </div>

      {/* Suggestion */}
      <p className="text-xs italic text-gray-400 flex items-center gap-1.5">
        <Lightbulb className="w-3 h-3 text-amber-400 shrink-0" />
        {suggestion}
      </p>
    </Card>
  );
}
