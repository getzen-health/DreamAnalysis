/**
 * EFSVitalCard — Individual vital sign card for the Emotional Fitness Score.
 *
 * Displays icon + name + score + mini sparkline + insight text.
 * If unavailable, shows Lock icon + unlockHint with muted styling.
 * Expand/collapse on tap reveals explanation + tips.
 */

import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Lock, ChevronDown, ChevronUp } from "lucide-react";
import {
  LineChart,
  Line,
  ResponsiveContainer,
} from "recharts";
import type { EFSVitalData } from "@/lib/ml-api";

// ── Static explanations per vital ─────────────────────────────────────────────

const VITAL_INFO: Record<string, { explanation: string; tips: string }> = {
  resilience: {
    explanation: "How quickly your emotions return to baseline after a negative event.",
    tips: "Practice 4-7-8 breathing. Regular sleep improves recovery.",
  },
  regulation: {
    explanation: "How effectively you manage emotional intensity.",
    tips: "Try biofeedback exercises. Even 2 minutes helps.",
  },
  awareness: {
    explanation: "How accurately you perceive your own emotions.",
    tips: "Pause before answering. Notice body sensations first.",
  },
  range: {
    explanation: "How many distinct emotions you can differentiate.",
    tips: "Be more specific: frustrated? disappointed? anxious?",
  },
  stability: {
    explanation: "How consistent your emotional baseline is.",
    tips: "Consistent sleep, meals, and routines help.",
  },
};

// ── Score color ───────────────────────────────────────────────────────────────

function vitalScoreColor(score: number): string {
  if (score >= 70) return "#0891b2";
  if (score >= 40) return "#d4a017";
  return "#e879a8";
}

function vitalScoreClass(score: number): string {
  if (score >= 70) return "text-cyan-400";
  if (score >= 40) return "text-amber-400";
  return "text-rose-400";
}

// ── Props ─────────────────────────────────────────────────────────────────────

interface EFSVitalCardProps {
  name: string;
  icon: React.ElementType;
  vital: EFSVitalData;
}

// ── Component ─────────────────────────────────────────────────────────────────

export function EFSVitalCard({ name, icon: Icon, vital }: EFSVitalCardProps) {
  const [expanded, setExpanded] = useState(false);
  const info = VITAL_INFO[name];
  const isAvailable = vital.status === "available" && vital.score !== null;

  // Sparkline data — take last 14 points
  const sparkData = vital.history.slice(-14).map((h, i) => ({
    idx: i,
    score: h.score,
  }));

  if (!isAvailable) {
    return (
      <Card className="bg-card/50 rounded-xl border border-border/30 shadow-sm opacity-60">
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            <Lock className="h-5 w-5 text-muted-foreground/50 shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-muted-foreground/70 capitalize">{name}</p>
              {vital.unlockHint && (
                <p className="text-xs text-muted-foreground/50 mt-0.5">{vital.unlockHint}</p>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const score = vital.score!;
  const lineColor = vitalScoreColor(score);

  return (
    <Card className="bg-card rounded-xl border border-border/50 shadow-sm">
      <CardContent className="p-4">
        {/* Top row: icon + name + score + sparkline */}
        <button
          className="flex items-center gap-3 w-full text-left"
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
        >
          <Icon className="h-5 w-5 text-cyan-400 shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between">
              <p className="text-sm font-medium text-foreground capitalize">{name}</p>
              <span className={`text-lg font-bold tabular-nums ${vitalScoreClass(score)}`}>
                {score}
              </span>
            </div>
            {/* Mini sparkline */}
            {sparkData.length > 1 && (
              <div className="h-6 w-full mt-1">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={sparkData}>
                    <Line
                      type="monotone"
                      dataKey="score"
                      stroke={lineColor}
                      strokeWidth={1.5}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
          <div className="shrink-0 ml-1">
            {expanded ? (
              <ChevronUp className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            )}
          </div>
        </button>

        {/* Insight text */}
        {vital.insight && (
          <p className="text-xs text-muted-foreground mt-2">{vital.insight}</p>
        )}

        {/* Expanded: explanation + tips */}
        {expanded && info && (
          <div className="mt-3 pt-3 border-t border-border/30 space-y-2">
            <p className="text-xs text-zinc-300">{info.explanation}</p>
            <p className="text-xs text-muted-foreground/70 italic">{info.tips}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
