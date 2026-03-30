/**
 * DreamSummaryCard — shareable morning dream summary.
 *
 * Rendered in-page as a styled card. The user can screenshot/share it
 * using the native Web Share API or long-press on mobile.
 *
 * Props mirror the dreamAnalysis response from POST /api/study/morning.
 */

import { useRef } from "react";
import { Moon, Sparkles, Share2 } from "lucide-react";
import { Button } from "@/components/ui/button";

interface DreamAnalysis {
  symbols: Array<{ symbol: string; meaning: string }>;
  emotions: string[];
  themes: string[];
  insights: string;
  morningMoodPrediction: string;
}

interface Props {
  dreamText: string;
  analysis: DreamAnalysis;
  date?: string; // ISO date string
}

const THEME_ICONS: Record<string, string> = {
  water: "🌊",
  flying: "🦅",
  falling: "🍂",
  chasing: "🏃",
  house: "🏠",
  forest: "🌲",
  ocean: "🌊",
  city: "🌆",
  family: "👨‍👩‍👧",
  work: "💼",
  default: "✨",
};

function themeIcon(theme: string): string {
  const lower = theme.toLowerCase();
  for (const [k, v] of Object.entries(THEME_ICONS)) {
    if (lower.includes(k)) return v;
  }
  return THEME_ICONS.default;
}

export function DreamSummaryCard({ dreamText, analysis, date }: Props) {
  const cardRef = useRef<HTMLDivElement>(null);

  const dateStr = date
    ? new Date(date).toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric" })
    : new Date().toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric" });

  const primaryTheme = analysis.themes?.[0] ?? "dream";
  const primarySymbol = analysis.symbols?.[0];
  const primaryEmotion = analysis.emotions?.[0] ?? "";
  const icon = themeIcon(primaryTheme);

  const handleShare = async () => {
    const text = [
      `🌙 Dream from ${dateStr}`,
      `Theme: ${primaryTheme}`,
      primarySymbol ? `Symbol: ${primarySymbol.symbol} — ${primarySymbol.meaning}` : "",
      `Insight: ${analysis.insights}`,
      `Tomorrow: ${analysis.morningMoodPrediction}`,
      "",
      "#NeuralDreamWorkshop #DreamJournal",
    ]
      .filter(Boolean)
      .join("\n");

    if (navigator.share) {
      try {
        await navigator.share({ title: "My dream", text });
      } catch {
        // User cancelled — no-op
      }
    } else {
      await navigator.clipboard.writeText(text);
    }
  };

  return (
    <div className="space-y-3">
      {/* The shareable card itself */}
      <div
        ref={cardRef}
        className="rounded-2xl overflow-hidden border border-primary/20"
        style={{
          background: "linear-gradient(135deg, hsl(240 15% 8%), hsl(260 20% 12%) 60%, hsl(240 18% 10%))",
        }}
      >
        {/* Top bar */}
        <div className="flex items-center justify-between px-5 pt-5 pb-3">
          <div className="flex items-center gap-2">
            <Moon className="h-4 w-4 text-primary opacity-70" />
            <span className="text-xs text-muted-foreground font-medium">{dateStr}</span>
          </div>
          <span className="text-xs text-muted-foreground opacity-60">NeuralDreamWorkshop</span>
        </div>

        {/* Theme + icon */}
        <div className="px-5 pb-3 flex items-center gap-4">
          <div
            className="w-14 h-14 rounded-2xl flex items-center justify-center text-3xl shrink-0"
            style={{ background: "hsl(var(--primary)/0.15)", border: "1px solid hsl(var(--primary)/0.2)" }}
          >
            {icon}
          </div>
          <div>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Primary theme</p>
            <p className="text-lg font-semibold capitalize text-foreground">{primaryTheme}</p>
            {primaryEmotion && (
              <p className="text-xs text-muted-foreground capitalize">{primaryEmotion}</p>
            )}
          </div>
        </div>

        {/* Dream text excerpt */}
        {dreamText && (
          <div className="mx-5 mb-3 px-3 py-2 rounded-lg bg-white/4 border border-white/8">
            <p className="text-xs text-foreground/70 italic leading-relaxed line-clamp-2">
              "{dreamText.length > 120 ? dreamText.slice(0, 120) + "…" : dreamText}"
            </p>
          </div>
        )}

        {/* Symbol */}
        {primarySymbol && (
          <div className="mx-5 mb-3 flex gap-3 items-start">
            <div className="text-primary shrink-0 mt-0.5">
              <Sparkles className="h-3.5 w-3.5" />
            </div>
            <div>
              <p className="text-xs font-medium text-foreground">{primarySymbol.symbol}</p>
              <p className="text-[11px] text-muted-foreground leading-snug">{primarySymbol.meaning}</p>
            </div>
          </div>
        )}

        {/* Insight */}
        {analysis.insights && (
          <div className="mx-5 mb-3 rounded-lg bg-primary/8 border border-primary/15 px-3 py-2">
            <p className="text-xs text-foreground/80 leading-relaxed">{analysis.insights}</p>
          </div>
        )}

        {/* Mood prediction */}
        {analysis.morningMoodPrediction && (
          <div className="px-5 pb-5">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Today's forecast</p>
            <p className="text-xs text-foreground/70 leading-relaxed">{analysis.morningMoodPrediction}</p>
          </div>
        )}
      </div>

      {/* Share button */}
      <Button
        variant="outline"
        size="sm"
        onClick={handleShare}
        className="w-full gap-2 text-xs"
      >
        <Share2 className="h-3.5 w-3.5" />
        Share dream summary
      </Button>
    </div>
  );
}
