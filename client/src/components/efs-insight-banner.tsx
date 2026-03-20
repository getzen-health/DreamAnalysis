/**
 * EFSInsightBanner — Daily insight card for Emotional Fitness Score.
 *
 * Highlighted card with left accent border (cyan).
 * Insight text in normal weight, actionNudge in muted text below.
 * Share icon in top-right (copies text or uses navigator.share).
 */

import { Share2, Lightbulb } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

// ── Props ─────────────────────────────────────────────────────────────────────

interface EFSInsightBannerProps {
  insight: { text: string; type: string; actionNudge: string } | null;
}

// ── Component ─────────────────────────────────────────────────────────────────

export function EFSInsightBanner({ insight }: EFSInsightBannerProps) {
  if (!insight) return null;

  const handleShare = async () => {
    const shareText = `${insight.text}\n\n${insight.actionNudge}`;
    if (navigator.share) {
      try {
        await navigator.share({ text: shareText });
      } catch {
        // User cancelled or share failed — fall through to clipboard
        await copyToClipboard(shareText);
      }
    } else {
      await copyToClipboard(shareText);
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Clipboard API not available — silent fail
    }
  };

  return (
    <Card className="bg-cyan-950/20 rounded-xl border border-cyan-500/20 shadow-sm border-l-2 border-l-cyan-500">
      <CardContent className="p-4">
        <div className="flex items-start gap-3">
          <Lightbulb className="h-5 w-5 text-cyan-400 shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="text-xs font-medium text-cyan-400/80 uppercase tracking-wider mb-1">Today's Insight</p>
            <p className="text-sm text-foreground leading-relaxed">{insight.text}</p>
            {insight.actionNudge && (
              <p className="text-xs text-muted-foreground mt-2">{insight.actionNudge}</p>
            )}
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="shrink-0 h-8 w-8 text-muted-foreground hover:text-foreground"
            onClick={handleShare}
            aria-label="Share insight"
          >
            <Share2 className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
