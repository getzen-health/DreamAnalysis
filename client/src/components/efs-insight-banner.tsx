/**
 * EFSInsightBanner — Daily insight card for Emotional Fitness Score.
 *
 * Highlighted card with left accent border (cyan).
 * Insight text in normal weight, actionNudge in muted text below.
 * Share icon in top-right (copies text or uses navigator.share).
 */

import { Share2 } from "lucide-react";
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
    <Card className="bg-card rounded-xl border border-border/50 shadow-sm border-l-2 border-l-cyan-500">
      <CardContent className="p-4">
        <div className="flex items-start gap-3">
          <div className="flex-1 min-w-0">
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
