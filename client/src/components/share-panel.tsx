/**
 * SharePanel — Bottom sheet for previewing and sharing generated cards.
 *
 * Uses shadcn Sheet (bottom variant) with glass-card styling.
 * Template selector, format toggle, and 3 action buttons.
 */

import { useState, useEffect, useCallback } from "react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import {
  Share2,
  Copy,
  Download,
  BarChart3,
  Dumbbell,
  CalendarDays,
  Brain,
  Flame,
  Loader2,
  Check,
  X,
} from "lucide-react";
import {
  generateShareCard,
  type ShareTemplate,
  type ShareFormat,
  type ShareData,
} from "@/components/share-card-generator";
import { shareImage, copyImageToClipboard, saveImageToPhotos } from "@/lib/share-utils";

// ── Types ──────────────────────────────────────────────────────────────────

interface SharePanelProps {
  open: boolean;
  onClose: () => void;
  data: ShareData;
  /** Which template to show first. Defaults to "daily-overview". */
  defaultTemplate?: ShareTemplate;
}

// ── Template definitions ───────────────────────────────────────────────────

const TEMPLATES: {
  id: ShareTemplate;
  label: string;
  icon: React.ElementType;
}[] = [
  { id: "daily-overview", label: "Daily", icon: BarChart3 },
  { id: "workout-summary", label: "Workout", icon: Dumbbell },
  { id: "weekly-summary", label: "Weekly", icon: CalendarDays },
  { id: "brain-report", label: "Brain", icon: Brain },
  { id: "habit-streak", label: "Streak", icon: Flame },
];

// ── Component ──────────────────────────────────────────────────────────────

export function SharePanel({
  open,
  onClose,
  data,
  defaultTemplate = "daily-overview",
}: SharePanelProps) {
  const [template, setTemplate] = useState<ShareTemplate>(defaultTemplate);
  const [format, setFormat] = useState<ShareFormat>("stories");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);

  // Reset when opened with a new default template
  useEffect(() => {
    if (open) {
      setTemplate(defaultTemplate);
      setCopySuccess(false);
      setCopyFailed(false);
    }
  }, [open, defaultTemplate]);

  // Generate card whenever template, format, or data changes
  useEffect(() => {
    if (!open) return;

    let cancelled = false;
    setGenerating(true);
    setImageUrl(null);

    generateShareCard(template, data, format)
      .then((url) => {
        if (!cancelled) {
          setImageUrl(url);
          setGenerating(false);
        }
      })
      .catch(() => {
        if (!cancelled) setGenerating(false);
      });

    return () => {
      cancelled = true;
    };
  }, [open, template, format, data]);

  const handleCopy = useCallback(async () => {
    if (!imageUrl) return;
    const ok = await copyImageToClipboard(imageUrl);
    if (ok) {
      setCopySuccess(true);
      setCopyFailed(false);
      setTimeout(() => setCopySuccess(false), 2000);
    } else {
      setCopyFailed(true);
      setTimeout(() => setCopyFailed(false), 3000);
    }
  }, [imageUrl]);

  const handleSave = useCallback(async () => {
    if (!imageUrl) return;
    const filename = `antarai-${template}-${Date.now()}.png`;
    await saveImageToPhotos(imageUrl, filename);
  }, [imageUrl, template]);

  const handleShare = useCallback(async () => {
    if (!imageUrl) return;
    const filename = `antarai-${template}-${Date.now()}.png`;
    await shareImage(imageUrl, filename);
  }, [imageUrl, template]);

  return (
    <Sheet open={open} onOpenChange={(o) => !o && onClose()}>
      <SheetContent
        side="bottom"
        className="rounded-t-2xl border-t border-border/50 bg-background/80 backdrop-blur-xl max-h-[85vh] overflow-y-auto"
      >
        <SheetHeader className="pb-2">
          <SheetTitle className="text-base font-semibold">
            Share Card
          </SheetTitle>
          <SheetDescription className="text-xs text-muted-foreground">
            Choose a template and format, then share
          </SheetDescription>
        </SheetHeader>

        {/* Canvas preview */}
        <div className="relative w-full flex justify-center py-3">
          {generating ? (
            <div
              className="flex items-center justify-center rounded-xl bg-muted/30 border border-border/30"
              style={{
                width: format === "stories" ? 180 : 240,
                height: format === "stories" ? 320 : 240,
              }}
            >
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : imageUrl ? (
            <img
              src={imageUrl}
              alt="Share card preview"
              className="rounded-xl shadow-lg border border-border/20"
              style={{
                maxHeight: 320,
                width: "auto",
                objectFit: "contain",
              }}
            />
          ) : null}
        </div>

        {/* Template selector */}
        <div className="flex gap-2 overflow-x-auto py-2 px-1 no-scrollbar">
          {TEMPLATES.map((t) => {
            const Icon = t.icon;
            const active = template === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setTemplate(t.id)}
                className={`
                  flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium
                  whitespace-nowrap transition-all duration-200
                  ${
                    active
                      ? "bg-primary/15 text-primary border border-primary/30"
                      : "bg-muted/40 text-muted-foreground border border-transparent hover:bg-muted/60"
                  }
                `}
              >
                <Icon className="h-3.5 w-3.5" />
                {t.label}
              </button>
            );
          })}
        </div>

        {/* Format toggle */}
        <div className="flex items-center justify-center gap-1 py-2">
          <button
            onClick={() => setFormat("stories")}
            className={`px-3 py-1 rounded-l-lg text-xs font-medium transition-all duration-200 ${
              format === "stories"
                ? "bg-primary/15 text-primary"
                : "bg-muted/30 text-muted-foreground"
            }`}
          >
            Stories
          </button>
          <button
            onClick={() => setFormat("square")}
            className={`px-3 py-1 rounded-r-lg text-xs font-medium transition-all duration-200 ${
              format === "square"
                ? "bg-primary/15 text-primary"
                : "bg-muted/30 text-muted-foreground"
            }`}
          >
            Square
          </button>
        </div>

        {/* Action buttons */}
        <div className="flex gap-2 pt-2 pb-4">
          <button
            onClick={handleCopy}
            disabled={!imageUrl}
            className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl
              bg-muted/40 border border-border/30 text-sm font-medium
              text-foreground transition-all duration-200
              hover:bg-muted/60 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {copySuccess ? (
              <Check className="h-4 w-4 text-green-400" />
            ) : copyFailed ? (
              <X className="h-4 w-4 text-red-400" />
            ) : (
              <Copy className="h-4 w-4" />
            )}
            {copySuccess ? "Copied!" : copyFailed ? "Copy failed" : "Copy"}
          </button>

          <button
            onClick={handleSave}
            disabled={!imageUrl}
            className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl
              bg-muted/40 border border-border/30 text-sm font-medium
              text-foreground transition-all duration-200
              hover:bg-muted/60 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Download className="h-4 w-4" />
            Save
          </button>

          <button
            onClick={handleShare}
            disabled={!imageUrl}
            className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl
              bg-primary/15 border border-primary/30 text-sm font-medium
              text-primary transition-all duration-200
              hover:bg-primary/25 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Share2 className="h-4 w-4" />
            Share
          </button>
        </div>
      </SheetContent>
    </Sheet>
  );
}
