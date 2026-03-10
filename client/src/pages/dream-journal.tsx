import { useState, useEffect, useRef } from "react";
import { Card } from "@/components/ui/card";
import { useDevice } from "@/hooks/use-device";
import { useLocation } from "wouter";
import { Moon, PenLine } from "lucide-react";

/* ---------- types ---------- */
interface DreamEpisode {
  startTime: string;
  duration: number;
  intensity: number;
}

/* ---------- helpers ---------- */
const STAGE_LABELS: Record<string, { label: string; color: string }> = {
  Wake:  { label: "Awake",       color: "hsl(38, 85%, 58%)" },
  N1:    { label: "Light Sleep", color: "hsl(200, 70%, 55%)" },
  N2:    { label: "Sleep",       color: "hsl(220, 50%, 50%)" },
  N3:    { label: "Deep Sleep",  color: "hsl(262, 45%, 55%)" },
  REM:   { label: "REM",         color: "hsl(152, 60%, 48%)" },
};

function intensityLabel(pct: number): string {
  if (pct >= 70) return "vivid";
  if (pct >= 40) return "moderate";
  return "faint";
}

/* ========== Component ========== */
export default function DreamJournal() {
  const [, navigate] = useLocation();
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const analysis    = latestFrame?.analysis;

  const dreamDetection = analysis?.dream_detection;
  const sleepStaging   = analysis?.sleep_staging;

  const isDreaming       = dreamDetection?.is_dreaming ?? false;
  const dreamProbability = Math.round((dreamDetection?.probability    ?? 0) * 100);
  const remLikelihood    = Math.round((dreamDetection?.rem_likelihood  ?? 0) * 100);
  const dreamIntensity   = Math.round((dreamDetection?.dream_intensity ?? 0) * 100);
  const sleepStage       = sleepStaging?.stage ?? "Wake";

  const stageInfo = STAGE_LABELS[sleepStage] ?? STAGE_LABELS.Wake;

  // Track dream episodes (start/end transitions)
  const [episodes, setEpisodes] = useState<DreamEpisode[]>([]);
  const wasDreamingRef  = useRef(false);
  const dreamStartRef   = useRef<Date | null>(null);
  const lastIntensityRef = useRef(0);

  useEffect(() => {
    if (!isStreaming) return;
    lastIntensityRef.current = dreamIntensity;

    if (isDreaming && !wasDreamingRef.current) {
      dreamStartRef.current = new Date();
      wasDreamingRef.current = true;
    } else if (!isDreaming && wasDreamingRef.current && dreamStartRef.current) {
      const duration = Math.max(1, Math.round((Date.now() - dreamStartRef.current.getTime()) / 60000));
      const startTime = dreamStartRef.current.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
      setEpisodes((prev) => [...prev, { startTime, duration, intensity: lastIntensityRef.current }]);
      wasDreamingRef.current = false;
      dreamStartRef.current  = null;
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isDreaming, isStreaming]);

  return (
    <div className="max-w-lg mx-auto px-4 py-8 space-y-4">

      {/* ── Card 1: Tonight ──────────────────────────────────────────────── */}
      <Card className="p-5">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-4">
          Tonight
        </p>

        {!isStreaming ? (
          <div className="flex flex-col items-center gap-3 py-6 text-center">
            <Moon className="w-10 h-10 text-muted-foreground/40" />
            <p className="text-sm font-medium">Log a dream manually</p>
            <p className="text-xs text-muted-foreground max-w-[220px]">
              Record a dream below using voice or text. Connect Muse 2 while sleeping for automatic detection.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Sleep stage pill */}
            <div className="flex items-center gap-3">
              <div
                className="w-3 h-3 rounded-full shrink-0"
                style={{
                  background: stageInfo.color,
                  boxShadow: `0 0 8px ${stageInfo.color}80`,
                }}
              />
              <span className="text-base font-semibold" style={{ color: stageInfo.color }}>
                {stageInfo.label}
              </span>
              <span className="text-[10px] font-mono text-primary animate-pulse ml-auto">
                LIVE
              </span>
            </div>

            {/* Dreaming banner */}
            {isDreaming ? (
              <div
                className="rounded-xl px-4 py-3 flex items-center gap-3"
                style={{
                  background: "hsl(262,45%,55%,0.12)",
                  border: "1px solid hsl(262,45%,55%,0.35)",
                }}
              >
                <span className="text-xl animate-pulse">✨</span>
                <div>
                  <p className="text-sm font-medium">Dream detected</p>
                  <p className="text-xs text-muted-foreground">
                    {dreamProbability}% probability · {intensityLabel(dreamIntensity)} activity
                  </p>
                </div>
              </div>
            ) : (
              <div className="rounded-xl px-4 py-3 bg-muted/20 border border-border/30">
                <p className="text-sm text-muted-foreground">
                  No dream activity detected right now
                </p>
                <p className="text-xs text-muted-foreground/60 mt-0.5">
                  REM likelihood: {remLikelihood}%
                </p>
              </div>
            )}
          </div>
        )}
      </Card>

      {/* ── Card 2: Dream episodes ───────────────────────────────────────── */}
      <Card className="p-5">
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Episodes tonight
          </p>
          <span className="text-xs text-muted-foreground">
            {episodes.length} detected
          </span>
        </div>

        {episodes.length === 0 ? (
          <p className="text-xs text-muted-foreground py-4 text-center">
            {isStreaming
              ? "Dream episodes will appear here as you sleep."
              : "Auto-detected episodes appear here when streaming with Muse 2. Use the journal button to log memories manually."}
          </p>
        ) : (
          <div className="space-y-2">
            {[...episodes].reverse().map((ep, i) => (
              <div
                key={i}
                className="flex items-center gap-3 py-2 border-b border-border/20 last:border-0"
              >
                <span className="text-xl shrink-0">🌙</span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium">{ep.startTime}</p>
                  <p className="text-xs text-muted-foreground">
                    {ep.duration} min · {intensityLabel(ep.intensity)} dream
                  </p>
                </div>
                <div
                  className="w-8 h-8 rounded-full flex items-center justify-center text-[10px] font-semibold shrink-0"
                  style={{
                    background: "hsl(262,45%,55%,0.12)",
                    color: "hsl(262,45%,70%)",
                    border: "1px solid hsl(262,45%,55%,0.3)",
                  }}
                >
                  {ep.intensity}%
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* ── Card 3: Record on waking ─────────────────────────────────────── */}
      <button
        onClick={() => navigate("/research/morning")}
        className={`w-full flex items-center gap-3 rounded-xl px-4 py-3.5 transition-colors text-left ${
          !isStreaming
            ? "border border-primary/30 bg-primary/5 hover:bg-primary/10"
            : "border border-border/50 bg-muted/10 hover:bg-muted/20"
        }`}
      >
        <PenLine className={`w-4 h-4 shrink-0 ${!isStreaming ? "text-primary" : "text-muted-foreground"}`} />
        <div className="flex-1 min-w-0">
          <p className={`text-sm font-medium ${!isStreaming ? "text-primary" : ""}`}>Record this morning's dream</p>
          <p className="text-xs text-muted-foreground">
            Write what you remember — even a word counts
          </p>
        </div>
        <span className="text-muted-foreground/40 text-sm shrink-0">→</span>
      </button>

    </div>
  );
}
