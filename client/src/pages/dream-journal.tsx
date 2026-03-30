import { useState, useEffect, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useDevice } from "@/hooks/use-device";
import { useAuth } from "@/hooks/use-auth";
import { useLocation } from "wouter";
import { Moon, PenLine, Brain } from "lucide-react";

/* ---------- types ---------- */
interface DreamEpisode {
  startTime: string;
  duration: number;
  intensity: number;
}

interface DreamFrameBuffer {
  dreamIntensity: number;
  remLikelihood: number;
  valence?: number;
  arousal?: number;
  lucidityScore?: number;
  lucidityState?: string;
  thetaActivity?: number;
  betaActivation?: number;
  eyeMovementIndex?: number;
  dominantEmotion?: string;
}

interface DreamReport {
  narrative: string;
  primaryTheme: string;
  keyInsight: string;
  morningMoodPrediction: string;
  eegSummary: string;
  episode: {
    durationMinutes: number;
    peakIntensity: number;
    peakLucidityState: string;
  };
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

function generateSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

/* ========== Component ========== */
export default function DreamJournal() {
  const [, navigate] = useLocation();
  const { latestFrame, state: deviceState } = useDevice();
  const { user } = useAuth();
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
  const wasDreamingRef   = useRef(false);
  const dreamStartRef    = useRef<Date | null>(null);
  const lastIntensityRef = useRef(0);

  // EEG frame buffer for dream frames
  const frameBufferRef   = useRef<DreamFrameBuffer[]>([]);
  const sessionIdRef     = useRef<string>(generateSessionId());
  const [totalFramesSaved, setTotalFramesSaved] = useState(0);

  // Report state
  const [report, setReport]           = useState<DreamReport | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);

  // Collect EEG frames while dreaming
  useEffect(() => {
    if (!isStreaming || !analysis) return;
    lastIntensityRef.current = dreamIntensity;

    if (isDreaming) {
      // Buffer this frame
      const frame: DreamFrameBuffer = {
        dreamIntensity: dreamDetection?.dream_intensity ?? 0,
        remLikelihood:  dreamDetection?.rem_likelihood ?? 0,
        valence:        analysis.emotions?.valence,
        arousal:        analysis.emotions?.arousal,
        lucidityScore:  dreamDetection?.lucidity_estimate,
        lucidityState:  analysis.lucid_dream?.state ?? undefined,
        thetaActivity:  analysis.band_powers?.theta != null
          ? Math.min(1, analysis.band_powers.theta / 50)
          : undefined,
        betaActivation: analysis.band_powers?.beta != null
          ? Math.min(1, analysis.band_powers.beta / 30)
          : undefined,
        eyeMovementIndex: analysis.band_powers?.delta != null
          ? Math.min(1, analysis.band_powers.delta / 100)
          : undefined,
        dominantEmotion: analysis.emotions?.emotion ?? undefined,
      };
      frameBufferRef.current.push(frame);

      if (!wasDreamingRef.current) {
        dreamStartRef.current = new Date();
        wasDreamingRef.current = true;
      }
    } else if (!isDreaming && wasDreamingRef.current && dreamStartRef.current) {
      // Episode ended — flush buffered frames to the server
      const frames = [...frameBufferRef.current];
      frameBufferRef.current = [];
      wasDreamingRef.current = false;

      const duration = Math.max(1, Math.round((Date.now() - dreamStartRef.current.getTime()) / 60000));
      const startTime = dreamStartRef.current.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
      dreamStartRef.current = null;

      setEpisodes((prev) => [...prev, { startTime, duration, intensity: lastIntensityRef.current }]);

      if (frames.length > 0 && user?.id) {
        fetch("/api/dream-frames", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            frames,
            sessionId: sessionIdRef.current,
            userId: user.id,
          }),
        })
          .then((r) => r.json())
          .then((data: { saved?: number }) => {
            if (data.saved) setTotalFramesSaved((n) => n + data.saved!);
          })
          .catch(() => { /* non-fatal — frames buffered client-side already */ });
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isDreaming, isStreaming]);

  async function handleGenerateReport() {
    if (!user?.id) return;

    // If still dreaming, flush whatever we have buffered first
    const pendingFrames = [...frameBufferRef.current];
    if (pendingFrames.length > 0) {
      frameBufferRef.current = [];
      try {
        const r = await fetch("/api/dream-frames", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            frames: pendingFrames,
            sessionId: sessionIdRef.current,
            userId: user.id,
          }),
        });
        const d = await r.json() as { saved?: number };
        if (d.saved) setTotalFramesSaved((n) => n + d.saved!);
      } catch { /* ignore */ }
    }

    setIsGenerating(true);
    setGenerateError(null);
    try {
      const r = await fetch("/api/dream-session-complete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId: sessionIdRef.current,
          userId: user.id,
        }),
      });
      if (!r.ok) {
        const err = await r.json() as { message?: string };
        throw new Error(err.message || `HTTP ${r.status}`);
      }
      const data = await r.json() as {
        narrative?: string;
        primaryTheme?: string;
        keyInsight?: string;
        morningMoodPrediction?: string;
        eegSummary?: string;
        episode?: DreamReport["episode"];
        message?: string;
      };

      if (data.message && !data.narrative) {
        setGenerateError(data.message);
        return;
      }

      setReport({
        narrative: data.narrative ?? "",
        primaryTheme: data.primaryTheme ?? "neutral",
        keyInsight: data.keyInsight ?? "",
        morningMoodPrediction: data.morningMoodPrediction ?? "neutral",
        eegSummary: data.eegSummary ?? "",
        episode: data.episode ?? { durationMinutes: 0, peakIntensity: 0, peakLucidityState: "non_lucid" },
      });

      // Rotate to a new session ID for the next night
      sessionIdRef.current = generateSessionId();
      setTotalFramesSaved(0);
    } catch (e) {
      setGenerateError(e instanceof Error ? e.message : "Failed to generate report");
    } finally {
      setIsGenerating(false);
    }
  }

  return (
    <div className="max-w-lg mx-auto px-4 py-8 pb-24 space-y-4">

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
              Record a dream below using voice or text. Optional overnight EEG can add automatic dream detection later.
            </p>
          </div>
        ) : (
          <div className="space-y-4" aria-live="polite" aria-label="Dream detection status">
            {/* Sleep stage pill */}
            <div className="flex items-center gap-3">
              <div
                className="w-3 h-3 rounded-full shrink-0"
                aria-hidden="true"
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
                  <p className="text-sm font-medium">Dream detected — recording EEG</p>
                  <p className="text-xs text-muted-foreground">
                    {dreamProbability}% probability · {intensityLabel(dreamIntensity)} activity
                    {frameBufferRef.current.length > 0 && (
                      <span className="ml-2 text-primary/70">
                        · {frameBufferRef.current.length} frames buffered
                      </span>
                    )}
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
                  {totalFramesSaved > 0 && (
                    <span className="ml-2 text-primary/60">· {totalFramesSaved} frames saved from tonight</span>
                  )}
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
              : "Manual dream entries are the default here. Auto-detected episodes appear later if you add overnight EEG."}
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

        {/* Generate dream report button — shown when episodes exist */}
        {(episodes.length > 0 || totalFramesSaved > 0) && (
          <div className="mt-4 pt-3 border-t border-border/20">
            <Button
              className="w-full"
              variant="outline"
              onClick={handleGenerateReport}
              disabled={isGenerating || !user?.id}
            >
              <Brain className="w-4 h-4 mr-2" />
              {isGenerating ? "Generating from EEG..." : "Generate Dream Report from EEG"}
            </Button>
            {generateError && (
              <p className="text-xs text-destructive mt-2 text-center">{generateError}</p>
            )}
          </div>
        )}
      </Card>

      {/* ── Card 3: EEG-generated dream narrative ────────────────────────── */}
      {report && (
        <Card className="p-5 space-y-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1">
              Based on your EEG activity tonight
            </p>
            <p className="text-sm leading-relaxed">{report.narrative}</p>
          </div>

          {report.eegSummary && (
            <p className="text-xs text-muted-foreground italic border-l-2 border-primary/30 pl-3">
              {report.eegSummary}
            </p>
          )}

          <div className="flex items-center gap-2 flex-wrap">
            <span
              className="inline-flex items-center rounded-full px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide"
              style={{
                background: "hsl(262,45%,55%,0.15)",
                color: "hsl(262,45%,75%)",
                border: "1px solid hsl(262,45%,55%,0.3)",
              }}
            >
              {report.primaryTheme}
            </span>
            <span
              className="inline-flex items-center rounded-full px-2.5 py-0.5 text-[10px] font-semibold"
              style={{
                background:
                  report.morningMoodPrediction === "positive"
                    ? "hsl(152,60%,48%,0.15)"
                    : report.morningMoodPrediction === "negative"
                    ? "hsl(0,70%,55%,0.15)"
                    : "hsl(220,50%,50%,0.15)",
                color:
                  report.morningMoodPrediction === "positive"
                    ? "hsl(152,60%,65%)"
                    : report.morningMoodPrediction === "negative"
                    ? "hsl(0,70%,70%)"
                    : "hsl(220,50%,70%)",
                border:
                  report.morningMoodPrediction === "positive"
                    ? "1px solid hsl(152,60%,48%,0.3)"
                    : report.morningMoodPrediction === "negative"
                    ? "1px solid hsl(0,70%,55%,0.3)"
                    : "1px solid hsl(220,50%,50%,0.3)",
              }}
            >
              Morning: {report.morningMoodPrediction}
            </span>
            {report.episode.peakLucidityState !== "non_lucid" && (
              <span
                className="inline-flex items-center rounded-full px-2.5 py-0.5 text-[10px] font-semibold"
                style={{
                  background: "hsl(38,85%,58%,0.15)",
                  color: "hsl(38,85%,70%)",
                  border: "1px solid hsl(38,85%,58%,0.3)",
                }}
              >
                {report.episode.peakLucidityState.replace("_", " ")}
              </span>
            )}
          </div>

          {report.keyInsight && (
            <div className="rounded-lg bg-muted/20 px-3 py-2.5">
              <p className="text-xs text-muted-foreground font-medium mb-0.5">Insight</p>
              <p className="text-xs leading-relaxed">{report.keyInsight}</p>
            </div>
          )}
        </Card>
      )}

      {/* ── Card 4: Record on waking ──────────────────────────────────────── */}
      <button
        onClick={() => navigate("/research/morning")}
        aria-label="Record this morning's dream — open dream journal entry"
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
