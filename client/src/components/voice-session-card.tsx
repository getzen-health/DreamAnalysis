/**
 * VoiceSessionCard — continuous voice emotion monitoring.
 *
 * Renders a card with:
 *   - Idle: "Start Voice Session" button
 *   - Active: current emotion, EMA-smoothed valence gauge, elapsed time,
 *             mini timeline of the last 10 results, "Stop Session" button
 *   - Summary: dominant emotion, valence trend, total duration
 */

import { useMemo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Mic, MicOff, Square, BarChart2, Clock, TrendingUp, TrendingDown, Minus } from "lucide-react";
import { useVoiceSession, type VoiceSessionEntry } from "@/hooks/use-voice-session";

// ── helpers ──────────────────────────────────────────────────────────────────

const EMOTION_COLOR: Record<string, string> = {
  happy:    "#34d399",
  sad:      "#60a5fa",
  angry:    "#f87171",
  fear:     "#fbbf24",
  surprise: "#f472b6",
  neutral:  "#94a3b8",
};

const EMOTION_LABEL: Record<string, string> = {
  happy:    "Happy",
  sad:      "Sad",
  angry:    "Angry",
  fear:     "Fearful",
  surprise: "Surprised",
  neutral:  "Neutral",
};

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

/** valence -1..1 → percentage for Progress (0..100) */
function valenceToPercent(v: number): number {
  return Math.round(((v + 1) / 2) * 100);
}

function valenceBadgeClass(v: number): string {
  if (v >= 0.3) return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
  if (v <= -0.3) return "bg-rose-500/20 text-rose-400 border-rose-500/30";
  return "bg-muted/50 text-muted-foreground border-border/40";
}

function valenceLabel(v: number): string {
  if (v >= 0.3) return "Positive";
  if (v <= -0.3) return "Negative";
  return "Neutral";
}

/** Dominant emotion from an array of entries (most frequent). */
function dominantEmotion(entries: VoiceSessionEntry[]): string {
  if (entries.length === 0) return "neutral";
  const counts: Record<string, number> = {};
  for (const e of entries) {
    const em = e.result.emotion;
    counts[em] = (counts[em] ?? 0) + 1;
  }
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
}

/** Valence trend: compare first half vs second half of session. */
function valenceTrend(entries: VoiceSessionEntry[]): "up" | "down" | "flat" {
  if (entries.length < 2) return "flat";
  const mid = Math.floor(entries.length / 2);
  const first = entries.slice(0, mid).reduce((s, e) => s + e.result.valence, 0) / mid;
  const second = entries.slice(mid).reduce((s, e) => s + e.result.valence, 0) / (entries.length - mid);
  if (second - first > 0.1) return "up";
  if (first - second > 0.1) return "down";
  return "flat";
}

// ── mini timeline ─────────────────────────────────────────────────────────────

function MiniTimeline({ entries }: { entries: VoiceSessionEntry[] }) {
  const last10 = entries.slice(-10);
  if (last10.length === 0) return null;
  return (
    <div className="flex items-end gap-1 h-10">
      {last10.map((entry, i) => {
        const color = EMOTION_COLOR[entry.result.emotion] ?? "#94a3b8";
        const heightPct = Math.round(((entry.result.valence + 1) / 2) * 100);
        const minH = 6;
        const maxH = 40;
        const h = minH + Math.round((heightPct / 100) * (maxH - minH));
        return (
          <div
            key={i}
            title={`${EMOTION_LABEL[entry.result.emotion] ?? entry.result.emotion} — valence ${entry.result.valence.toFixed(2)}`}
            className="flex-1 rounded-sm transition-all duration-300"
            style={{ height: `${h}px`, backgroundColor: color, opacity: 0.7 + (i / last10.length) * 0.3 }}
          />
        );
      })}
    </div>
  );
}

// ── component ────────────────────────────────────────────────────────────────

interface VoiceSessionCardProps {
  userId?: string;
  /** Chunk interval in ms. Default: 15000. */
  chunkMs?: number;
}

export function VoiceSessionCard({ userId, chunkMs = 15000 }: VoiceSessionCardProps) {
  const {
    startSession,
    stopSession,
    isActive,
    isAnalyzing,
    results,
    currentEmotion,
    smoothedValence,
    smoothedArousal,
    duration,
    error,
  } = useVoiceSession({ userId, chunkMs });

  // summary computed only after session ends (not active, has results)
  const showSummary = !isActive && results.length > 0;
  const summary = useMemo(() => {
    if (!showSummary) return null;
    return {
      dominant: dominantEmotion(results),
      trend: valenceTrend(results),
      avgValence: results.reduce((s, e) => s + e.result.valence, 0) / results.length,
      totalDuration: results[results.length - 1]?.elapsed ?? 0,
      count: results.length,
    };
  }, [showSummary, results]);

  const nextChunkIn = isActive ? chunkMs / 1000 : null;

  return (
    <Card className="bg-card rounded-xl border border-border/50 shadow-sm">
      <CardContent className="p-4 space-y-3">

        {/* ── Header ─────────────────────────────────────────────────── */}
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-semibold">Voice Session</p>
            <p className="text-xs text-muted-foreground">
              {isActive
                ? `Recording every ${chunkMs / 1000}s`
                : showSummary
                ? "Session complete"
                : "Continuous emotion tracking"}
            </p>
          </div>
          {isActive && (
            <div className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-rose-500 animate-pulse" />
              <span className="text-xs font-mono text-rose-400">{formatDuration(duration)}</span>
            </div>
          )}
        </div>

        {/* ── Idle state ─────────────────────────────────────────────── */}
        {!isActive && !showSummary && (
          <div className="flex flex-col items-center gap-3 py-3">
            <Button
              size="lg"
              variant="outline"
              className="h-14 w-14 rounded-full border-2 border-primary/40 hover:border-primary hover:bg-primary/10 transition-all"
              onClick={startSession}
            >
              <Mic className="h-6 w-6 text-primary" />
            </Button>
            <p className="text-xs text-muted-foreground text-center">
              Mic stays on — emotion detected every {chunkMs / 1000}s
            </p>
            {error && (
              <p className="text-xs text-destructive text-center">{error}</p>
            )}
          </div>
        )}

        {/* ── Active state ─────────────────────────────────────────────── */}
        {isActive && (
          <div className="space-y-3">
            {/* Current emotion */}
            <div className="flex items-center gap-3">
              <div
                className="h-10 w-10 rounded-full flex items-center justify-center shrink-0"
                style={{
                  backgroundColor: `${EMOTION_COLOR[currentEmotion ?? "neutral"] ?? "#94a3b8"}20`,
                  border: `2px solid ${EMOTION_COLOR[currentEmotion ?? "neutral"] ?? "#94a3b8"}50`,
                }}
              >
                {isAnalyzing ? (
                  <div className="h-4 w-4 rounded-full border-2 border-primary border-t-transparent animate-spin" />
                ) : (
                  <MicOff
                    className="h-5 w-5"
                    style={{ color: EMOTION_COLOR[currentEmotion ?? "neutral"] ?? "#94a3b8" }}
                  />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold capitalize">
                  {currentEmotion
                    ? (EMOTION_LABEL[currentEmotion] ?? currentEmotion)
                    : "Waiting for first chunk…"}
                </p>
                {isAnalyzing && (
                  <p className="text-xs text-muted-foreground">Analyzing…</p>
                )}
              </div>
              {smoothedValence !== null && (
                <Badge className={`text-xs border shrink-0 ${valenceBadgeClass(smoothedValence)}`}>
                  {valenceLabel(smoothedValence)}
                </Badge>
              )}
            </div>

            {/* Valence gauge */}
            {smoothedValence !== null && (
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Valence</span>
                  <span className="font-mono">{smoothedValence >= 0 ? "+" : ""}{smoothedValence.toFixed(2)}</span>
                </div>
                <Progress value={valenceToPercent(smoothedValence)} className="h-1.5" />
              </div>
            )}

            {/* Arousal gauge */}
            {smoothedArousal !== null && (
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Arousal</span>
                  <span className="font-mono">{Math.round(smoothedArousal * 100)}%</span>
                </div>
                <Progress value={Math.round(smoothedArousal * 100)} className="h-1.5" />
              </div>
            )}

            {/* Mini timeline */}
            {results.length > 0 && (
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground flex items-center gap-1">
                  <BarChart2 className="h-3 w-3" />
                  Last {Math.min(10, results.length)} readings
                </p>
                <MiniTimeline entries={results} />
              </div>
            )}

            {/* Stats row */}
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {formatDuration(duration)}
              </span>
              <span>{results.length} sample{results.length !== 1 ? "s" : ""}</span>
              {nextChunkIn !== null && (
                <span className="font-mono">~{nextChunkIn}s chunks</span>
              )}
            </div>

            {error && (
              <p className="text-xs text-destructive">{error}</p>
            )}

            {/* Stop button */}
            <Button
              variant="destructive"
              size="sm"
              className="w-full gap-2"
              onClick={stopSession}
            >
              <Square className="h-3.5 w-3.5" />
              Stop Session
            </Button>
          </div>
        )}

        {/* ── Summary state ───────────────────────────────────────────── */}
        {showSummary && summary && (
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div
                className="h-10 w-10 rounded-full flex items-center justify-center shrink-0"
                style={{
                  backgroundColor: `${EMOTION_COLOR[summary.dominant] ?? "#94a3b8"}20`,
                  border: `2px solid ${EMOTION_COLOR[summary.dominant] ?? "#94a3b8"}50`,
                }}
              >
                <BarChart2
                  className="h-5 w-5"
                  style={{ color: EMOTION_COLOR[summary.dominant] ?? "#94a3b8" }}
                />
              </div>
              <div className="flex-1">
                <p className="text-sm font-semibold capitalize">
                  Dominant: {EMOTION_LABEL[summary.dominant] ?? summary.dominant}
                </p>
                <p className="text-xs text-muted-foreground">
                  {summary.count} sample{summary.count !== 1 ? "s" : ""} over {formatDuration(summary.totalDuration)}
                </p>
              </div>
              <Badge className={`text-xs border shrink-0 ${valenceBadgeClass(summary.avgValence)}`}>
                {valenceLabel(summary.avgValence)}
              </Badge>
            </div>

            {/* Valence trend */}
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              {summary.trend === "up" && (
                <>
                  <TrendingUp className="h-3.5 w-3.5 text-emerald-400" />
                  <span className="text-emerald-400">Valence improved over the session</span>
                </>
              )}
              {summary.trend === "down" && (
                <>
                  <TrendingDown className="h-3.5 w-3.5 text-rose-400" />
                  <span className="text-rose-400">Valence declined over the session</span>
                </>
              )}
              {summary.trend === "flat" && (
                <>
                  <Minus className="h-3.5 w-3.5" />
                  <span>Valence remained stable</span>
                </>
              )}
            </div>

            {/* Full timeline */}
            {results.length > 1 && (
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Emotion timeline</p>
                <MiniTimeline entries={results} />
              </div>
            )}

            {/* Start new session */}
            <Button
              variant="outline"
              size="sm"
              className="w-full gap-2 border-primary/30 hover:bg-primary/10"
              onClick={startSession}
            >
              <Mic className="h-3.5 w-3.5" />
              Start New Session
            </Button>
          </div>
        )}

      </CardContent>
    </Card>
  );
}
