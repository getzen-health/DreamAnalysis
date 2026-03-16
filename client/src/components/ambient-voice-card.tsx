/**
 * AmbientVoiceCard — passive VAD-driven background voice emotion monitor.
 *
 * Uses useAmbientVoice to:
 *   - Monitor mic energy continuously via AnalyserNode (no recording)
 *   - Start a MediaRecorder only when speech is detected
 *   - Send the speech chunk to /voice-watch/analyze after silence
 *   - Apply a 10-second cooldown between analyses
 *
 * UI:
 *   - Idle: "Start Ambient Mode" button
 *   - Active: pulsing energy meter, status text, emotion badge, stop button
 */

import { useAmbientVoice } from "@/hooks/use-ambient-voice";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Ear, EarOff, Mic, Radio } from "lucide-react";

// ── helpers ───────────────────────────────────────────────────────────────────

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

// ── energy meter ──────────────────────────────────────────────────────────────

function EnergyMeter({
  level,
  isSpeech,
  isAnalyzing,
}: {
  level: number;
  isSpeech: boolean;
  isAnalyzing: boolean;
}) {
  // 20 bars, each lights up proportionally to the energy level
  const BAR_COUNT = 20;
  const activeBars = Math.round(level * BAR_COUNT);

  return (
    <div className="flex items-end gap-0.5 h-8" aria-label={`Audio level ${Math.round(level * 100)}%`}>
      {Array.from({ length: BAR_COUNT }).map((_, i) => {
        const isActive = i < activeBars;
        const baseHeight = 4 + Math.round((i / (BAR_COUNT - 1)) * 24); // 4–28 px
        const color = isAnalyzing
          ? "#a78bfa"
          : isSpeech
          ? "#34d399"
          : "#475569";

        return (
          <div
            key={i}
            className="flex-1 rounded-sm transition-all duration-75"
            style={{
              height: `${baseHeight}px`,
              backgroundColor: isActive ? color : "#1e293b",
              opacity: isActive ? 1 : 0.35,
            }}
          />
        );
      })}
    </div>
  );
}

// ── status line ───────────────────────────────────────────────────────────────

function statusText(
  isSpeechDetected: boolean,
  isAnalyzing: boolean,
): string {
  if (isAnalyzing) return "Analyzing speech...";
  if (isSpeechDetected) return "Speech detected";
  return "Listening for speech...";
}

// ── component ─────────────────────────────────────────────────────────────────

export function AmbientVoiceCard() {
  const {
    start,
    stop,
    isListening,
    isSpeechDetected,
    isAnalyzing,
    currentEmotion,
    energyLevel,
    speechCount,
    results,
    error,
  } = useAmbientVoice();

  const latestResult = results.length > 0 ? results[results.length - 1] : null;
  const emotionColor =
    EMOTION_COLOR[currentEmotion ?? "neutral"] ?? "#94a3b8";

  return (
    <Card className="bg-card rounded-xl border border-border/50 shadow-sm">
      <CardContent className="p-4 space-y-3">

        {/* ── Header ──────────────────────────────────────────────────────── */}
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-semibold">Ambient Mode</p>
            <p className="text-xs text-muted-foreground">
              {isListening
                ? statusText(isSpeechDetected, isAnalyzing)
                : "Passive voice emotion detection"}
            </p>
          </div>

          {/* Active indicator */}
          {isListening && (
            <div className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-xs text-emerald-400 font-medium">Live</span>
            </div>
          )}
        </div>

        {/* ── Idle state ─────────────────────────────────────────────────── */}
        {!isListening && (
          <div className="flex flex-col items-center gap-3 py-3">
            <Button
              size="lg"
              variant="outline"
              className="h-14 w-14 rounded-full border-2 border-primary/40 hover:border-primary hover:bg-primary/10 transition-all"
              onClick={start}
              aria-label="Start ambient mode"
            >
              <Ear className="h-6 w-6 text-primary" />
            </Button>
            <p className="text-xs text-muted-foreground text-center max-w-[220px]">
              Mic monitors passively — only records when you speak
            </p>
            {error && (
              <p className="text-xs text-destructive text-center">{error}</p>
            )}
          </div>
        )}

        {/* ── Active state ───────────────────────────────────────────────── */}
        {isListening && (
          <div className="space-y-3">

            {/* Energy meter */}
            <EnergyMeter
              level={energyLevel}
              isSpeech={isSpeechDetected}
              isAnalyzing={isAnalyzing}
            />

            {/* Emotion badge + spinner row */}
            <div className="flex items-center gap-2 min-h-[28px]">
              {isAnalyzing ? (
                <>
                  <div className="h-4 w-4 rounded-full border-2 border-violet-400 border-t-transparent animate-spin" />
                  <span className="text-xs text-violet-400">Analyzing…</span>
                </>
              ) : currentEmotion ? (
                <>
                  <div
                    className="h-2.5 w-2.5 rounded-full shrink-0"
                    style={{ backgroundColor: emotionColor }}
                  />
                  <Badge
                    className="text-xs border capitalize"
                    style={{
                      backgroundColor: `${emotionColor}20`,
                      color: emotionColor,
                      borderColor: `${emotionColor}40`,
                    }}
                  >
                    {EMOTION_LABEL[currentEmotion] ?? currentEmotion}
                  </Badge>
                  {latestResult && (
                    <span className="text-xs text-muted-foreground ml-auto">
                      {Math.round(latestResult.confidence * 100)}% conf
                    </span>
                  )}
                </>
              ) : (
                <span className="text-xs text-muted-foreground">
                  No emotion detected yet
                </span>
              )}
            </div>

            {/* Valence / arousal quick stats from latest result */}
            {latestResult && !isAnalyzing && (
              <div className="flex gap-3 text-xs text-muted-foreground">
                <span>
                  Valence{" "}
                  <span className="font-mono text-foreground">
                    {latestResult.valence >= 0 ? "+" : ""}
                    {latestResult.valence.toFixed(2)}
                  </span>
                </span>
                <span>
                  Arousal{" "}
                  <span className="font-mono text-foreground">
                    {Math.round(latestResult.arousal * 100)}%
                  </span>
                </span>
                {latestResult.stress_index != null && (
                  <span>
                    Stress{" "}
                    <span className="font-mono text-foreground">
                      {Math.round(latestResult.stress_index * 100)}%
                    </span>
                  </span>
                )}
              </div>
            )}

            {/* Speech count + recording indicator */}
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span className="flex items-center gap-1">
                <Radio className="h-3 w-3" />
                {speechCount} segment{speechCount !== 1 ? "s" : ""} analysed
              </span>
              {isSpeechDetected && !isAnalyzing && (
                <span className="flex items-center gap-1 text-emerald-400">
                  <Mic className="h-3 w-3" />
                  Recording
                </span>
              )}
            </div>

            {error && (
              <p className="text-xs text-destructive">{error}</p>
            )}

            {/* Stop button */}
            <Button
              variant="outline"
              size="sm"
              className="w-full gap-2 border-rose-500/30 text-rose-400 hover:bg-rose-500/10 hover:border-rose-500/60"
              onClick={stop}
            >
              <EarOff className="h-3.5 w-3.5" />
              Stop Ambient Mode
            </Button>
          </div>
        )}

      </CardContent>
    </Card>
  );
}
