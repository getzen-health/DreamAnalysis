import { useState, useEffect } from "react";
import { Link } from "wouter";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useDevice } from "@/hooks/use-device";
import { useAuth } from "@/hooks/use-auth";
import { useVoiceEmotion } from "@/hooks/use-voice-emotion";
import { SimulationModeBanner } from "@/components/simulation-mode-banner";
import { useToast } from "@/hooks/use-toast";
import { submitFeedback } from "@/lib/ml-api";
import { Music, Mic, MicOff } from "lucide-react";
import EmotionStateCard from "@/components/emotion-state-card";
import EmotionFlow, { type EmotionDataPoint } from "@/components/emotion-flow";

/* ---------- helpers ---------- */

const EMOTION_EMOJI: Record<string, string> = {
  happy:   "😊",
  sad:     "😔",
  angry:   "😠",
  fearful: "😨",
  relaxed: "😌",
  focused: "🎯",
  neutral: "😶",
  surprise:"😲",
};

function getSmartLabel(emotion: string, bands: Record<string, number>): string {
  const alpha = bands.alpha ?? 0;
  const beta  = bands.beta  ?? 0;
  const theta = bands.theta ?? 0;
  if (theta > 0.35 && beta < 0.22)  return "Drifting · Meditative";
  if (alpha > 0.35 && beta < 0.12)  return "Calm & Resting";
  if (alpha > 0.25 && beta > 0.22)  return "Focused Calm";
  if (beta  > 0.35 && alpha < 0.15) return "Alert & Active";
  if (alpha > 0.20 && theta > 0.25) return "Relaxed & Dreamy";
  const map: Record<string, string> = {
    relaxed: "Relaxed & At Ease",
    focused: "Mentally Engaged",
    happy:   "Positive & Uplifted",
    sad:     "Low & Withdrawn",
    angry:   "Tense & Activated",
    fearful: "Anxious & On-Edge",
  };
  return map[emotion] ?? "Neutral State";
}

function moodLine(valence: number): string {
  if (valence >  0.5) return "Very positive mood";
  if (valence >  0.2) return "Positive mood";
  if (valence > -0.2) return "Neutral mood";
  if (valence > -0.5) return "Slightly negative";
  return "Negative mood";
}

interface BarProps { label: string; value: number; color: string }
function Bar({ label, value, color }: BarProps) {
  const pct = Math.round(Math.max(0, Math.min(100, value)));
  const intensity =
    pct >= 70 ? "HIGH" : pct >= 40 ? "MED" : "LOW";
  const badgeColor =
    pct >= 70 ? "text-red-400 bg-red-500/10 border-red-500/30" :
    pct >= 40 ? "text-amber-400 bg-amber-500/10 border-amber-500/30" :
               "text-emerald-400 bg-emerald-500/10 border-emerald-500/30";
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded border ${badgeColor}`}>
          {intensity}
        </span>
      </div>
      <div className="h-2 rounded-full bg-muted/40 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}

/* ---------- component ---------- */

interface HistoryItem {
  emotion: string;
  label: string;
  time: string;
  confidence: number;
}

const CORRECT_OPTIONS = [
  { value: "happy",   emoji: "😊", label: "Happy" },
  { value: "relaxed", emoji: "😌", label: "Relaxed" },
  { value: "focus",   emoji: "🎯", label: "Focused" },
  { value: "neutral", emoji: "😶", label: "Neutral" },
  { value: "stress",  emoji: "😬", label: "Stress" },
  { value: "sad",     emoji: "😔", label: "Sad" },
  { value: "angry",   emoji: "😠", label: "Angry" },
  { value: "fear",    emoji: "😨", label: "Fear" },
];

export default function EmotionLab() {
  const { latestFrame, state: deviceState, reconnectCount } = useDevice();
  const { user } = useAuth();
  const { toast } = useToast();
  const voiceEmotion = useVoiceEmotion();
  const isStreaming = deviceState === "streaming";
  const analysis    = latestFrame?.analysis;
  const emotions    = analysis?.emotions;
  const bandPowers  = analysis?.band_powers ?? {};

  // Is the 30-second buffer still filling?
  const bufferedSec = emotions?.buffered_sec ?? 0;
  const windowSec   = emotions?.window_sec ?? 30;
  const emotionReady = !emotions || emotions.ready !== false || emotions.emotion != null;

  // Correction state
  const [showCorrect, setShowCorrect] = useState(false);
  const [correcting, setCorrecting] = useState(false);
  const [corrected, setCorrected] = useState<string | null>(null);

  // Reset correction UI when emotion changes
  useEffect(() => { setShowCorrect(false); setCorrected(null); }, [emotions?.emotion]);

  async function handleCorrect(value: string) {
    if (!user?.id || correcting) return;
    setCorrecting(true);
    try {
      // Save correction to Express DB for history
      await fetch(`/api/emotions/correct-latest/${user.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ userCorrectedEmotion: value }),
      });
      // Send to ML backend for personal model training (SGDClassifier + k-NN)
      const signals = latestFrame?.signals;
      submitFeedback(
        signals ?? [],
        emotion,
        value,
        user.id.toString(),
      ).catch(() => {}); // fire-and-forget, don't block UI
      setCorrected(value);
      setShowCorrect(false);
      toast({ title: "Thanks for the correction!", description: "This helps improve your personal model." });
    } catch {
      toast({ title: "Couldn't save correction", variant: "destructive" });
    } finally {
      setCorrecting(false);
    }
  }

  // Current values
  const emotion      = emotionReady ? (emotions?.emotion ?? "neutral") : "neutral";
  const confidence   = emotions?.confidence ?? 0;
  const stress       = (emotions?.stress_index     ?? 0) * 100;
  const focus        = (emotions?.focus_index      ?? 0) * 100;
  const relaxation   = (emotions?.relaxation_index ?? 0) * 100;
  const valence      = emotions?.valence ?? 0;

  const label   = getSmartLabel(emotion, bandPowers);
  const emoji   = EMOTION_EMOJI[emotion] ?? "🧠";

  // Confidence ring color
  const confColor =
    confidence >= 0.40 ? "hsl(152, 60%, 45%)" :
    confidence >= 0.30 ? "hsl(38, 85%, 55%)"  :
                         "hsl(220, 12%, 50%)";

  // History — last 5 distinct emotion changes (updated every 30s window)
  const [history, setHistory] = useState<HistoryItem[]>([]);

  useEffect(() => {
    if (!emotions?.ready || !emotions?.emotion) return;
    const now = new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    const newLabel = getSmartLabel(emotions.emotion, emotions.band_powers ?? {});
    setHistory((prev) => {
      const last = prev[prev.length - 1];
      if (last?.label === newLabel && last?.emotion === emotions.emotion) return prev;
      return [
        ...prev.slice(-4),
        { emotion: emotions.emotion!, label: newLabel, time: now, confidence: emotions.confidence },
      ];
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [emotions?.emotion, emotions?.ready]);

  // EmotionFlow readings — last 10 data points for the chart
  const [emotionReadings, setEmotionReadings] = useState<EmotionDataPoint[]>([]);

  useEffect(() => {
    if (!emotions?.ready || !emotions?.emotion) return;
    const now = new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    const point: EmotionDataPoint = {
      time: now,
      valence: emotions.valence ?? 0,
      arousal: emotions.arousal ?? 0,
      stress: emotions.stress_index ?? 0,
      label: emotions.emotion,
    };
    setEmotionReadings((prev) => [...prev.slice(-9), point]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [emotions?.emotion, emotions?.ready]);

  return (
    <div className="max-w-lg mx-auto px-4 py-8 space-y-4">
      {isStreaming && reconnectCount > 0 && (
        <div className="rounded-md bg-amber-500/10 border border-amber-500/40 px-4 py-2 text-sm font-medium text-amber-400">
          Reconnecting to EEG stream… (attempt {reconnectCount})
        </div>
      )}
      <SimulationModeBanner />

      {/* Voice emotion fallback — shown when EEG not streaming */}
      {deviceState !== "streaming" && (
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-amber-400">Voice Emotion Analysis</p>
              <p className="text-xs text-muted-foreground mt-0.5">
                No EEG headband connected — detect emotion via microphone
              </p>
            </div>
            <Button
              size="sm"
              variant={voiceEmotion.isRecording ? "destructive" : "outline"}
              onClick={voiceEmotion.isRecording ? voiceEmotion.stopRecording : voiceEmotion.startRecording}
              disabled={voiceEmotion.isAnalyzing}
              className="gap-2"
            >
              {voiceEmotion.isRecording ? (
                <><MicOff className="w-4 h-4" /> Stop &amp; Analyze</>
              ) : voiceEmotion.isAnalyzing ? (
                <><Mic className="w-4 h-4 animate-pulse" /> Analyzing…</>
              ) : (
                <><Mic className="w-4 h-4" /> Detect Emotion</>
              )}
            </Button>
          </div>

          {voiceEmotion.lastResult && (
            <div className="grid grid-cols-3 gap-2 text-center text-xs">
              <div className="rounded bg-background/50 p-2">
                <div className="font-semibold capitalize">{voiceEmotion.lastResult.emotion}</div>
                <div className="text-muted-foreground">Emotion</div>
              </div>
              <div className="rounded bg-background/50 p-2">
                <div className="font-semibold">
                  {voiceEmotion.lastResult.valence >= 0 ? "+" : ""}
                  {voiceEmotion.lastResult.valence.toFixed(2)}
                </div>
                <div className="text-muted-foreground">Valence</div>
              </div>
              <div className="rounded bg-background/50 p-2">
                <div className="font-semibold">
                  {Math.round(voiceEmotion.lastResult.confidence * 100)}%
                </div>
                <div className="text-muted-foreground">Confidence</div>
              </div>
            </div>
          )}

          {voiceEmotion.error && (
            <p className="text-xs text-destructive">{voiceEmotion.error}</p>
          )}
        </div>
      )}

      {/* ── Card 1: Right now ─────────────────────────────────────────────── */}
      <Card className="p-5">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-4">
          Right now
        </p>

        {!isStreaming ? (
          /* Voice-first mode */
          <div className="flex flex-col items-center gap-3 py-6 text-center">
            <div className="text-5xl">🎙️</div>
            <p className="text-sm font-medium">Voice mode is ready</p>
            <p className="text-xs text-muted-foreground max-w-[260px]">
              Run a microphone check-in above for emotion detection now. EEG is optional if you want continuous live readings.
            </p>
          </div>
        ) : !emotionReady ? (
          /* Buffering */
          <div className="flex flex-col items-center gap-3 py-6 text-center">
            <div className="text-5xl animate-pulse">🧠</div>
            <p className="text-sm font-medium">Calibrating…</p>
            <div className="w-40 h-1.5 rounded-full bg-muted overflow-hidden">
              <div
                className="h-full bg-primary transition-all duration-1000"
                style={{ width: `${Math.round((bufferedSec / windowSec) * 100)}%` }}
              />
            </div>
            <p className="text-xs text-muted-foreground">
              {bufferedSec}s of {windowSec}s collected
            </p>
          </div>
        ) : (
          /* Live emotion */
          <div className="space-y-5">
            {/* Emotion display */}
            <div className="flex items-center gap-4">
              <div
                className="w-16 h-16 rounded-2xl flex items-center justify-center text-3xl shrink-0"
                style={{ background: `${confColor}18`, border: `2px solid ${confColor}50` }}
              >
                {emoji}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-lg font-semibold leading-tight">{label}</p>
                <p className="text-xs text-muted-foreground mt-0.5">{moodLine(valence)}</p>
                <div className="flex items-center gap-3 mt-1">
                  <p className="text-[10px] text-muted-foreground/60">
                    {Math.round(confidence * 100)}% confidence
                  </p>
                  <Link
                    href={`/biofeedback?tab=music&mood=${valence < 0 || stress > 0.4 ? "calm" : "focus"}`}
                    className="flex items-center gap-1 text-[10px] text-violet-400 hover:text-violet-300 transition-colors font-medium"
                  >
                    <Music className="h-3 w-3" />
                    Full music session
                  </Link>
                </div>
              </div>
            </div>

            {/* ── Label correction ────────────────────────────────────────── */}
            {corrected ? (
              <p className="text-xs text-emerald-400">
                ✓ Corrected to <span className="font-medium capitalize">{corrected}</span> — model will learn from this
              </p>
            ) : showCorrect ? (
              <div className="space-y-2">
                <p className="text-xs text-muted-foreground">What were you actually feeling?</p>
                <div className="flex flex-wrap gap-1.5">
                  {CORRECT_OPTIONS.map(opt => (
                    <Button
                      key={opt.value}
                      size="sm"
                      variant="outline"
                      className="h-7 text-xs px-2 gap-1"
                      disabled={correcting}
                      onClick={() => handleCorrect(opt.value)}
                    >
                      {opt.emoji} {opt.label}
                    </Button>
                  ))}
                </div>
                <button onClick={() => setShowCorrect(false)} className="text-[10px] text-muted-foreground/60 hover:text-muted-foreground">
                  Cancel
                </button>
              </div>
            ) : (
              <button
                onClick={() => setShowCorrect(true)}
                className="text-[10px] text-muted-foreground/50 hover:text-muted-foreground transition-colors"
              >
                Not quite right? Correct it →
              </button>
            )}

            {/* Bars */}
            <div className="space-y-3">
              <Bar label="Stress"      value={stress}      color="hsl(0,72%,55%)" />
              <Bar label="Focus"       value={focus}       color="hsl(152,60%,48%)" />
              <Bar label="Relaxation"  value={relaxation}  color="hsl(217,91%,60%)" />
            </div>
            <p className="text-[10px] text-muted-foreground/40">
              Emotion indices computed from 30s EEG window · updates every 30s
            </p>

            {/* Creativity Score */}
            {analysis?.creativity?.creativity_score !== undefined && (
              <div className="flex items-center gap-2 mt-2">
                <span className="text-xs text-muted-foreground">Creativity</span>
                <Badge variant="outline" className="text-xs">
                  {Math.round((analysis.creativity.creativity_score ?? 0) * 100)}%
                </Badge>
              </div>
            )}

            {/* All Emotion Probabilities */}
            {emotions?.probabilities && Object.keys(emotions.probabilities).length > 0 && (
              <div className="space-y-2 mt-4">
                <h4 className="text-sm font-medium text-muted-foreground">Emotion Probabilities</h4>
                {Object.entries(emotions.probabilities).map(([emo, prob]) => (
                  <div key={emo} className="flex items-center gap-2">
                    <span className="text-xs w-16 capitalize">{emo}</span>
                    <Progress value={Math.round((prob as number) * 100)} className="flex-1 h-2" />
                    <span className="text-xs w-8 text-right">{Math.round((prob as number) * 100)}%</span>
                  </div>
                ))}
              </div>
            )}

            {/* Emotional State — human-readable card */}
            {emotions && (
              <EmotionStateCard
                emotion={emotion}
                valence={emotions.valence ?? 0}
                arousal={emotions.arousal ?? 0}
                stressIndex={emotions.stress_index}
                focusIndex={emotions.focus_index}
                confidence={emotions.confidence}
                source="eeg"
              />
            )}
          </div>
        )}
      </Card>

      {/* ── EmotionFlow chart ────────────────────────────────────────────── */}
      {emotionReadings.length > 0 && (
        <EmotionFlow data={emotionReadings} height={160} />
      )}

      {/* ── Card 2: Today's emotions ─────────────────────────────────────── */}
      <Card className="p-5">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-3">
          Today's emotions
        </p>

        {history.length === 0 ? (
          <p className="text-xs text-muted-foreground py-4 text-center">
            {isStreaming
              ? "Emotion readings will appear here as they come in."
              : "Start a session to track your emotions today."}
          </p>
        ) : (
          <div className="space-y-2">
            {[...history].reverse().map((item, i) => (
              <div key={i} className="flex items-center gap-3 py-1.5 border-b border-border/20 last:border-0">
                <span className="text-xl shrink-0">{EMOTION_EMOJI[item.emotion] ?? "🧠"}</span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium leading-tight">{item.label}</p>
                  <p className="text-xs text-muted-foreground">{item.time}</p>
                </div>
                <span className="text-[10px] text-muted-foreground/60 shrink-0">
                  {Math.round(item.confidence * 100)}%
                </span>
              </div>
            ))}
          </div>
        )}
      </Card>

    </div>
  );
}
