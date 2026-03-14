import { useState, useEffect, useMemo, useRef } from "react";
import { Link } from "wouter";
import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useDevice } from "@/hooks/use-device";
import { useAuth } from "@/hooks/use-auth";
import { useVoiceEmotion } from "@/hooks/use-voice-emotion";
import { SimulationModeBanner } from "@/components/simulation-mode-banner";
import { VoiceCheckinCard } from "@/components/voice-checkin-card";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, resolveUrl } from "@/lib/queryClient";
import { submitFeedback } from "@/lib/ml-api";
import { getParticipantId } from "@/lib/participant";
import {
  Music,
  Mic,
  MicOff,
  TrendingUp,
  Clock,
  Heart,
  Sparkles,
  ChevronRight,
  Brain,
  RefreshCw,
} from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import EmotionStateCard from "@/components/emotion-state-card";
import EmotionFlow, { type EmotionDataPoint } from "@/components/emotion-flow";

/* ---------- constants & helpers ---------- */

const EMOTION_COLORS: Record<string, string> = {
  happy: "#34d399",
  sad: "#60a5fa",
  angry: "#f87171",
  fearful: "#fbbf24",
  fear: "#fbbf24",
  relaxed: "#a78bfa",
  focused: "#2dd4bf",
  neutral: "#94a3b8",
  surprise: "#f472b6",
};

const EMOTION_LABELS: Record<string, string> = {
  happy: "Happy",
  sad: "Sad",
  angry: "Angry",
  fearful: "Fearful",
  fear: "Fearful",
  relaxed: "Relaxed",
  focused: "Focused",
  neutral: "Neutral",
  surprise: "Surprised",
};

const QUICK_MOODS = [
  { value: "happy", label: "Happy", color: "#34d399", bgClass: "bg-emerald-500/10 border-emerald-500/30 hover:bg-emerald-500/20" },
  { value: "sad", label: "Sad", color: "#60a5fa", bgClass: "bg-blue-500/10 border-blue-500/30 hover:bg-blue-500/20" },
  { value: "angry", label: "Angry", color: "#f87171", bgClass: "bg-red-500/10 border-red-500/30 hover:bg-red-500/20" },
  { value: "fearful", label: "Fearful", color: "#fbbf24", bgClass: "bg-amber-500/10 border-amber-500/30 hover:bg-amber-500/20" },
  { value: "surprise", label: "Surprised", color: "#f472b6", bgClass: "bg-pink-500/10 border-pink-500/30 hover:bg-pink-500/20" },
  { value: "neutral", label: "Neutral", color: "#94a3b8", bgClass: "bg-slate-500/10 border-slate-500/30 hover:bg-slate-500/20" },
];

const VALENCE_MAP: Record<string, number> = {
  happy: 0.7,
  sad: -0.6,
  angry: -0.5,
  fearful: -0.4,
  surprise: 0.2,
  neutral: 0.0,
};

const AROUSAL_MAP: Record<string, number> = {
  happy: 0.6,
  sad: 0.2,
  angry: 0.8,
  fearful: 0.7,
  surprise: 0.7,
  neutral: 0.3,
};

function getSmartLabel(emotion: string, bands: Record<string, number>): string {
  const alpha = bands.alpha ?? 0;
  const beta = bands.beta ?? 0;
  const theta = bands.theta ?? 0;
  if (theta > 0.35 && beta < 0.22) return "Drifting \u00b7 Meditative";
  if (alpha > 0.35 && beta < 0.12) return "Calm & Resting";
  if (alpha > 0.25 && beta > 0.22) return "Focused Calm";
  if (beta > 0.35 && alpha < 0.15) return "Alert & Active";
  if (alpha > 0.20 && theta > 0.25) return "Relaxed & Dreamy";
  const map: Record<string, string> = {
    relaxed: "Relaxed & At Ease",
    focused: "Mentally Engaged",
    happy: "Positive & Uplifted",
    sad: "Low & Withdrawn",
    angry: "Tense & Activated",
    fearful: "Anxious & On-Edge",
  };
  return map[emotion] ?? "Neutral State";
}

function moodLine(valence: number): string {
  if (valence > 0.5) return "Very positive mood";
  if (valence > 0.2) return "Positive mood";
  if (valence > -0.2) return "Neutral mood";
  if (valence > -0.5) return "Slightly negative";
  return "Negative mood";
}

interface BarProps {
  label: string;
  value: number;
  color: string;
}
function Bar({ label, value, color }: BarProps) {
  const pct = Math.round(Math.max(0, Math.min(100, value)));
  const intensity = pct >= 70 ? "HIGH" : pct >= 40 ? "MED" : "LOW";
  const badgeColor =
    pct >= 70
      ? "text-red-400 bg-red-500/10 border-red-500/30"
      : pct >= 40
      ? "text-amber-400 bg-amber-500/10 border-amber-500/30"
      : "text-emerald-400 bg-emerald-500/10 border-emerald-500/30";
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span
          className={`text-[10px] font-semibold px-1.5 py-0.5 rounded border ${badgeColor}`}
        >
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

/* ---------- types ---------- */

interface HistoryItem {
  emotion: string;
  label: string;
  time: string;
  confidence: number;
}

interface EmotionReadingRow {
  id: string;
  userId: string;
  stress: number;
  happiness: number;
  focus: number;
  energy: number;
  dominantEmotion: string;
  valence: number | null;
  arousal: number | null;
  timestamp: string;
}

const CORRECT_OPTIONS = [
  { value: "happy", label: "Happy" },
  { value: "relaxed", label: "Relaxed" },
  { value: "focus", label: "Focused" },
  { value: "neutral", label: "Neutral" },
  { value: "stress", label: "Stress" },
  { value: "sad", label: "Sad" },
  { value: "angry", label: "Angry" },
  { value: "fear", label: "Fear" },
];

/* ---------- mini-chart tooltip ---------- */

interface MoodTooltipProps {
  active?: boolean;
  payload?: Array<{ value: number }>;
  label?: string;
}

function MoodTooltip({ active, payload, label }: MoodTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const val = payload[0].value;
  return (
    <div className="rounded-lg bg-muted border border-border px-3 py-2 text-xs shadow-lg">
      <p className="text-muted-foreground">{label}</p>
      <p className="text-foreground font-medium mt-0.5">
        Valence: {val >= 0 ? "+" : ""}
        {val.toFixed(2)}
      </p>
    </div>
  );
}

/* ========== COMPONENT ========== */

export default function EmotionLab() {
  const { latestFrame, state: deviceState, reconnectCount } = useDevice();
  const { user } = useAuth();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const voiceEmotion = useVoiceEmotion();

  // Voice recording countdown timer (30s default duration)
  const VOICE_DURATION_SEC = 30;
  const [voiceCountdown, setVoiceCountdown] = useState(VOICE_DURATION_SEC);
  useEffect(() => {
    if (!voiceEmotion.isRecording) {
      setVoiceCountdown(VOICE_DURATION_SEC);
      return;
    }
    const interval = setInterval(() => {
      setVoiceCountdown((prev) => (prev > 0 ? prev - 1 : 0));
    }, 1000);
    return () => clearInterval(interval);
  }, [voiceEmotion.isRecording]);

  // Track last analysis timestamp for "Last synced" display
  const [lastAnalysisTime, setLastAnalysisTime] = useState<Date | null>(null);
  const [lastAnalysisAgo, setLastAnalysisAgo] = useState<string>("");
  const prevResultRef = useRef(voiceEmotion.lastResult);
  useEffect(() => {
    if (voiceEmotion.lastResult && voiceEmotion.lastResult !== prevResultRef.current) {
      setLastAnalysisTime(new Date());
    }
    prevResultRef.current = voiceEmotion.lastResult;
  }, [voiceEmotion.lastResult]);

  // Update relative time string every 30s
  useEffect(() => {
    function formatAgo(date: Date): string {
      const diffSec = Math.floor((Date.now() - date.getTime()) / 1000);
      if (diffSec < 10) return "just now";
      if (diffSec < 60) return `${diffSec}s ago`;
      const diffMin = Math.floor(diffSec / 60);
      if (diffMin < 60) return `${diffMin}m ago`;
      const diffHr = Math.floor(diffMin / 60);
      return `${diffHr}h ago`;
    }
    if (!lastAnalysisTime) {
      setLastAnalysisAgo("");
      return;
    }
    setLastAnalysisAgo(formatAgo(lastAnalysisTime));
    const interval = setInterval(() => {
      setLastAnalysisAgo(formatAgo(lastAnalysisTime));
    }, 30000);
    return () => clearInterval(interval);
  }, [lastAnalysisTime]);

  const isStreaming = deviceState === "streaming";
  const analysis = latestFrame?.analysis;
  const emotions = analysis?.emotions;
  const bandPowers = analysis?.band_powers ?? {};

  const participantId = user?.id?.toString() ?? getParticipantId();

  // Is the 30-second buffer still filling?
  const bufferedSec = emotions?.buffered_sec ?? 0;
  const windowSec = emotions?.window_sec ?? 30;
  const emotionReady =
    !emotions || emotions.ready !== false || emotions.emotion != null;

  // Correction state
  const [showCorrect, setShowCorrect] = useState(false);
  const [correcting, setCorrecting] = useState(false);
  const [corrected, setCorrected] = useState<string | null>(null);

  // Reset correction UI when emotion changes
  useEffect(() => {
    setShowCorrect(false);
    setCorrected(null);
  }, [emotions?.emotion]);

  async function handleCorrect(value: string) {
    if (!user?.id || correcting) return;
    setCorrecting(true);
    try {
      await fetch(resolveUrl(`/api/emotions/correct-latest/${user.id}`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ userCorrectedEmotion: value }),
      });
      const signals = latestFrame?.signals;
      submitFeedback(signals ?? [], emotion, value, user.id.toString()).catch(
        () => {}
      );
      setCorrected(value);
      setShowCorrect(false);
      toast({
        title: "Thanks for the correction!",
        description: "This helps improve your personal model.",
      });
    } catch {
      toast({ title: "Couldn't save correction", variant: "destructive" });
    } finally {
      setCorrecting(false);
    }
  }

  // Current values
  const emotion = emotionReady ? (emotions?.emotion ?? "neutral") : "neutral";
  const confidence = emotions?.confidence ?? 0;
  const stress = (emotions?.stress_index ?? 0) * 100;
  const focus = (emotions?.focus_index ?? 0) * 100;
  const relaxation = (emotions?.relaxation_index ?? 0) * 100;
  const valence = emotions?.valence ?? 0;

  const label = getSmartLabel(emotion, bandPowers);

  // Confidence ring color
  const confColor =
    confidence >= 0.4
      ? "hsl(152, 60%, 45%)"
      : confidence >= 0.3
      ? "hsl(38, 85%, 55%)"
      : "hsl(220, 12%, 50%)";

  // History -- last 5 distinct emotion changes (updated every 30s window)
  const [history, setHistory] = useState<HistoryItem[]>([]);

  useEffect(() => {
    if (!emotions?.ready || !emotions?.emotion) return;
    const now = new Date().toLocaleTimeString([], {
      hour: "numeric",
      minute: "2-digit",
    });
    const newLabel = getSmartLabel(
      emotions.emotion,
      emotions.band_powers ?? {}
    );
    setHistory((prev) => {
      const last = prev[prev.length - 1];
      if (last?.label === newLabel && last?.emotion === emotions.emotion)
        return prev;
      return [
        ...prev.slice(-4),
        {
          emotion: emotions.emotion!,
          label: newLabel,
          time: now,
          confidence: emotions.confidence,
        },
      ];
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [emotions?.emotion, emotions?.ready]);

  // EmotionFlow readings -- last 10 data points for the chart
  const [emotionReadings, setEmotionReadings] = useState<EmotionDataPoint[]>(
    []
  );

  useEffect(() => {
    if (!emotions?.ready || !emotions?.emotion) return;
    const now = new Date().toLocaleTimeString([], {
      hour: "numeric",
      minute: "2-digit",
    });
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

  /* ---------- Mood trend data (7-day) ---------- */
  const { data: moodTrendRaw } = useQuery<EmotionReadingRow[]>({
    queryKey: [`/api/brain/history/${participantId}?days=7`],
    enabled: !!participantId,
    staleTime: 5 * 60 * 1000, // 5 min
    retry: false,
  });

  const moodTrendData = useMemo(() => {
    if (!moodTrendRaw || moodTrendRaw.length === 0) return [];
    // Bucket by day
    const buckets: Record<string, { sum: number; count: number }> = {};
    for (const r of moodTrendRaw) {
      const d = new Date(r.timestamp);
      const key = d.toLocaleDateString(undefined, {
        weekday: "short",
      });
      if (!buckets[key]) buckets[key] = { sum: 0, count: 0 };
      buckets[key].sum += r.valence ?? 0;
      buckets[key].count += 1;
    }
    return Object.entries(buckets).map(([day, b]) => ({
      day,
      valence: Math.round((b.sum / b.count) * 100) / 100,
    }));
  }, [moodTrendRaw]);

  /* ---------- Recent emotion readings ---------- */
  const { data: recentReadingsRaw } = useQuery<EmotionReadingRow[]>({
    queryKey: [`/api/brain/history/${participantId}?days=1`],
    enabled: !!participantId,
    staleTime: 2 * 60 * 1000,
    retry: false,
  });

  const recentReadings = useMemo(() => {
    if (!recentReadingsRaw || recentReadingsRaw.length === 0) return [];
    // Take last 7 readings, newest first
    return [...recentReadingsRaw].reverse().slice(0, 7);
  }, [recentReadingsRaw]);

  /* ---------- Quick mood log mutation ---------- */
  const quickMoodMutation = useMutation({
    mutationFn: async (moodValue: string) => {
      const reading = {
        userId: participantId,
        sessionId: `quick-${Date.now()}`,
        stress: moodValue === "angry" || moodValue === "fearful" ? 0.6 : 0.1,
        happiness: moodValue === "happy" ? 0.8 : 0.0,
        focus: 0.5,
        energy: AROUSAL_MAP[moodValue] ?? 0.5,
        dominantEmotion: moodValue,
        valence: VALENCE_MAP[moodValue] ?? 0.0,
        arousal: AROUSAL_MAP[moodValue] ?? 0.5,
      };
      await apiRequest("POST", "/api/emotion-readings/batch", {
        readings: [reading],
      });
      return moodValue;
    },
    onSuccess: (moodValue) => {
      toast({
        title: `Mood logged: ${EMOTION_LABELS[moodValue] ?? moodValue}`,
        description: "Your check-in has been saved.",
      });
      queryClient.invalidateQueries({
        queryKey: [`/api/brain/history/${participantId}?days=1`],
      });
      queryClient.invalidateQueries({
        queryKey: [`/api/brain/history/${participantId}?days=7`],
      });
      queryClient.invalidateQueries({ queryKey: ["emotions"] });
    },
    onError: () => {
      toast({ title: "Failed to log mood", variant: "destructive" });
    },
  });

  const [selectedQuickMood, setSelectedQuickMood] = useState<string | null>(
    null
  );

  function handleQuickMood(value: string) {
    setSelectedQuickMood(value);
    quickMoodMutation.mutate(value);
    // Reset selection after a short delay
    setTimeout(() => setSelectedQuickMood(null), 2000);
  }

  return (
    <div className="max-w-lg mx-auto px-4 py-4 pb-24 space-y-4">
      {isStreaming && reconnectCount > 0 && (
        <div className="rounded-2xl bg-amber-500/10 border border-amber-500/40 px-4 py-2.5 text-[13px] font-medium text-amber-400">
          Reconnecting to EEG stream... (attempt {reconnectCount})
        </div>
      )}
      <SimulationModeBanner />

      {/* ── Hero header ────────────────────────────────────────────────── */}
      <div className="text-center space-y-1 pt-1">
        <h1 className="text-[22px] font-bold tracking-tight bg-gradient-to-r from-violet-400 via-purple-400 to-indigo-400 bg-clip-text text-transparent">
          Mood &amp; Emotions
        </h1>
        <p className="text-[13px] text-muted-foreground">
          Track how you feel, discover patterns
        </p>
        <p className="text-[11px] text-muted-foreground/50 mt-1">
          {lastAnalysisTime
            ? `Last check-in ${lastAnalysisAgo}`
            : "No check-in yet today"}
        </p>
      </div>

      {/* ── Quick Mood Log ─────────────────────────────────────────────── */}
      <div
        className="rounded-2xl p-4 space-y-4"
        style={{
          background: "hsl(270,20%,8%,0.7)",
          border: "1px solid hsl(270,20%,18%,0.5)",
          boxShadow: "0 1px 3px hsl(222,30%,3%,0.4)",
        }}
      >
        <div className="flex items-center gap-2">
          <Heart className="h-4 w-4 text-violet-400" />
          <p className="text-[13px] font-semibold">How are you feeling?</p>
        </div>
        <div className="grid grid-cols-3 gap-2" role="group" aria-label="Quick mood selection">
          {QUICK_MOODS.map((mood) => (
            <button
              key={mood.value}
              onClick={() => handleQuickMood(mood.value)}
              disabled={quickMoodMutation.isPending}
              aria-label={`Log mood: ${mood.label}`}
              aria-pressed={selectedQuickMood === mood.value}
              className={`
                flex flex-col items-center justify-center gap-1.5 rounded-2xl border
                px-2 py-3.5 min-h-[64px] transition-all duration-200 cursor-pointer
                ${mood.bgClass}
                ${selectedQuickMood === mood.value ? "ring-2 ring-violet-400 scale-[0.97]" : "active:scale-[0.97]"}
                disabled:opacity-50
              `}
            >
              <span
                className="w-3 h-3 rounded-full shrink-0"
                style={{ backgroundColor: mood.color }}
                aria-hidden="true"
              />
              <span className="text-[12px] font-semibold text-foreground/80">
                {mood.label}
              </span>
            </button>
          ))}
        </div>
        {quickMoodMutation.isSuccess && (
          <p className="text-xs text-emerald-400 text-center animate-in fade-in duration-300" aria-live="polite">
            Logged successfully
          </p>
        )}
      </div>

      {/* ── Voice check-in card ────────────────────────────────────────── */}
      <VoiceCheckinCard userId={participantId} />

      {/* ── Voice Emotion Analysis — centered large mic button ─────────── */}
      {deviceState !== "streaming" && (
        <div
          className="rounded-2xl p-5 space-y-5"
          style={{
            background: "hsl(38,25%,7%,0.7)",
            border: "1px solid hsl(38,30%,18%,0.5)",
            boxShadow: "0 1px 3px hsl(222,30%,3%,0.4)",
          }}
        >
          <div className="text-center">
            <p className="text-[13px] font-semibold text-amber-400 mb-0.5">
              Voice Emotion Analysis
            </p>
            <p
              className="text-[11px] text-muted-foreground"
              role="status"
              aria-live="polite"
            >
              {voiceEmotion.isRecording
                ? "Recording… speak naturally"
                : voiceEmotion.isAnalyzing
                ? "Analyzing your voice…"
                : "Tap to detect emotion from your voice"}
            </p>
          </div>

          {/* Large centered mic button */}
          <div className="flex flex-col items-center gap-3">
            <button
              onClick={
                voiceEmotion.isRecording
                  ? voiceEmotion.stopRecording
                  : voiceEmotion.startRecording
              }
              disabled={voiceEmotion.isAnalyzing}
              aria-label={
                voiceEmotion.isRecording
                  ? "Stop voice recording"
                  : voiceEmotion.isAnalyzing
                  ? "Analyzing voice, please wait"
                  : "Start voice recording"
              }
              aria-pressed={voiceEmotion.isRecording}
              className={`
                relative w-20 h-20 rounded-full flex items-center justify-center
                transition-all duration-200 active:scale-95 disabled:opacity-50
                ${voiceEmotion.isRecording
                  ? "bg-red-500 shadow-lg shadow-red-500/40"
                  : voiceEmotion.isAnalyzing
                  ? "bg-amber-500/80 shadow-lg shadow-amber-500/30"
                  : "bg-amber-500 shadow-lg shadow-amber-500/30 hover:bg-amber-400"
                }
              `}
            >
              {voiceEmotion.isRecording && (
                <span className="absolute inset-0 rounded-full bg-red-400/30 animate-ping" aria-hidden="true" />
              )}
              {voiceEmotion.isRecording ? (
                <MicOff className="w-9 h-9 text-white" aria-hidden="true" />
              ) : voiceEmotion.isAnalyzing ? (
                <Mic className="w-9 h-9 text-white animate-pulse" aria-hidden="true" />
              ) : (
                <Mic className="w-9 h-9 text-white" aria-hidden="true" />
              )}
            </button>
            {voiceEmotion.isRecording ? (
              <div className="flex flex-col items-center gap-1.5">
                <div className="flex items-center gap-1.5 text-[13px] font-semibold text-amber-400">
                  <span>{voiceCountdown}s</span>
                  <span className="text-[11px] font-normal text-muted-foreground">remaining</span>
                </div>
                <div className="w-32 h-1.5 rounded-full bg-muted/40 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-amber-400 transition-all duration-1000 ease-linear"
                    style={{ width: `${(voiceCountdown / VOICE_DURATION_SEC) * 100}%` }}
                  />
                </div>
                <p className="text-[11px] font-medium text-muted-foreground">
                  Tap to stop &amp; analyze
                </p>
              </div>
            ) : (
              <p className="text-[11px] font-medium text-muted-foreground">
                {voiceEmotion.isAnalyzing
                  ? "Please wait…"
                  : "Tap to record"}
              </p>
            )}
          </div>

          {voiceEmotion.lastResult && (
            <div className="grid grid-cols-3 gap-2 text-center" aria-live="assertive" aria-label="Voice emotion analysis result">
              <div className="rounded-xl bg-muted/20 p-2.5">
                <div className="text-[14px] font-bold capitalize text-amber-400">
                  {voiceEmotion.lastResult.emotion}
                </div>
                <div className="text-[10px] text-muted-foreground mt-0.5">Emotion</div>
              </div>
              <div className="rounded-xl bg-muted/20 p-2.5">
                <div className="text-[14px] font-bold text-violet-400">
                  {voiceEmotion.lastResult.valence >= 0 ? "+" : ""}
                  {voiceEmotion.lastResult.valence.toFixed(2)}
                </div>
                <div className="text-[10px] text-muted-foreground mt-0.5">Valence</div>
              </div>
              <div className="rounded-xl bg-muted/20 p-2.5">
                <div className="text-[14px] font-bold text-emerald-400">
                  {Math.round(voiceEmotion.lastResult.confidence * 100)}%
                </div>
                <div className="text-[10px] text-muted-foreground mt-0.5">Confidence</div>
              </div>
            </div>
            )}

            {voiceEmotion.error && (
              <div className="flex items-center justify-center gap-2">
                <p className="text-xs text-destructive">{voiceEmotion.error}</p>
                <button
                  onClick={() => voiceEmotion.startRecording()}
                  className="inline-flex items-center gap-1 text-xs font-medium text-amber-400 hover:text-amber-300 transition-colors px-2 py-1 rounded-lg border border-amber-500/30 bg-amber-500/10 hover:bg-amber-500/20"
                >
                  <RefreshCw className="h-3 w-3" />
                  Retry
                </button>
              </div>
            )}
        </div>
      )}

      {/* ── Card 1: Right now (live EEG) ───────────────────────────────── */}
      <div
        className="rounded-2xl overflow-hidden"
        style={{
          background: "hsl(165,20%,7%,0.7)",
          border: "1px solid hsl(165,20%,17%,0.5)",
          boxShadow: "0 1px 3px hsl(222,30%,3%,0.4)",
        }}
      >
        <div className="p-5">
          <div className="flex items-center gap-2 mb-4">
            <Brain className="h-4 w-4 text-emerald-400" />
            <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground/60">
              Right now
            </p>
          </div>

          {!isStreaming ? (
            <div className="flex flex-col items-center gap-3 py-6 text-center">
              <div className="w-16 h-16 rounded-2xl bg-muted/20 border border-border/30 flex items-center justify-center">
                <Mic className="h-7 w-7 text-muted-foreground/50" />
              </div>
              <p className="text-sm font-medium">Voice mode is ready</p>
              <p className="text-xs text-muted-foreground max-w-[200px] sm:max-w-[260px]">
                Run a voice check-in above for emotion detection. Connect an EEG
                headband for continuous live readings.
              </p>
            </div>
          ) : !emotionReady ? (
            <div className="flex flex-col items-center gap-3 py-6 text-center">
              <div className="w-16 h-16 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center">
                <Brain className="h-7 w-7 text-primary animate-pulse" />
              </div>
              <p className="text-sm font-medium">Calibrating...</p>
              <div className="w-40 h-1.5 rounded-full bg-muted overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-1000"
                  style={{
                    width: `${Math.round((bufferedSec / windowSec) * 100)}%`,
                  }}
                />
              </div>
              <p className="text-xs text-muted-foreground">
                {bufferedSec}s of {windowSec}s collected
              </p>
            </div>
          ) : (
            <div className="space-y-5">
              {/* Emotion display */}
              <div className="flex items-center gap-4">
                <div
                  className="w-16 h-16 rounded-2xl flex items-center justify-center shrink-0"
                  style={{
                    background: `${confColor}18`,
                    border: `2px solid ${confColor}50`,
                  }}
                >
                  <span
                    className="w-6 h-6 rounded-full"
                    style={{
                      backgroundColor:
                        EMOTION_COLORS[emotion] ?? EMOTION_COLORS.neutral,
                    }}
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-lg font-semibold leading-tight">{label}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {moodLine(valence)}
                  </p>
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

              {/* Label correction */}
              {corrected ? (
                <p className="text-xs text-emerald-400">
                  Corrected to{" "}
                  <span className="font-medium capitalize">{corrected}</span> --
                  model will learn from this
                </p>
              ) : showCorrect ? (
                <div className="space-y-2">
                  <p className="text-xs text-muted-foreground">
                    What were you actually feeling?
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {CORRECT_OPTIONS.map((opt) => (
                      <Button
                        key={opt.value}
                        size="sm"
                        variant="outline"
                        className="h-9 text-xs px-3 gap-1 min-w-[44px]"
                        disabled={correcting}
                        onClick={() => handleCorrect(opt.value)}
                      >
                        {opt.label}
                      </Button>
                    ))}
                  </div>
                  <button
                    onClick={() => setShowCorrect(false)}
                    className="text-[10px] text-muted-foreground/60 hover:text-muted-foreground"
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setShowCorrect(true)}
                  className="text-[10px] text-muted-foreground/50 hover:text-muted-foreground transition-colors"
                >
                  Not quite right? Correct it
                </button>
              )}

              {/* Bars */}
              <div className="space-y-3">
                <Bar
                  label="Stress"
                  value={stress}
                  color="hsl(0,72%,55%)"
                />
                <Bar
                  label="Focus"
                  value={focus}
                  color="hsl(152,60%,48%)"
                />
                <Bar
                  label="Relaxation"
                  value={relaxation}
                  color="hsl(217,91%,60%)"
                />
              </div>
              <p className="text-[10px] text-muted-foreground/40">
                Emotion indices computed from 30s EEG window
              </p>

              {/* Creativity Score */}
              {analysis?.creativity?.creativity_score !== undefined && (
                <div className="flex items-center gap-2 mt-2">
                  <span className="text-xs text-muted-foreground">
                    Creativity
                  </span>
                  <Badge variant="outline" className="text-xs">
                    {Math.round(
                      (analysis.creativity.creativity_score ?? 0) * 100
                    )}
                    %
                  </Badge>
                </div>
              )}

              {/* All Emotion Probabilities */}
              {emotions?.probabilities &&
                Object.keys(emotions.probabilities).length > 0 && (
                  <div className="space-y-2 mt-4">
                    <h4 className="text-sm font-medium text-muted-foreground">
                      Emotion Probabilities
                    </h4>
                    {Object.entries(emotions.probabilities).map(
                      ([emo, prob]) => (
                        <div key={emo} className="flex items-center gap-2">
                          <span className="text-xs w-16 capitalize">{emo}</span>
                          <Progress
                            value={Math.round((prob as number) * 100)}
                            className="flex-1 h-2"
                          />
                          <span className="text-xs w-8 text-right">
                            {Math.round((prob as number) * 100)}%
                          </span>
                        </div>
                      )
                    )}
                  </div>
                )}

              {/* Emotional State -- human-readable card */}
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
        </div>
      </div>

      {/* ── EmotionFlow chart (live session) ──────────────────────────── */}
      {emotionReadings.length > 0 && (
        <EmotionFlow data={emotionReadings} height={160} />
      )}

      {/* ── Mood Trend (7-day) ─────────────────────────────────────────── */}
      <div
        className="rounded-2xl overflow-hidden"
        style={{
          background: "hsl(38,20%,7%,0.7)",
          border: "1px solid hsl(38,20%,17%,0.5)",
          boxShadow: "0 1px 3px hsl(222,30%,3%,0.4)",
        }}
      >
        <div className="p-5 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-amber-400" />
              <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground/60">
                Mood Trend
              </p>
            </div>
            <Badge
              variant="outline"
              className="text-[10px] border-border/40 text-muted-foreground"
            >
              7 days
            </Badge>
          </div>

          {moodTrendData.length === 0 ? (
            <div className="flex flex-col items-center gap-2 py-6 text-center">
              <TrendingUp className="h-8 w-8 text-muted-foreground/30" />
              <p className="text-xs text-muted-foreground">
                No mood data yet. Log your mood above to start tracking trends.
              </p>
            </div>
          ) : (
            <div className="h-[120px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={moodTrendData}
                  margin={{ top: 8, right: 8, left: -20, bottom: 0 }}
                >
                  <defs>
                    <linearGradient
                      id="moodGradient"
                      x1="0"
                      y1="0"
                      x2="0"
                      y2="1"
                    >
                      <stop
                        offset="5%"
                        stopColor="#f59e0b"
                        stopOpacity={0.6}
                      />
                      <stop
                        offset="95%"
                        stopColor="#f59e0b"
                        stopOpacity={0.05}
                      />
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="day"
                    tick={{ fill: "#9ca3af", fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    domain={[-1, 1]}
                    tick={{ fill: "#6b7280", fontSize: 9 }}
                    axisLine={false}
                    tickLine={false}
                    tickFormatter={(v: number) =>
                      v === 0 ? "0" : v > 0 ? `+${v}` : `${v}`
                    }
                  />
                  <Tooltip content={<MoodTooltip />} />
                  <Area
                    type="monotone"
                    dataKey="valence"
                    stroke="#f59e0b"
                    strokeWidth={2}
                    fill="url(#moodGradient)"
                    dot={{ fill: "#f59e0b", r: 3, strokeWidth: 0 }}
                    activeDot={{ r: 5, fill: "#f59e0b" }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      {/* ── Recent Emotions ────────────────────────────────────────────── */}
      <div
        className="rounded-2xl overflow-hidden"
        style={{
          background: "hsl(220,20%,7%,0.7)",
          border: "1px solid hsl(220,20%,17%,0.5)",
          boxShadow: "0 1px 3px hsl(222,30%,3%,0.4)",
        }}
      >
        <div className="p-5 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-blue-400" />
              <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground/60">
                Recent Emotions
              </p>
            </div>
            {recentReadings.length > 0 && (
              <Link
                href="/sessions"
                className="flex items-center gap-0.5 text-[10px] text-muted-foreground hover:text-foreground transition-colors"
              >
                View all <ChevronRight className="h-3 w-3" />
              </Link>
            )}
          </div>

          {recentReadings.length === 0 && history.length === 0 ? (
            <div className="flex flex-col items-center gap-2 py-6 text-center">
              <Sparkles className="h-8 w-8 text-muted-foreground/30" />
              <p className="text-xs text-muted-foreground">
                {isStreaming
                  ? "Emotion readings will appear here as they come in."
                  : "Log a mood or start a session to see your emotion history."}
              </p>
            </div>
          ) : (
            <div className="space-y-1">
              {/* Show persisted readings from DB if available */}
              {recentReadings.length > 0
                ? recentReadings.map((reading) => {
                    const emo = reading.dominantEmotion;
                    const dotColor =
                      EMOTION_COLORS[emo] ?? EMOTION_COLORS.neutral;
                    const ts = new Date(reading.timestamp);
                    const timeStr = ts.toLocaleTimeString([], {
                      hour: "numeric",
                      minute: "2-digit",
                    });
                    const dateStr = ts.toLocaleDateString([], {
                      month: "short",
                      day: "numeric",
                    });
                    return (
                      <div
                        key={reading.id}
                        className="flex items-center gap-3 py-2.5 border-b border-border/15 last:border-0"
                      >
                        <span
                          className="w-3 h-3 rounded-full shrink-0"
                          aria-hidden="true"
                          style={{ backgroundColor: dotColor }}
                        />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium leading-tight capitalize">
                            {EMOTION_LABELS[emo] ?? emo}
                          </p>
                          <p className="text-[10px] text-muted-foreground">
                            {dateStr} at {timeStr}
                          </p>
                        </div>
                        <div className="flex items-center gap-2 shrink-0">
                          {reading.valence != null && (
                            <Badge
                              variant="outline"
                              className={`text-[10px] border px-1.5 py-0 ${
                                reading.valence > 0.2
                                  ? "border-emerald-500/30 text-emerald-400 bg-emerald-500/10"
                                  : reading.valence < -0.2
                                  ? "border-rose-500/30 text-rose-400 bg-rose-500/10"
                                  : "border-border/40 text-muted-foreground"
                              }`}
                            >
                              {reading.valence >= 0 ? "+" : ""}
                              {reading.valence.toFixed(1)}
                            </Badge>
                          )}
                        </div>
                      </div>
                    );
                  })
                : /* Fall back to session history when no DB readings */
                  [...history].reverse().map((item, i) => {
                    const dotColor =
                      EMOTION_COLORS[item.emotion] ?? EMOTION_COLORS.neutral;
                    return (
                      <div
                        key={i}
                        className="flex items-center gap-3 py-2.5 border-b border-border/15 last:border-0"
                      >
                        <span
                          className="w-3 h-3 rounded-full shrink-0"
                          aria-hidden="true"
                          style={{ backgroundColor: dotColor }}
                        />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium leading-tight">
                            {item.label}
                          </p>
                          <p className="text-[10px] text-muted-foreground">
                            {item.time}
                          </p>
                        </div>
                        <span className="text-[10px] text-muted-foreground/60 shrink-0">
                          {Math.round(item.confidence * 100)}%
                        </span>
                      </div>
                    );
                  })}
            </div>
          )}
        </div>
      </div>

      {/* ── Today's Session Emotions (live EEG history) ────────────────── */}
      {history.length > 0 && isStreaming && (
        <div
          className="rounded-2xl overflow-hidden"
          style={{
            background: "hsl(165,20%,7%,0.7)",
            border: "1px solid hsl(165,20%,17%,0.5)",
            boxShadow: "0 1px 3px hsl(222,30%,3%,0.4)",
          }}
        >
          <div className="p-5">
            <div className="flex items-center gap-2 mb-3">
              <Brain className="h-4 w-4 text-emerald-400" />
              <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground/60">
                This Session
              </p>
            </div>
            <div className="space-y-1">
              {[...history].reverse().map((item, i) => {
                const dotColor =
                  EMOTION_COLORS[item.emotion] ?? EMOTION_COLORS.neutral;
                return (
                  <div
                    key={i}
                    className="flex items-center gap-3 py-2 border-b border-border/15 last:border-0"
                  >
                    <span
                      className="w-2.5 h-2.5 rounded-full shrink-0"
                      aria-hidden="true"
                      style={{ backgroundColor: dotColor }}
                    />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium leading-tight">
                        {item.label}
                      </p>
                      <p className="text-[10px] text-muted-foreground">
                        {item.time}
                      </p>
                    </div>
                    <span className="text-[10px] text-muted-foreground/60 shrink-0">
                      {Math.round(item.confidence * 100)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
