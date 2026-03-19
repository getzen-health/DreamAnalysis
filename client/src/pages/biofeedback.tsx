import { useState, useEffect, useRef, useMemo } from "react";
// PRODUCT.md: Phase 1 — The aha moment. User watches stress drop live during breathing.
import { useLocation } from "wouter";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useDevice } from "@/hooks/use-device";
import { useVoiceEmotion } from "@/hooks/use-voice-emotion";
import { getMLApiUrl } from "@/lib/ml-api";
import { Wind, Play, Square, Radio, TrendingDown, TrendingUp, Minus, Music, Headphones, ExternalLink } from "lucide-react";
import { getParticipantId } from "@/lib/participant";
import { hapticLight, hapticMedium, hapticSuccess } from "@/lib/haptics";
import SpotifyConnect from "@/components/spotify-connect";
import { InterventionSummary, type InterventionMetrics } from "@/components/intervention-summary";

// ─── Breathing exercise definitions ──────────────────────────────────────────

interface BreathPhase {
  label: string;
  duration: number;  // seconds
  expand: boolean;   // true = lungs filling
}

interface Exercise {
  id: string;
  name: string;
  tagline: string;
  science: string;
  phases: BreathPhase[];
  color: string;
  fillColor: string;    // rgba for SVG area fill
  strokeColor: string;  // for the arc
}

const EXERCISES: Exercise[] = [
  {
    id: "coherence",
    name: "Coherence",
    tagline: "Heart-brain sync",
    science: "5.5 breaths/min maximises heart rate variability — the gold standard for calm.",
    phases: [
      { label: "Inhale", duration: 5, expand: true },
      { label: "Exhale", duration: 5, expand: false },
    ],
    color: "hsl(142, 65%, 48%)",
    fillColor: "rgba(34,197,94,0.15)",
    strokeColor: "hsl(142, 65%, 48%)",
  },
  {
    id: "478",
    name: "4-7-8",
    tagline: "Fast anxiety relief",
    science: "Extended hold triggers the vagal brake, dropping heart rate within 30 seconds.",
    phases: [
      { label: "Inhale", duration: 4, expand: true },
      { label: "Hold",   duration: 7, expand: true },
      { label: "Exhale", duration: 8, expand: false },
    ],
    color: "hsl(250, 85%, 65%)",
    fillColor: "rgba(99,102,241,0.15)",
    strokeColor: "hsl(250, 85%, 65%)",
  },
  {
    id: "box",
    name: "Box",
    tagline: "Steady focus",
    science: "Equal-ratio breathing resets the autonomic nervous system. Used by Navy SEALs under fire.",
    phases: [
      { label: "Inhale", duration: 4, expand: true },
      { label: "Hold",   duration: 4, expand: true },
      { label: "Exhale", duration: 4, expand: false },
      { label: "Hold",   duration: 4, expand: false },
    ],
    color: "hsl(210, 85%, 60%)",
    fillColor: "rgba(59,130,246,0.15)",
    strokeColor: "hsl(210, 85%, 60%)",
  },
  {
    id: "deep",
    name: "Deep Relax",
    tagline: "Activates rest mode",
    science: "Extended exhale activates the parasympathetic system — the body's built-in off-switch.",
    phases: [
      { label: "Inhale", duration: 4, expand: true },
      { label: "Hold",   duration: 2, expand: true },
      { label: "Exhale", duration: 8, expand: false },
    ],
    color: "hsl(270, 70%, 65%)",
    fillColor: "rgba(168,85,247,0.15)",
    strokeColor: "hsl(270, 70%, 65%)",
  },
  {
    id: "physiological-sigh",
    name: "Physio Sigh",
    tagline: "Fastest anxiety relief",
    science: "Double inhale maximally inflates alveoli, flushing CO₂. Stanford 2023: superior to mindfulness for acute stress in real-time.",
    phases: [
      { label: "Inhale",  duration: 2, expand: true },
      { label: "Inhale+", duration: 1, expand: true },
      { label: "Exhale",  duration: 8, expand: false },
    ],
    color: "hsl(176, 70%, 45%)",
    fillColor: "rgba(20,184,166,0.15)",
    strokeColor: "hsl(176, 70%, 45%)",
  },
  {
    id: "cyclic-sighing",
    name: "Cyclic Sigh",
    tagline: "Sustained calm",
    science: "1:2 inhale:exhale ratio prolongs vagal tone. Shown to reduce self-reported anxiety 44% over 5 minutes.",
    phases: [
      { label: "Inhale", duration: 5, expand: true },
      { label: "Exhale", duration: 10, expand: false },
    ],
    color: "hsl(43, 90%, 55%)",
    fillColor: "rgba(234,179,8,0.15)",
    strokeColor: "hsl(43, 90%, 55%)",
  },
  {
    id: "power-breath",
    name: "Power Breath",
    tagline: "Energise & focus",
    science: "Short 2:1 ratio raises sympathetic tone and blood oxygenation — a legal pre-performance stimulant.",
    phases: [
      { label: "Inhale", duration: 4, expand: true },
      { label: "Exhale", duration: 2, expand: false },
    ],
    color: "hsl(22, 90%, 58%)",
    fillColor: "rgba(249,115,22,0.15)",
    strokeColor: "hsl(22, 90%, 58%)",
  },
];

// ─── Music playlists ──────────────────────────────────────────────────────────

interface Playlist {
  id: string;
  title: string;
  description: string;
  detail: string;       // science note
  bpm?: string;
  spotifyUrl: string;   // Spotify web URL
  youtubeUrl: string;   // YouTube URL
  mood: "calm" | "focus";
  color: string;
}

const PLAYLISTS: Playlist[] = [
  // ── calm mood ─────────────────────────────────────────────────────────────
  {
    id: "deep-calm",
    title: "Deep Calm",
    description: "Spotify's official deep calm playlist",
    detail: "Curated slow ambient tracks (55–65 BPM) that lower heart rate and cortisol. EEG studies show increased alpha power within 8 min.",
    bpm: "55–65 BPM",
    spotifyUrl: "https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0",
    youtubeUrl: "https://www.youtube.com/results?search_query=deep+calm+music+relaxation+1+hour",
    mood: "calm",
    color: "hsl(250, 70%, 65%)",
  },
  {
    id: "sleep-ambient",
    title: "Sleep & Deep Relaxation",
    description: "Delta-wave ambient for deep rest",
    detail: "Delta-range (0.5–4 Hz) binaural beats embedded in ambient soundscapes promote deep relaxation and sleep-onset. Best with headphones.",
    bpm: "< 60 BPM",
    spotifyUrl: "https://open.spotify.com/playlist/37i9dQZF1DWZd79rJ6a7lp",
    youtubeUrl: "https://www.youtube.com/results?search_query=delta+waves+deep+relaxation+sleep+music+8+hours",
    mood: "calm",
    color: "hsl(230, 60%, 55%)",
  },
  {
    id: "nature-rain",
    title: "Rain & Nature Sounds",
    description: "Pink noise + natural ambience",
    detail: "Pink noise reduces cortisol and masks sudden sounds that spike stress. Used in clinical anxiety and insomnia research.",
    spotifyUrl: "https://open.spotify.com/playlist/37i9dQZF1DWN1Y7lu9lY4v",
    youtubeUrl: "https://www.youtube.com/results?search_query=rain+sounds+deep+sleep+calm+10+hours",
    mood: "calm",
    color: "hsl(176, 65%, 45%)",
  },
  // ── focus mood ─────────────────────────────────────────────────────────────
  {
    id: "deep-focus",
    title: "Deep Focus",
    description: "Spotify's official deep focus playlist",
    detail: "Spotify's most-streamed focus playlist — low-tempo instrumental tracks that sustain beta-wave activity without lyric distraction. 90–110 BPM.",
    bpm: "90–110 BPM",
    spotifyUrl: "https://open.spotify.com/playlist/37i9dQZF1DWZeKCadgRdKQ",
    youtubeUrl: "https://www.youtube.com/results?search_query=deep+focus+music+study+concentration+3+hours",
    mood: "focus",
    color: "hsl(142, 65%, 48%)",
  },
  {
    id: "focus-flow",
    title: "Focus Flow",
    description: "Alpha-theta entrainment for flow state",
    detail: "Alpha (8–12 Hz) binaural beats target the flow-state EEG signature — sustained attention without anxiety. Use headphones for entrainment to work.",
    bpm: "Alpha 10 Hz",
    spotifyUrl: "https://open.spotify.com/playlist/37i9dQZF1DWSnM28ZrCAsO",
    youtubeUrl: "https://www.youtube.com/results?search_query=alpha+waves+deep+focus+flow+state+study+music",
    mood: "focus",
    color: "hsl(210, 85%, 60%)",
  },
  {
    id: "intense-study",
    title: "Intense Studying",
    description: "High-beta instrumental for deep work",
    detail: "Beta-range (15–30 Hz) music activates prefrontal cortex during demanding cognitive tasks. Best for analytical work, coding, and writing.",
    bpm: "110–130 BPM",
    spotifyUrl: "https://open.spotify.com/playlist/37i9dQZF1DX8NTLI2TtZa6",
    youtubeUrl: "https://www.youtube.com/results?search_query=intense+studying+music+deep+concentration+no+lyrics",
    mood: "focus",
    color: "hsl(30, 80%, 55%)",
  },
];

// ─── Intervention outcome helper ──────────────────────────────────────────────


function reportInterventionOutcome(
  userId: string,
  interventionType: string,
  stressAfter: number,        // 0-1 scale
  focusAfter: number,         // 0-1 scale
  feltHelpful: boolean,
) {
  const base = getMLApiUrl();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (base.includes("ngrok")) headers["ngrok-skip-browser-warning"] = "true";
  fetch(`${base}/api/interventions/outcome`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      user_id: userId,
      intervention_type: interventionType,
      stress_after: stressAfter,
      focus_after: focusAfter,
      felt_helpful: feltHelpful,
    }),
  }).catch(() => {});
}

// ─── Timing helper ────────────────────────────────────────────────────────────

function getBreathState(elapsedMs: number, phases: BreathPhase[]) {
  const cycleDurationMs = phases.reduce((s, p) => s + p.duration * 1000, 0);
  const cycleElapsed = ((elapsedMs % cycleDurationMs) + cycleDurationMs) % cycleDurationMs;
  let acc = 0;
  for (let i = 0; i < phases.length; i++) {
    const phaseDurationMs = phases[i].duration * 1000;
    if (cycleElapsed < acc + phaseDurationMs) {
      return { phaseIdx: i, phaseProgress: (cycleElapsed - acc) / phaseDurationMs };
    }
    acc += phaseDurationMs;
  }
  return { phaseIdx: 0, phaseProgress: 0 };
}

// ─── Chart helpers ────────────────────────────────────────────────────────────

const CHART_W = 600;
const CHART_H = 100;
const MAX_READINGS = 90; // ~2 min at 1.5s intervals

interface Reading { t: number; stress: number; }

function buildPaths(readings: Reading[]) {
  if (readings.length < 2) return { line: "", area: "" };
  const pts = readings.map((r, i) => ({
    x: (i / (readings.length - 1)) * CHART_W,
    y: CHART_H - (r.stress / 100) * CHART_H,
  }));
  const line = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
  const area = line + ` L ${CHART_W} ${CHART_H} L 0 ${CHART_H} Z`;
  return { line, area };
}

// ─── Phase guidance text ──────────────────────────────────────────────────────

function guidanceText(label: string): string {
  if (label === "Inhale+") return "One more quick sniff to top off your lungs…";
  if (label === "Inhale") return "Breathe in slowly through your nose…";
  if (label === "Exhale") return "Let it all go — out through your mouth…";
  return "Hold gently. Stay still.";
}

// ─── Main component ───────────────────────────────────────────────────────────

type SessionPhase = "idle" | "active" | "done";
type ActiveTab = "breathing" | "music";
type MusicMood = "calm" | "focus";

export default function Biofeedback() {
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const { lastResult: voiceResult } = useVoiceEmotion();
  const [location] = useLocation();
  const userId = useRef(getParticipantId());

  // ── Tab + music state ──────────────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState<ActiveTab>("breathing");
  const [musicMood, setMusicMood] = useState<MusicMood>("calm");

  // ── Parse URL params once on mount ────────────────────────────────────────
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const tab = params.get("tab");
    const mood = params.get("mood");
    const protocol = params.get("protocol");
    const autoStart = params.get("auto") === "true";

    if (tab === "music") {
      setActiveTab("music");
      if (mood === "focus") setMusicMood("focus");
      else setMusicMood("calm");
    } else if (protocol) {
      const found = EXERCISES.find(e => e.id === protocol);
      if (found) setExercise(found);
      if (autoStart) {
        // Short delay so the page renders before auto-start kicks in
        setTimeout(() => setAutoStartPending(true), 400);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location]);

  const [sessionPhase, setSessionPhase] = useState<SessionPhase>("idle");
  const [exercise, setExercise] = useState<Exercise>(EXERCISES[0]);
  const [autoStartPending, setAutoStartPending] = useState(false);
  const [readings, setReadings] = useState<Reading[]>([]);
  const [breathPhaseIdx, setBreathPhaseIdx] = useState(0);
  const [breathPhaseProgress, setBreathPhaseProgress] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [startStress, setStartStress] = useState<number | null>(null);
  // Before/after snapshots for InterventionSummary
  const [beforeSnapshot, setBeforeSnapshot] = useState<InterventionMetrics | null>(null);
  const [afterSnapshot, setAfterSnapshot] = useState<InterventionMetrics | null>(null);

  const sessionStartRef = useRef<number>(0);
  const elapsedSecRef = useRef(0);

  // ── Track latest stress via ref so intervals don't go stale ──────────────
  const latestStressRef = useRef<number | null>(null);
  useEffect(() => {
    const s =
      latestFrame?.analysis?.emotions?.stress_index ??
      latestFrame?.analysis?.stress?.stress_index ??
      null;
    if (s != null) latestStressRef.current = s * 100;
  }, [latestFrame]);

  // ── Get current stress (real EEG → voice → simulation) ───────────────────
  const getStress = () => {
    if (isStreaming && latestStressRef.current != null) return latestStressRef.current;
    // Use voice-derived stress as the simulation baseline if available
    const voiceBase = voiceResult ? (voiceResult.stress_from_watch ?? 0.6) * 100 : 60;
    const progress = elapsedSecRef.current / 180;
    const base = voiceBase - (voiceBase * 0.45) * Math.min(progress * 1.3, 1);
    return Math.max(8, Math.min(92, base + (Math.random() - 0.5) * 9));
  };

  // ── Build InterventionMetrics snapshot from current frame ─────────────────
  const buildSnapshot = (): InterventionMetrics => {
    const stress =
      latestFrame?.analysis?.emotions?.stress_index ??
      latestFrame?.analysis?.stress?.stress_index ??
      getStress() / 100;
    const focus =
      latestFrame?.analysis?.emotions?.focus_index ??
      0.5; // default when not streaming
    const hrv: number | null =
      ((latestFrame?.analysis as Record<string, unknown> | undefined)?.hrv as number | undefined) ?? null;
    return { stress, focus, hrv };
  };

  // ── Breathing animation (50ms, smooth) ────────────────────────────────────
  useEffect(() => {
    if (sessionPhase !== "active") return;
    const id = setInterval(() => {
      const ms = Date.now() - sessionStartRef.current;
      const { phaseIdx, phaseProgress } = getBreathState(ms, exercise.phases);
      setBreathPhaseIdx(phaseIdx);
      setBreathPhaseProgress(phaseProgress);
    }, 50);
    return () => clearInterval(id);
  }, [sessionPhase, exercise]);

  // ── Haptic feedback on phase transitions ──────────────────────────────────
  // Fires once per phase change (not every 50ms tick).
  useEffect(() => {
    if (sessionPhase !== "active") return;
    const phase = exercise.phases[breathPhaseIdx];
    if (!phase) return;
    const label = phase.label.toLowerCase();
    if (label.startsWith("inhale")) {
      hapticLight();          // gentle tap: new breath starting
    } else if (label.startsWith("hold")) {
      hapticMedium();         // medium tap: hold transition
    } else if (label.startsWith("exhale")) {
      hapticMedium();         // medium tap: release breath
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [breathPhaseIdx, sessionPhase]);

  // ── Session clock + reading collector (1.5s) ──────────────────────────────
  useEffect(() => {
    if (sessionPhase !== "active") return;
    const id = setInterval(() => {
      elapsedSecRef.current += 1.5;
      setElapsed(Math.round(elapsedSecRef.current));
      const stress = getStress();
      setReadings(prev => [...prev, { t: elapsedSecRef.current, stress }].slice(-MAX_READINGS));
    }, 1500);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionPhase, isStreaming]);

  // ── Auto-start when pending (after URL-param protocol selection) ──────────
  useEffect(() => {
    if (autoStartPending && sessionPhase === "idle") {
      setAutoStartPending(false);
      handleStart();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoStartPending]);

  // ── Start / stop ──────────────────────────────────────────────────────────
  const handleStart = () => {
    sessionStartRef.current = Date.now();
    elapsedSecRef.current = 0;
    setReadings([]);
    setElapsed(0);
    setBreathPhaseIdx(0);
    setBreathPhaseProgress(0);
    setStartStress(getStress());
    setBeforeSnapshot(buildSnapshot());
    setAfterSnapshot(null);
    setSessionPhase("active");
  };

  const handleStop = () => {
    setAfterSnapshot(buildSnapshot());
    setSessionPhase("done");
    hapticSuccess(); // celebrate session completion on mobile
    // Record session to breathing API for coherence scoring
    const finalStressNorm = getStress() / 100;
    const preStressNorm = startStress != null ? startStress / 100 : undefined;
    fetch(`${getMLApiUrl()}/api/breathing/session/complete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: userId.current,
        pattern_id: exercise.id,
        duration_s: elapsed,
        completed_cycles: Math.floor(elapsed / (exercise.phases.reduce((s, p) => s + p.duration, 0))),
        stress_after: finalStressNorm,
        ...(preStressNorm != null ? { stress_before: preStressNorm } : {}),
      }),
    }).catch(() => {});
    // Report outcome to intervention engine after 5 minutes
    const finalStress = (latestStressRef.current ?? 50) / 100;
    setTimeout(() => {
      reportInterventionOutcome(
        userId.current,
        "breathing",
        finalStress,
        0.5,   // focus unknown at this point
        true,  // assume helpful (user stayed for the full session)
      );
    }, 5 * 60 * 1000);
  };

  const handleReset = () => {
    setSessionPhase("idle");
    setReadings([]);
    setStartStress(null);
    setBeforeSnapshot(null);
    setAfterSnapshot(null);
    setElapsed(0);
  };

  // ── "Before you start" context card values ────────────────────────────────
  // stress_index is 0-1 from the ML backend; keep on 0-1 for badge thresholds
  const stressLevel: number =
    latestFrame?.analysis?.emotions?.stress_index ??
    latestFrame?.analysis?.stress?.stress_index ??
    0.3; // default to moderate when offline

  const bandPowers = latestFrame?.analysis?.band_powers;
  const alphaBetaRatio: number = (() => {
    if (!bandPowers) return 1.0;
    const alpha = bandPowers["alpha"] ?? 0;
    const beta = bandPowers["beta"] ?? 0;
    if (beta === 0) return 1.0;
    return alpha / beta;
  })();

  // ── Derived display values ────────────────────────────────────────────────
  const currentPhase = exercise.phases[breathPhaseIdx];

  const expansion = useMemo(() => {
    if (!currentPhase) return 0;
    if (currentPhase.label === "Inhale") return breathPhaseProgress;
    if (currentPhase.label === "Exhale") return 1 - breathPhaseProgress;
    return currentPhase.expand ? 1 : 0; // Hold
  }, [currentPhase, breathPhaseProgress]);

  const phaseCountdown = currentPhase
    ? Math.max(1, Math.ceil(currentPhase.duration * (1 - breathPhaseProgress)))
    : 0;

  const currentStress = readings.length > 0 ? readings[readings.length - 1].stress : null;
  const stressDelta =
    startStress != null && currentStress != null && readings.length > 3
      ? Math.round(currentStress - startStress)
      : null;

  const { line: stressLine, area: stressArea } = buildPaths(readings);

  // Circle geometry
  const MIN_R = 32;
  const MAX_R = 70;
  const RING_R = 84;
  const circleR = MIN_R + (MAX_R - MIN_R) * expansion;
  const ringCircumference = 2 * Math.PI * RING_R;
  const ringOffset = ringCircumference * (1 - breathPhaseProgress);

  const formatTime = (s: number) =>
    `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, "0")}`;

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <main className="p-4 md:p-6 space-y-6 max-w-5xl mx-auto">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {activeTab === "music"
            ? <Music className="h-6 w-6 text-primary" />
            : <Wind className="h-6 w-6 text-primary" />
          }
          <div>
            <h2 className="text-xl font-semibold">Biofeedback</h2>
            <p className="text-xs text-muted-foreground">
              {activeTab === "music"
                ? "Music shown to reduce stress and sharpen focus"
                : "Watch your stress respond in real time as you breathe"}
            </p>
          </div>
        </div>
        {activeTab === "breathing" && sessionPhase === "active" && (
          <div className="flex items-center gap-4">
            <span className="text-sm font-mono text-foreground/50">{formatTime(elapsed)}</span>
            <Button
              size="sm"
              variant="ghost"
              onClick={handleStop}
              className="border border-destructive/30 text-destructive hover:bg-destructive/10"
            >
              <Square className="h-3 w-3 mr-1" />
              Stop
            </Button>
          </div>
        )}
      </div>

      {/* Tab switcher */}
      <div className="flex gap-1 p-1 rounded-lg bg-muted/20 border border-border/30 w-fit">
        <button
          onClick={() => setActiveTab("breathing")}
          className={`flex items-center gap-1.5 px-4 py-1.5 rounded-md text-sm font-medium transition-all ${
            activeTab === "breathing"
              ? "bg-background/80 text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <Wind className="h-3.5 w-3.5" />
          Breathing
        </button>
        <button
          onClick={() => setActiveTab("music")}
          className={`flex items-center gap-1.5 px-4 py-1.5 rounded-md text-sm font-medium transition-all ${
            activeTab === "music"
              ? "bg-background/80 text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <Headphones className="h-3.5 w-3.5" />
          Music
        </button>
      </div>

      {/* ── MUSIC TAB ── */}
      {activeTab === "music" && (
        <div className="space-y-5">
          {/* Spotify connect / auto-play */}
          <SpotifyConnect autoPlayMood={musicMood as "calm" | "focus"} />

          {/* Mood switcher */}
          <div className="flex items-center gap-3">
            <span className="text-xs text-muted-foreground">Mood:</span>
            <div className="flex gap-1 p-0.5 rounded-lg bg-muted/20 border border-border/30">
              <button
                onClick={() => setMusicMood("calm")}
                className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                  musicMood === "calm"
                    ? "bg-background/80 text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Calm / Stress relief
              </button>
              <button
                onClick={() => setMusicMood("focus")}
                className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                  musicMood === "focus"
                    ? "bg-background/80 text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Focus / Deep work
              </button>
            </div>
          </div>

          {/* Playlist cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {PLAYLISTS.filter(p => p.mood === musicMood).map(playlist => (
              <Card key={playlist.id} className="glass-card p-5 rounded-xl space-y-3">
                <div className="flex items-start justify-between">
                  <div
                    className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
                    style={{ background: `${playlist.color}20`, border: `1px solid ${playlist.color}40` }}
                  >
                    <Music className="h-4 w-4" style={{ color: playlist.color }} />
                  </div>
                  {playlist.bpm && (
                    <span
                      className="text-[10px] font-mono px-2 py-0.5 rounded-full"
                      style={{ background: `${playlist.color}15`, color: playlist.color }}
                    >
                      {playlist.bpm}
                    </span>
                  )}
                </div>
                <div>
                  <p className="text-sm font-semibold">{playlist.title}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">{playlist.description}</p>
                </div>
                <p className="text-[10px] text-muted-foreground/60 leading-relaxed">
                  {playlist.detail}
                </p>
                <div className="flex gap-2 pt-1">
                  <a
                    href={playlist.spotifyUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex-1 flex items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs font-medium transition-colors"
                    style={{ background: `${playlist.color}15`, color: playlist.color }}
                  >
                    Spotify
                    <ExternalLink className="h-3 w-3" />
                  </a>
                  <a
                    href={playlist.youtubeUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex-1 flex items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs font-medium text-muted-foreground hover:text-foreground border border-border/30 transition-colors"
                  >
                    YouTube
                    <ExternalLink className="h-3 w-3" />
                  </a>
                </div>
              </Card>
            ))}
          </div>

          {/* Effectiveness note */}
          <Card className="glass-card p-4 rounded-xl">
            <p className="text-xs text-muted-foreground leading-relaxed">
              <span className="text-foreground font-medium">How it works: </span>
              After you listen, go back to the Brain Monitor — if your stress dropped, it worked.
              The intervention engine tracks which music actually reduced your cortisol and will
              recommend it again next time.
            </p>
          </Card>
        </div>
      )}

      {/* ── BREATHING TAB content below ── */}
      {activeTab === "breathing" && (
      <>

      {/* Before You Start context card — always visible */}
      <Card className="mb-6">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Before You Start</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {/* Stress level indicator */}
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Current Stress</span>
              <Badge variant={stressLevel > 0.6 ? "destructive" : stressLevel > 0.3 ? "secondary" : "outline"}>
                {stressLevel > 0.6 ? "Elevated" : stressLevel > 0.3 ? "Moderate" : "Low"}
              </Badge>
            </div>
            {/* Recommendation sentence */}
            <p className="text-xs text-muted-foreground">
              {stressLevel > 0.6
                ? "Your stress is elevated — this session will help lower it."
                : stressLevel > 0.3
                ? "Some tension detected — breathing exercises will help you relax."
                : "You're looking calm — use this session to deepen your focus."}
            </p>
            {/* Alpha/beta ratio — shown when we have live or any frame data */}
            {latestFrame && (
              <div className="flex items-center gap-2 mt-2">
                <span className="text-xs text-muted-foreground">Alpha/Beta ratio</span>
                <span className="text-xs font-mono">
                  {alphaBetaRatio.toFixed(2)}
                </span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* No-device banner */}
      {!isStreaming && sessionPhase === "idle" && (
        <div className="flex items-center gap-3 p-3 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning">
          <Radio className="h-4 w-4 shrink-0" />
          {voiceResult
            ? `Using a voice-derived stress baseline (${Math.round((voiceResult.stress_from_watch ?? 0.5) * 100)}%). Optional EEG can add live stress tracking later.`
            : "Run a voice analysis on the dashboard to set your stress baseline. EEG is optional later for live stress tracking."}
        </div>
      )}

      {/* ── IDLE ── exercise picker ── */}
      {sessionPhase === "idle" && (
        <div className="space-y-5">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {EXERCISES.map(ex => {
              const active = exercise.id === ex.id;
              return (
                <button
                  key={ex.id}
                  onClick={() => setExercise(ex)}
                  className={`p-4 rounded-xl border text-left transition-all ${
                    active
                      ? "border-primary/50 bg-primary/8"
                      : "border-border/30 hover:border-border/60 bg-card/30"
                  }`}
                >
                  <div className="w-3 h-3 rounded-full mb-2.5" style={{ background: ex.color }} />
                  <p className="text-sm font-semibold">{ex.name}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">{ex.tagline}</p>
                  <p className="text-[10px] text-muted-foreground/50 mt-2 font-mono">
                    {ex.phases.map(p => p.duration).join("–")} sec
                  </p>
                </button>
              );
            })}
          </div>

          {/* Selected exercise detail card */}
          <Card className="glass-card p-6 rounded-xl">
            <div className="flex flex-col md:flex-row md:items-center gap-6">
              <div className="flex-1 space-y-3">
                <div>
                  <h3 className="font-semibold text-base">{exercise.name} Breathing</h3>
                  <p className="text-sm text-muted-foreground mt-1">{exercise.science}</p>
                </div>
                <div className="flex gap-4">
                  {exercise.phases.map((p, i) => (
                    <div key={i} className="text-center">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">{p.label}</p>
                      <p className="text-xl font-mono font-bold mt-0.5" style={{ color: exercise.color }}>
                        {p.duration}
                      </p>
                      <p className="text-[10px] text-muted-foreground">sec</p>
                    </div>
                  ))}
                </div>
              </div>
              <Button
                onClick={handleStart}
                className="bg-primary/20 border border-primary/30 text-primary hover:bg-primary/30 px-10 h-11 shrink-0"
              >
                <Play className="h-4 w-4 mr-2" />
                Start Session
              </Button>
            </div>
          </Card>
        </div>
      )}

      {/* ── ACTIVE ── breathing + chart ── */}
      {sessionPhase === "active" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* Breathing circle */}
          <Card className="glass-card p-6 rounded-xl flex flex-col items-center gap-4">
            <svg width="224" height="224" viewBox="0 0 224 224">
              <defs>
                <radialGradient id="breath-fill" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor={exercise.color} stopOpacity="0.3" />
                  <stop offset="100%" stopColor={exercise.color} stopOpacity="0.05" />
                </radialGradient>
              </defs>
              {/* outer guide ring */}
              <circle cx="112" cy="112" r={MAX_R + 14} fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
              {/* expanding fill blob */}
              <circle
                cx="112" cy="112" r={circleR}
                fill="url(#breath-fill)"
                style={{ transition: "r 0.08s linear" }}
              />
              {/* expanding border */}
              <circle
                cx="112" cy="112" r={circleR}
                fill="none"
                stroke={exercise.color}
                strokeWidth="1.5"
                opacity="0.5"
                style={{ transition: "r 0.08s linear" }}
              />
              {/* progress arc */}
              <circle
                cx="112" cy="112" r={RING_R}
                fill="none"
                stroke={exercise.color}
                strokeWidth="7"
                strokeDasharray={ringCircumference}
                strokeDashoffset={ringOffset}
                strokeLinecap="round"
                transform="rotate(-90 112 112)"
                style={{ transition: "stroke-dashoffset 0.08s linear" }}
              />
              {/* phase label */}
              <text
                x="112" y="106"
                textAnchor="middle"
                fill="white"
                fontSize="15"
                fontFamily="system-ui, sans-serif"
                fontWeight="500"
              >
                {currentPhase?.label}
              </text>
              {/* countdown */}
              <text
                x="112" y="134"
                textAnchor="middle"
                fontSize="34"
                fontFamily="ui-monospace, monospace"
                fontWeight="700"
                fill={exercise.color}
              >
                {phaseCountdown}
              </text>
            </svg>

            {/* Phase breadcrumbs */}
            <div className="flex items-center gap-3">
              {exercise.phases.map((p, i) => (
                <span
                  key={i}
                  className="text-xs px-2 py-0.5 rounded transition-all"
                  style={
                    i === breathPhaseIdx
                      ? { color: exercise.color, fontWeight: 600 }
                      : { color: "rgba(255,255,255,0.25)" }
                  }
                >
                  {p.label}
                </span>
              ))}
            </div>

            {/* Guidance text */}
            <p className="text-xs text-muted-foreground text-center max-w-[200px]">
              {guidanceText(currentPhase?.label ?? "Inhale")}
            </p>
          </Card>

          {/* Stress chart — live when EEG connected, simulated estimate otherwise */}
          <Card className="glass-card p-6 rounded-xl flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium">
                {isStreaming ? "Live Stress Response" : "Estimated Stress Response"}
              </h3>
              <div className="flex items-center gap-3 text-xs font-mono text-muted-foreground">
                {startStress != null && (
                  <span>Start <span className="text-foreground">{Math.round(startStress)}</span></span>
                )}
                {currentStress != null && (
                  <span>Now <span className="text-foreground">{Math.round(currentStress)}</span></span>
                )}
                {stressDelta !== null && (
                  <span
                    className="flex items-center gap-0.5"
                    style={{ color: stressDelta < 0 ? "hsl(142,65%,48%)" : "hsl(0,70%,60%)" }}
                  >
                    {stressDelta < 0
                      ? <TrendingDown className="h-3 w-3" />
                      : stressDelta > 0
                        ? <TrendingUp className="h-3 w-3" />
                        : <Minus className="h-3 w-3" />}
                    {Math.abs(stressDelta)}
                  </span>
                )}
              </div>
            </div>

            {/* SVG chart */}
            <div className="flex-1 relative min-h-[140px]">
              {readings.length < 3 ? (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
                  <div className="w-6 h-6 rounded-full border-2 border-primary/40 border-t-primary animate-spin" />
                  <p className="text-xs text-muted-foreground animate-pulse">Collecting data…</p>
                </div>
              ) : (
                <svg
                  viewBox={`0 0 ${CHART_W} ${CHART_H}`}
                  preserveAspectRatio="none"
                  className="w-full h-full"
                >
                  <defs>
                    <linearGradient id="stress-area-grad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="hsl(0,70%,60%)" stopOpacity="0.35" />
                      <stop offset="100%" stopColor="hsl(0,70%,60%)" stopOpacity="0.02" />
                    </linearGradient>
                    <linearGradient id="stress-line-grad" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%" stopColor="hsl(0,70%,50%)" />
                      <stop offset="100%" stopColor="hsl(0,70%,70%)" />
                    </linearGradient>
                    <radialGradient id="stress-dot-grad" cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="hsl(0,70%,70%)" />
                      <stop offset="100%" stopColor="hsl(0,70%,50%)" />
                    </radialGradient>
                  </defs>
                  {/* grid lines */}
                  <line x1="0" y1={CHART_H * 0.5} x2={CHART_W} y2={CHART_H * 0.5}
                    stroke="rgba(255,255,255,0.06)" strokeWidth="1" strokeDasharray="4 4" />
                  <line x1="0" y1={CHART_H} x2={CHART_W} y2={CHART_H}
                    stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
                  {/* area fill */}
                  <path d={stressArea} fill="url(#stress-area-grad)" />
                  {/* stress line */}
                  <path d={stressLine} fill="none" stroke="url(#stress-line-grad)" strokeWidth="2.5" strokeLinejoin="round" />
                  {/* latest dot */}
                  {(() => {
                    const last = readings[readings.length - 1];
                    const x = CHART_W;
                    const y = CHART_H - (last.stress / 100) * CHART_H;
                    return (
                      <>
                        <circle cx={x} cy={y} r="5" fill="url(#stress-dot-grad)" />
                        <circle cx={x} cy={y} r="9" fill="none" stroke="url(#stress-line-grad)" strokeWidth="1" opacity="0.4" />
                      </>
                    );
                  })()}
                </svg>
              )}
            </div>

            {/* Y labels */}
            <div className="flex justify-between text-[10px] text-muted-foreground/40 mt-2">
              <span>calm</span>
              <span>Stress level (0–100)</span>
              <span>tense</span>
            </div>

            {/* Session progress */}
            {elapsed > 0 && (
              <div className="mt-4 pt-4 border-t border-border/20 flex items-center justify-between text-xs text-muted-foreground">
                <span>{formatTime(elapsed)} elapsed</span>
                {!isStreaming && (
                  <span className="opacity-50">Estimated from voice/baseline — connect EEG for live data</span>
                )}
              </div>
            )}
          </Card>
        </div>
      )}

      {/* ── DONE ── summary ── */}
      {sessionPhase === "done" && (
        <div className="max-w-lg mx-auto space-y-4">
          <Card className="glass-card p-8 rounded-xl text-center space-y-6">
            {/* Icon */}
            <div
              className="w-16 h-16 rounded-full mx-auto flex items-center justify-center"
              style={{ background: exercise.fillColor, border: `1px solid ${exercise.color}40` }}
            >
              <Wind className="h-8 w-8" style={{ color: exercise.color }} />
            </div>

            <div>
              <h3 className="text-xl font-semibold">Session complete</h3>
              <p className="text-sm text-muted-foreground mt-1">
                {exercise.name} · {formatTime(elapsed)}
              </p>
            </div>

            {/* Before / After */}
            {startStress != null && currentStress != null && readings.length > 5 && (
              <div className="grid grid-cols-3 gap-2 py-4 border-y border-border/20">
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Before</p>
                  <p className="text-3xl font-mono font-bold">{Math.round(startStress)}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground mb-1">After</p>
                  <p className="text-3xl font-mono font-bold">{Math.round(currentStress)}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Change</p>
                  <p
                    className="text-3xl font-mono font-bold flex items-center justify-center gap-1"
                    style={{ color: (stressDelta ?? 0) < 0 ? "hsl(142,65%,48%)" : "hsl(0,70%,60%)" }}
                  >
                    {(stressDelta ?? 0) < 0
                      ? <TrendingDown className="h-6 w-6" />
                      : <TrendingUp className="h-6 w-6" />}
                    {Math.abs(stressDelta ?? 0)}
                  </p>
                </div>
              </div>
            )}

            {/* Mini replay chart */}
            {readings.length >= 5 && (
              <div className="h-16">
                <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} preserveAspectRatio="none" className="w-full h-full">
                  <defs>
                    <linearGradient id="done-grad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="hsl(0,70%,60%)" stopOpacity="0.3" />
                      <stop offset="100%" stopColor="hsl(0,70%,60%)" stopOpacity="0.0" />
                    </linearGradient>
                    <linearGradient id="done-line-grad" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%" stopColor="hsl(0,70%,50%)" />
                      <stop offset="100%" stopColor="hsl(0,70%,70%)" />
                    </linearGradient>
                  </defs>
                  <path d={stressArea} fill="url(#done-grad)" />
                  <path d={stressLine} fill="none" stroke="url(#done-line-grad)" strokeWidth="2.5" strokeLinejoin="round" />
                </svg>
              </div>
            )}

            {/* Interpretation */}
            {stressDelta !== null && (
              <p className="text-sm text-muted-foreground">
                {stressDelta <= -10
                  ? "Strong response — your nervous system shifted noticeably."
                  : stressDelta <= -4
                    ? "Mild shift — consistent practice amplifies the effect."
                    : stressDelta < 4
                      ? "Steady state — the exercise kept you stable."
                      : "Stress rose slightly — try a slower exhale next time."}
              </p>
            )}

            <div className="flex gap-3 justify-center">
              <Button
                onClick={handleReset}
                variant="ghost"
                className="border border-border/30"
              >
                Change exercise
              </Button>
              <Button
                onClick={handleStart}
                className="bg-primary/20 border border-primary/30 text-primary hover:bg-primary/30"
              >
                <Play className="h-4 w-4 mr-2" />
                Go again
              </Button>
            </div>
          </Card>

          {/* Detailed before/after summary card */}
          {beforeSnapshot && afterSnapshot && (
            <InterventionSummary
              beforeMetrics={beforeSnapshot}
              afterMetrics={afterSnapshot}
              duration={elapsed}
              type={`${exercise.name} breathing`}
            />
          )}
        </div>
      )}

      </> /* end breathing tab */
      )}
    </main>
  );
}
