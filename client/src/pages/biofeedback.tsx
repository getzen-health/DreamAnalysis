import { useState, useEffect, useRef, useMemo } from "react";
// PRODUCT.md: Phase 1 — The aha moment. User watches stress drop live during breathing.
import { useLocation } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useDevice } from "@/hooks/use-device";
import { Wind, Play, Square, Radio, TrendingDown, TrendingUp, Minus, Music, Headphones, ExternalLink, FlaskConical, CheckCircle2, AlertCircle } from "lucide-react";
import { getParticipantId } from "@/lib/participant";

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
  // calm mood
  {
    id: "peaceful-piano",
    title: "Peaceful Piano",
    description: "Slow instrumental pieces",
    detail: "60–75 BPM. Activates the parasympathetic system via the autonomous nervous system's response to low-tempo music.",
    bpm: "60–75 BPM",
    spotifyUrl: "https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO",
    youtubeUrl: "https://www.youtube.com/results?search_query=peaceful+piano+calm+music",
    mood: "calm",
    color: "hsl(250, 70%, 65%)",
  },
  {
    id: "nature-sounds",
    title: "Nature & Rain",
    description: "Pink noise + natural ambience",
    detail: "Pink noise reduces cortisol and masks sudden environmental sounds that spike stress. Used in clinical anxiety research.",
    spotifyUrl: "https://open.spotify.com/playlist/37i9dQZF1DWN1Y7lu9lY4v",
    youtubeUrl: "https://www.youtube.com/results?search_query=rain+sounds+10+hours+sleep",
    mood: "calm",
    color: "hsl(176, 65%, 45%)",
  },
  {
    id: "lo-fi-chill",
    title: "Lo-Fi Chill Beats",
    description: "Soft hip-hop, 70–85 BPM",
    detail: "Self-selected calming music (Thoma et al. 2013) reduces salivary cortisol during acute stress recovery better than silence.",
    bpm: "70–85 BPM",
    spotifyUrl: "https://open.spotify.com/playlist/37i9dQZF1DWWQRwui0ExPn",
    youtubeUrl: "https://www.youtube.com/results?search_query=lofi+hip+hop+chill+beats",
    mood: "calm",
    color: "hsl(22, 80%, 58%)",
  },
  // focus mood
  {
    id: "binaural-gamma",
    title: "Binaural Beats 40 Hz",
    description: "Gamma-frequency entrainment",
    detail: "40 Hz gamma binaural beats improve selective attention and working memory within 10 min (Kraus et al. 2021). Use headphones.",
    bpm: "40 Hz beat",
    spotifyUrl: "https://open.spotify.com/playlist/37i9dQZF1DX0SM0LYsmbMT",
    youtubeUrl: "https://www.youtube.com/results?search_query=40hz+binaural+beats+focus+gamma",
    mood: "focus",
    color: "hsl(142, 65%, 48%)",
  },
  {
    id: "brain-food",
    title: "Brain Food",
    description: "Instrumental focus music",
    detail: "Moderate-tempo (~90–110 BPM) instrumental music boosts sustained attention without the distraction of lyrics.",
    bpm: "90–110 BPM",
    spotifyUrl: "https://open.spotify.com/playlist/37i9dQZF1DWXLeA8Omikj7",
    youtubeUrl: "https://www.youtube.com/results?search_query=brain+food+instrumental+focus+music",
    mood: "focus",
    color: "hsl(210, 85%, 60%)",
  },
  {
    id: "brown-noise",
    title: "Brown Noise",
    description: "Deep low-frequency noise",
    detail: "Brown noise (lower frequency than white/pink) improves ADHD focus and masks distracting office sounds. Preferred by 38% of focus-music users.",
    spotifyUrl: "https://open.spotify.com/playlist/37i9dQZF1DWUZ5bk6qqDSy",
    youtubeUrl: "https://www.youtube.com/results?search_query=brown+noise+focus+work+8+hours",
    mood: "focus",
    color: "hsl(30, 60%, 50%)",
  },
];

// ─── Intervention outcome helper ──────────────────────────────────────────────

function getMLApiBase(): string {
  try {
    const s = localStorage.getItem("ml_backend_url");
    if (s?.trim()) return s.trim().replace(/\/$/, "");
  } catch { /* ignore */ }
  return (import.meta.env.VITE_ML_API_URL as string | undefined) ?? "http://localhost:8000";
}

function reportInterventionOutcome(
  userId: string,
  interventionType: string,
  stressAfter: number,        // 0-1 scale
  focusAfter: number,         // 0-1 scale
  feltHelpful: boolean,
) {
  const base = getMLApiBase();
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
type ActiveTab = "breathing" | "music" | "evidence";
type MusicMood = "calm" | "focus";

interface EffectivenessData {
  user_id: string;
  total_outcomes: number;
  by_type: Record<string, {
    count: number;
    worked: number;
    avg_stress_delta: number;
    success_rate: number;
  }>;
}

export default function Biofeedback() {
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const [location] = useLocation();
  const userId = useRef(getParticipantId());

  // ── Tab + music state ──────────────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState<ActiveTab>("breathing");
  const [musicMood, setMusicMood] = useState<MusicMood>("calm");

  // ── Effectiveness data (Evidence tab) ─────────────────────────────────────
  const mlBase = (() => {
    try {
      const s = localStorage.getItem("ml_backend_url");
      if (s?.trim()) return s.trim().replace(/\/$/, "");
    } catch { /* */ }
    return (import.meta.env.VITE_ML_API_URL as string | undefined) ?? "http://localhost:8000";
  })();

  const { data: effectiveness } = useQuery<EffectivenessData>({
    queryKey: ["interventions/effectiveness", userId.current],
    queryFn: async () => {
      const res = await fetch(`${mlBase}/api/interventions/effectiveness/${userId.current}`);
      if (!res.ok) throw new Error("ML backend offline");
      return res.json();
    },
    staleTime: 5 * 60 * 1000,
    retry: 1,
  });

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

  // ── Get current stress (real or simulated) ────────────────────────────────
  const getStress = () => {
    if (isStreaming && latestStressRef.current != null) return latestStressRef.current;
    // Simulation: starts 55-65, drifts down over session
    const progress = elapsedSecRef.current / 180;
    const base = 60 - 30 * Math.min(progress * 1.3, 1);
    return Math.max(8, Math.min(92, base + (Math.random() - 0.5) * 9));
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
    setSessionPhase("active");
  };

  const handleStop = () => {
    setSessionPhase("done");
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
    setElapsed(0);
  };

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
        <button
          onClick={() => setActiveTab("evidence")}
          className={`flex items-center gap-1.5 px-4 py-1.5 rounded-md text-sm font-medium transition-all ${
            activeTab === "evidence"
              ? "bg-background/80 text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <FlaskConical className="h-3.5 w-3.5" />
          Evidence
        </button>
      </div>

      {/* ── MUSIC TAB ── */}
      {activeTab === "music" && (
        <div className="space-y-5">
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

          {/* Context note */}
          <p className="text-xs text-muted-foreground/70">
            {musicMood === "calm"
              ? "60–80 BPM music activates your parasympathetic system within minutes. Self-selected calming music reduces cortisol (Thoma et al. 2013)."
              : "Binaural beats require headphones — the two slightly different tones create a beat in the brain. 40 Hz gamma improves selective attention (Kraus et al. 2021)."}
          </p>

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

      {/* ── EVIDENCE TAB ── before/after EEG comparison per exercise ── */}
      {activeTab === "evidence" && (
        <div className="space-y-5">
          {/* Header */}
          <div>
            <h3 className="text-sm font-semibold">Intervention Library</h3>
            <p className="text-xs text-muted-foreground mt-0.5">
              Evidence behind each technique — and your personal before/after data once you start using them.
            </p>
          </div>

          {/* Personal effectiveness (when ML backend has data) */}
          {effectiveness && effectiveness.total_outcomes > 0 && (
            <Card className="glass-card p-5 rounded-xl">
              <div className="flex items-center gap-2 mb-4">
                <FlaskConical className="h-4 w-4 text-primary" />
                <h4 className="text-sm font-medium">Your Personal Before/After Data</h4>
                <span className="ml-auto text-[10px] text-muted-foreground">
                  {effectiveness.total_outcomes} session{effectiveness.total_outcomes !== 1 ? "s" : ""} recorded
                </span>
              </div>
              <div className="space-y-4">
                {Object.entries(effectiveness.by_type).map(([type, stats]) => {
                  const label = type.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
                  const pct = Math.round(stats.success_rate * 100);
                  const delta = Math.round(stats.avg_stress_delta * 100);
                  const worked = stats.success_rate >= 0.6;
                  return (
                    <div key={type}>
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-xs font-medium">{label}</span>
                        <div className="flex items-center gap-2">
                          {worked
                            ? <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                            : <AlertCircle className="h-3.5 w-3.5 text-muted-foreground" />
                          }
                          <span className={`text-xs font-mono ${worked ? "text-emerald-400" : "text-muted-foreground"}`}>
                            {pct}% success · avg −{delta}% stress
                          </span>
                        </div>
                      </div>
                      {/* Before bar = fixed at 100%, After bar = 100% - avg_delta */}
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="text-[10px] text-muted-foreground w-12 shrink-0">Before</span>
                          <div className="flex-1 bg-rose-500/15 rounded-full h-2">
                            <div className="h-full w-full bg-rose-500/50 rounded-full" />
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-[10px] text-muted-foreground w-12 shrink-0">After</span>
                          <div className="flex-1 bg-muted/20 rounded-full h-2 overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all duration-700"
                              style={{
                                width: `${Math.max(10, 100 - delta)}%`,
                                background: worked ? "hsl(142, 60%, 45%)" : "hsl(0, 60%, 55%)",
                              }}
                            />
                          </div>
                        </div>
                      </div>
                      <p className="text-[10px] text-muted-foreground mt-1">
                        {stats.count} session{stats.count !== 1 ? "s" : ""} · {stats.worked} helped
                      </p>
                    </div>
                  );
                })}
              </div>
            </Card>
          )}

          {/* No personal data yet */}
          {(!effectiveness || effectiveness.total_outcomes === 0) && (
            <Card className="glass-card p-4 rounded-xl flex items-center gap-3">
              <AlertCircle className="h-4 w-4 text-muted-foreground shrink-0" />
              <p className="text-xs text-muted-foreground">
                Your before/after EEG comparison will appear here after you complete your first
                breathing session. Run a session above — the intervention engine records your
                stress before and 5 minutes after.
              </p>
            </Card>
          )}

          {/* Science library — all 7 exercises with citations */}
          <div className="space-y-3">
            <h4 className="text-xs text-muted-foreground uppercase tracking-wider">Research Citations</h4>
            {EXERCISES.map(ex => (
              <Card key={ex.id} className="glass-card p-4 rounded-xl">
                <div className="flex items-start gap-3">
                  <div
                    className="w-2 h-2 rounded-full mt-1.5 shrink-0"
                    style={{ background: ex.color }}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-sm font-medium">{ex.name} Breathing</p>
                      <span
                        className="text-[10px] px-1.5 py-0.5 rounded font-mono shrink-0"
                        style={{ background: `${ex.color}20`, color: ex.color }}
                      >
                        {ex.phases.map(p => p.duration).join("–")}s
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
                      {ex.science}
                    </p>
                    {/* Phase breakdown */}
                    <div className="flex gap-3 mt-2">
                      {ex.phases.map((p, i) => (
                        <div key={i} className="flex items-center gap-1">
                          <span className="text-[10px] text-muted-foreground/60">{p.label}</span>
                          <span className="text-[10px] font-mono" style={{ color: ex.color }}>
                            {p.duration}s
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* ── BREATHING TAB content below ── */}
      {activeTab === "breathing" && (
      <>

      {/* No-device banner */}
      {!isStreaming && sessionPhase === "idle" && (
        <div className="flex items-center gap-3 p-3 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning">
          <Radio className="h-4 w-4 shrink-0" />
          Connect your Muse 2 to see your real brain response. Showing simulation without it.
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

          {/* Live stress chart */}
          <Card className="glass-card p-6 rounded-xl flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium">Live Stress Response</h3>
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
                  </defs>
                  {/* grid lines */}
                  <line x1="0" y1={CHART_H * 0.5} x2={CHART_W} y2={CHART_H * 0.5}
                    stroke="rgba(255,255,255,0.06)" strokeWidth="1" strokeDasharray="4 4" />
                  <line x1="0" y1={CHART_H} x2={CHART_W} y2={CHART_H}
                    stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
                  {/* area fill */}
                  <path d={stressArea} fill="url(#stress-area-grad)" />
                  {/* stress line */}
                  <path d={stressLine} fill="none" stroke="hsl(0,70%,60%)" strokeWidth="2.5" strokeLinejoin="round" />
                  {/* latest dot */}
                  {(() => {
                    const last = readings[readings.length - 1];
                    const x = CHART_W;
                    const y = CHART_H - (last.stress / 100) * CHART_H;
                    return (
                      <>
                        <circle cx={x} cy={y} r="5" fill="hsl(0,70%,60%)" />
                        <circle cx={x} cy={y} r="9" fill="none" stroke="hsl(0,70%,60%)" strokeWidth="1" opacity="0.4" />
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
                  <span className="opacity-50">⚡ simulation mode</span>
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
              <div className="grid grid-cols-3 gap-4 py-4 border-y border-border/20">
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
                  </defs>
                  <path d={stressArea} fill="url(#done-grad)" />
                  <path d={stressLine} fill="none" stroke="hsl(0,70%,60%)" strokeWidth="2.5" strokeLinejoin="round" />
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
        </div>
      )}

      </> /* end breathing tab */
      )}
    </main>
  );
}
