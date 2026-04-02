/**
 * MoodMusicPlayer
 *
 * Binaural beat player + ambient drone with mood-adaptive presets.
 * Auto-detects the right brainwave mode from EEG/emotion state.
 * Generates audio entirely on-device via Web Audio API — no external services needed.
 *
 * Modes (binaural beat frequency → target brainwave band):
 *  Sleep     — 3 Hz  → Delta  — aids N3 sleep onset (Marshall et al., 2006)
 *  Calm      — 10 Hz → Alpha  — lowers cortisol within 8 min (PMC 2024)
 *  Meditate  — 6 Hz  → Theta  — deepens mindfulness (Lagopoulos et al., 2009)
 *  Focus     — 40 Hz → Gamma  — +30% sustained attention (Jirakittayakorn 2017)
 *  Energy    — 20 Hz → Beta   — prefrontal alertness (Engel & Fries, 2010)
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { useMoodMusic, type MusicMode } from "@/hooks/use-mood-music";
import { recommendMusic, type EegMusicState } from "@/lib/eeg-music";
import {
  Headphones, Play, Pause, Volume2, Moon, Wind, Brain, Zap, Flame, Timer,
} from "lucide-react";

// ── Mode metadata ─────────────────────────────────────────────────────────────

interface ModeInfo {
  id: MusicMode;
  label: string;
  binauralHz: number;
  bandLabel: string;
  bandHz: string;
  benefit: string;
  science: string;
  color: string;
  Icon: React.ElementType;
}

const MODES: ModeInfo[] = [
  {
    id: "sleep",
    label: "Sleep",
    binauralHz: 3,
    bandLabel: "Delta",
    bandHz: "0.5–4 Hz",
    benefit: "Deep sleep induction",
    science: "3 Hz delta beats promote N3 sleep onset (Marshall et al., 2006). Play 30 min before bed.",
    color: "#7c3aed",
    Icon: Moon,
  },
  {
    id: "calm",
    label: "Calm",
    binauralHz: 10,
    bandLabel: "Alpha",
    bandHz: "8–12 Hz",
    benefit: "Stress & anxiety relief",
    science: "10 Hz alpha entrainment reduces cortisol & systolic BP within 8 min (PMC 12145584, 2024).",
    color: "#0891b2",
    Icon: Wind,
  },
  {
    id: "meditation",
    label: "Meditate",
    binauralHz: 6,
    bandLabel: "Theta",
    bandHz: "4–8 Hz",
    benefit: "Deep mindfulness",
    science: "6 Hz theta deepens meditation states and enhances creativity (Lagopoulos et al., 2009).",
    color: "#a78bfa",
    Icon: Brain,
  },
  {
    id: "focus",
    label: "Focus",
    binauralHz: 40,
    bandLabel: "Gamma",
    bandHz: "30–100 Hz",
    benefit: "+30% sustained attention",
    science: "40 Hz gamma significantly improves attention in ADHD and healthy adults (Jirakittayakorn & Wongsawat, 2017).",
    color: "#6366f1",
    Icon: Zap,
  },
  {
    id: "energy",
    label: "Energy",
    binauralHz: 20,
    bandLabel: "Beta",
    bandHz: "15–30 Hz",
    benefit: "Alertness & motivation",
    science: "20 Hz beta activates prefrontal cortex, sustaining arousal and executive function (Engel & Fries, 2010).",
    color: "#ea580c",
    Icon: Flame,
  },
];

// ── Animated waveform ─────────────────────────────────────────────────────────

function WaveformCanvas({
  isPlaying,
  color,
  beatHz,
}: {
  isPlaying: boolean;
  color: string;
  beatHz: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = (ts: number) => {
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      if (!isPlaying) {
        ctx.beginPath();
        ctx.strokeStyle = color + "35";
        ctx.lineWidth = 1.5;
        ctx.moveTo(0, H / 2);
        ctx.lineTo(W, H / 2);
        ctx.stroke();
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const t = ts / 1000;
      const amp = H * 0.3;
      const freq = 2.8; // visual cycles across canvas width

      // Wave L (left ear)
      ctx.beginPath();
      ctx.strokeStyle = color + "55";
      ctx.lineWidth = 1.5;
      for (let x = 0; x <= W; x += 2) {
        const phase = (x / W) * Math.PI * 2 * freq + t * beatHz * 0.4;
        const y = H / 2 + Math.sin(phase) * amp * 0.65;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Wave R (right ear, slightly different frequency → binaural effect)
      ctx.beginPath();
      ctx.strokeStyle = color + "bb";
      ctx.lineWidth = 2;
      for (let x = 0; x <= W; x += 2) {
        const phase = (x / W) * Math.PI * 2 * freq + t * (beatHz * 0.4 + 0.1);
        const y = H / 2 + Math.sin(phase) * amp;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Perceived interference beat (dashed)
      ctx.beginPath();
      ctx.strokeStyle = color + "40";
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      for (let x = 0; x <= W; x += 2) {
        const env = 0.5 + 0.5 * Math.sin(t * beatHz * 0.08 + (x / W) * Math.PI);
        const phase = (x / W) * Math.PI * 2 * freq + t * beatHz * 0.4;
        const y = H / 2 + Math.sin(phase) * amp * 1.2 * env;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [isPlaying, color, beatHz]);

  return (
    <canvas
      ref={canvasRef}
      width={340}
      height={68}
      className="w-full"
      style={{ background: "transparent", display: "block" }}
    />
  );
}

// ── Session timer ─────────────────────────────────────────────────────────────

function useSessionTimer(running: boolean) {
  const [secs, setSecs] = useState(0);
  useEffect(() => {
    if (!running) { setSecs(0); return; }
    const id = setInterval(() => setSecs(s => s + 1), 1000);
    return () => clearInterval(id);
  }, [running]);
  return `${String(Math.floor(secs / 60)).padStart(2, "0")}:${String(secs % 60).padStart(2, "0")}`;
}

// ── Main component ────────────────────────────────────────────────────────────

interface MoodMusicPlayerProps {
  /** Optional: pass live EEG/emotion state for auto-mode suggestion */
  emotion?: string;
  isStreaming?: boolean;
  eegState?: Partial<EegMusicState>;
  className?: string;
}

export function MoodMusicPlayer({
  emotion,
  isStreaming,
  eegState,
  className = "",
}: MoodMusicPlayerProps) {
  const music = useMoodMusic(emotion, isStreaming);
  const sessionTimer = useSessionTimer(music.isPlaying);

  const activeMode = MODES.find(m => m.id === music.activeMode) ?? MODES[1];

  // Auto-suggest from full EEG state
  const suggestion = eegState?.dominantBand
    ? recommendMusic({
        dominantBand: eegState.dominantBand!,
        arousal: eegState.arousal ?? 0.5,
        stress: eegState.stress ?? 0.3,
        focus: eegState.focus ?? 0.5,
        emotion: eegState.emotion ?? emotion ?? "neutral",
      })
    : null;
  const suggestedMode = suggestion ? MODES.find(m => m.id === suggestion.category) : null;

  return (
    <div
      className={`rounded-2xl border border-border bg-card overflow-hidden ${className}`}
    >
      {/* ── Header gradient section ── */}
      <div
        className="px-5 pt-5 pb-4"
        style={{
          background: `linear-gradient(135deg, ${activeMode.color}18 0%, transparent 70%)`,
          borderBottom: `1px solid ${activeMode.color}20`,
        }}
      >
        {/* Title row */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2.5">
            <div
              className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0"
              style={{ background: activeMode.color + "20", border: `1px solid ${activeMode.color}40` }}
            >
              <activeMode.Icon className="h-4.5 w-4.5" style={{ color: activeMode.color }} />
            </div>
            <div>
              <p className="text-sm font-bold text-foreground leading-tight">
                {music.isEnabled ? activeMode.label : "Binaural Beats"}
              </p>
              <p className="text-[10px] leading-snug" style={{ color: activeMode.color }}>
                {activeMode.bandLabel} · {activeMode.binauralHz} Hz · {activeMode.bandHz}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {music.isPlaying && (
              <div className="flex items-center gap-1 text-[10px] font-mono text-muted-foreground">
                <Timer className="h-3 w-3" />
                {sessionTimer}
              </div>
            )}
            {music.isPlaying && (
              <div className="flex gap-0.5">
                {[1, 2, 3].map(i => (
                  <div
                    key={i}
                    className="w-0.5 rounded-full animate-pulse"
                    style={{
                      height: `${8 + i * 4}px`,
                      background: activeMode.color,
                      animationDelay: `${i * 0.15}s`,
                    }}
                  />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Waveform */}
        <WaveformCanvas
          isPlaying={music.isPlaying}
          color={activeMode.color}
          beatHz={activeMode.binauralHz}
        />
      </div>

      {/* ── Controls ── */}
      <div className="px-5 py-4 space-y-4">

        {/* Play + Volume */}
        <div className="flex items-center gap-3">
          <button
            onClick={music.isEnabled ? music.toggle : music.enable}
            className="w-11 h-11 rounded-full flex items-center justify-center shrink-0 transition-all active:scale-95"
            style={music.isPlaying
              ? { background: activeMode.color + "20", border: `2px solid ${activeMode.color}` }
              : { background: activeMode.color, border: `2px solid ${activeMode.color}` }
            }
            aria-label={music.isPlaying ? "Pause" : "Play"}
          >
            {music.isPlaying
              ? <Pause className="h-4 w-4" style={{ color: activeMode.color }} />
              : <Play className="h-4 w-4 text-white ml-0.5" />
            }
          </button>

          <div className="flex-1 flex items-center gap-2">
            <Volume2 className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
            <input
              type="range"
              min={0.01}
              max={0.45}
              step={0.01}
              value={music.volume}
              onChange={e => music.setVolume(parseFloat(e.target.value))}
              className="flex-1 h-1.5 rounded-full cursor-pointer"
              style={{ accentColor: activeMode.color }}
              aria-label="Volume"
            />
            <span className="text-[10px] font-mono text-muted-foreground w-7 text-right">
              {Math.round(music.volume * 222)}%
            </span>
          </div>
        </div>

        {/* Headphones warning */}
        <div className="flex items-start gap-2 p-2.5 rounded-xl bg-amber-500/8 border border-amber-500/20">
          <Headphones className="h-3.5 w-3.5 text-amber-400 shrink-0 mt-0.5" />
          <p className="text-[10px] text-amber-300/80 leading-snug">
            <span className="font-semibold text-amber-300">Headphones required</span> — binaural beats deliver different frequencies to each ear. Won't work on speakers.
          </p>
        </div>

        {/* Auto-suggest banner */}
        {suggestedMode && suggestedMode.id !== music.activeMode && (
          <button
            onClick={() => music.setMode(suggestedMode.id)}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-xl border transition-all active:scale-[0.98] text-left"
            style={{ borderColor: suggestedMode.color + "40", background: suggestedMode.color + "08" }}
          >
            <suggestedMode.Icon className="h-3.5 w-3.5 shrink-0" style={{ color: suggestedMode.color }} />
            <div className="flex-1 min-w-0">
              <p className="text-[10px] font-semibold" style={{ color: suggestedMode.color }}>
                Brain state suggests: {suggestedMode.label}
              </p>
              <p className="text-[10px] text-muted-foreground truncate">{suggestion?.reason}</p>
            </div>
            <span className="text-[10px] text-muted-foreground shrink-0">Switch →</span>
          </button>
        )}

        {/* Mode selector */}
        <div className="space-y-2">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">Brainwave Mode</p>
          <div className="grid grid-cols-5 gap-1.5">
            {MODES.map(mode => {
              const active = music.activeMode === mode.id;
              return (
                <button
                  key={mode.id}
                  onClick={() => music.setMode(mode.id)}
                  title={`${mode.label} — ${mode.binauralHz} Hz ${mode.bandLabel}`}
                  className="flex flex-col items-center gap-1 py-2.5 px-1 rounded-xl border transition-all active:scale-95"
                  style={active
                    ? { borderColor: mode.color, background: mode.color + "18" }
                    : { borderColor: "hsl(var(--border))" }
                  }
                >
                  <mode.Icon
                    className="h-4 w-4"
                    style={{ color: active ? mode.color : "hsl(var(--muted-foreground))" }}
                  />
                  <span
                    className="text-[9px] font-semibold leading-none"
                    style={{ color: active ? mode.color : "hsl(var(--muted-foreground))" }}
                  >
                    {mode.label}
                  </span>
                  <span
                    className="text-[8px] leading-none opacity-60 font-mono"
                    style={{ color: active ? mode.color : "hsl(var(--muted-foreground))" }}
                  >
                    {mode.binauralHz}Hz
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Science callout */}
        <div
          className="rounded-xl px-3.5 py-3 space-y-1.5"
          style={{ background: activeMode.color + "0c", border: `1px solid ${activeMode.color}25` }}
        >
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: activeMode.color }} />
            <p className="text-[11px] font-semibold" style={{ color: activeMode.color }}>
              {activeMode.benefit}
            </p>
          </div>
          <p className="text-[10px] text-muted-foreground leading-relaxed">{activeMode.science}</p>
        </div>

      </div>
    </div>
  );
}
