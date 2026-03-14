/**
 * sleep-story-player.tsx
 *
 * Plays a procedurally generated ambient soundscape using Web Audio API.
 *
 * Two operating modes:
 *   - Timer mode  (eegConnected === false / undefined):
 *       Fade-out starts after a user-selected preset (15 / 30 / 45 min).
 *   - Auto mode   (eegConnected === true):
 *       Listens for a `sleep-stage-transition` CustomEvent on window
 *       (dispatched by the WebSocket handler in the ML backend bridge).
 *       When the event carries { from: "N1", to: "N2" }, the fade begins
 *       and sleep latency is logged.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import {
  Play,
  Pause,
  Volume2,
  VolumeX,
  Clock,
  BrainCircuit,
  Moon,
  Check,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { fadeOutAudio, cancelFade } from "@/lib/audio-fade";
import { createAmbientAudio, type AmbientType, type AmbientHandle } from "@/lib/ambient-audio";

// ─── Types ────────────────────────────────────────────────────────────────────

interface SleepStageTransitionEvent extends CustomEvent {
  detail: {
    from: string;
    to: string;
    timestamp?: number;
  };
}

interface SleepLatencyRecord {
  storyTitle: string;
  latencyMs: number;
  recordedAt: string; // ISO timestamp
}

// LocalStorage key for persisting latency records across page refreshes
const LATENCY_STORAGE_KEY = "ndw_sleep_story_latency";

function loadLatencyRecords(): SleepLatencyRecord[] {
  try {
    return JSON.parse(localStorage.getItem(LATENCY_STORAGE_KEY) ?? "[]");
  } catch {
    return [];
  }
}

function saveLatencyRecord(record: SleepLatencyRecord): void {
  try {
    const records = loadLatencyRecords();
    records.unshift(record); // newest first
    localStorage.setItem(LATENCY_STORAGE_KEY, JSON.stringify(records.slice(0, 30)));
  } catch {
    // localStorage write failure is non-fatal
  }
}

/** Human-readable duration from milliseconds */
function formatLatency(ms: number): string {
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  if (minutes === 0) return `${seconds}s`;
  if (seconds === 0) return `${minutes}m`;
  return `${minutes}m ${seconds}s`;
}

// ─── Timer preset options ─────────────────────────────────────────────────────

const TIMER_PRESETS = [
  { label: "15 min", ms: 15 * 60 * 1000 },
  { label: "30 min", ms: 30 * 60 * 1000 },
  { label: "45 min", ms: 45 * 60 * 1000 },
];

// ─── Props ────────────────────────────────────────────────────────────────────

export interface SleepStoryPlayerProps {
  audioType: AmbientType;
  title: string;
  eegConnected?: boolean;
  onSleepDetected?: (latencyMs: number) => void;
}

// ─── Component ────────────────────────────────────────────────────────────────

export function SleepStoryPlayer({
  audioType,
  title,
  eegConnected = false,
  onSleepDetected,
}: SleepStoryPlayerProps) {
  const ambientRef = useRef<AmbientHandle | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(0.8);
  const [isMuted, setIsMuted] = useState(false);
  const [elapsed, setElapsed] = useState(0); // seconds since play started

  // Timer mode state
  const [selectedPreset, setSelectedPreset] = useState(TIMER_PRESETS[1]); // 30 min default
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const elapsedIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // EEG Auto mode state
  const [fadeStarted, setFadeStarted] = useState(false);
  const [sleepLatencyMs, setSleepLatencyMs] = useState<number | null>(null);
  const playStartRef = useRef<number | null>(null); // timestamp when playback started

  // Morning-summary latency record (last session)
  const [lastRecord, setLastRecord] = useState<SleepLatencyRecord | null>(() => {
    const records = loadLatencyRecords();
    return records.find((r) => r.storyTitle === title) ?? null;
  });

  // ── Cleanup on unmount or audioType change ─────────────────────────────────
  useEffect(() => {
    return () => {
      if (ambientRef.current) {
        ambientRef.current.stop();
        ambientRef.current = null;
      }
      if (timerRef.current) clearTimeout(timerRef.current);
      if (elapsedIntervalRef.current) clearInterval(elapsedIntervalRef.current);
    };
  }, [audioType]);

  // ── Sync volume changes to gain node ───────────────────────────────────────
  useEffect(() => {
    if (ambientRef.current && !fadeStarted) {
      ambientRef.current.gainNode.gain.setValueAtTime(
        isMuted ? 0 : volume,
        ambientRef.current.audioContext.currentTime,
      );
    }
  }, [volume, isMuted, fadeStarted]);

  // ── Timer mode: schedule fade after preset ─────────────────────────────────
  useEffect(() => {
    if (!isPlaying || eegConnected || fadeStarted) return;

    // Clear any existing timer when preset changes while playing
    if (timerRef.current) clearTimeout(timerRef.current);

    timerRef.current = setTimeout(() => {
      const ambient = ambientRef.current;
      if (ambient) {
        setFadeStarted(true);
        fadeOutAudio(ambient.gainNode, 90_000, () => {
          ambient.stop();
          ambientRef.current = null;
          setIsPlaying(false);
        });
      }
    }, selectedPreset.ms);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [isPlaying, eegConnected, fadeStarted, selectedPreset]);

  // ── EEG Auto mode: listen for sleep-stage-transition event ─────────────────
  useEffect(() => {
    if (!eegConnected) return;

    const handleTransition = (evt: Event) => {
      const e = evt as SleepStageTransitionEvent;
      if (e.detail?.from === "N1" && e.detail?.to === "N2") {
        const ambient = ambientRef.current;
        if (ambient && isPlaying && !fadeStarted) {
          setFadeStarted(true);

          // Compute sleep latency
          const latencyMs =
            playStartRef.current != null
              ? Date.now() - playStartRef.current
              : 0;

          setSleepLatencyMs(latencyMs);
          onSleepDetected?.(latencyMs);

          // Persist record
          const record: SleepLatencyRecord = {
            storyTitle: title,
            latencyMs,
            recordedAt: new Date().toISOString(),
          };
          saveLatencyRecord(record);
          setLastRecord(record);

          fadeOutAudio(ambient.gainNode, 90_000, () => {
            ambient.stop();
            ambientRef.current = null;
            setIsPlaying(false);
          });
        }
      }
    };

    window.addEventListener("sleep-stage-transition", handleTransition);
    return () => window.removeEventListener("sleep-stage-transition", handleTransition);
  }, [eegConnected, isPlaying, fadeStarted, title, onSleepDetected]);

  // ── Transport controls ──────────────────────────────────────────────────────
  const togglePlayPause = useCallback(() => {
    if (isPlaying) {
      // Stop
      if (ambientRef.current) {
        cancelFade(ambientRef.current.gainNode);
        ambientRef.current.stop();
        ambientRef.current = null;
      }
      if (timerRef.current) clearTimeout(timerRef.current);
      if (elapsedIntervalRef.current) clearInterval(elapsedIntervalRef.current);
      setIsPlaying(false);
      setElapsed(0);
    } else {
      // Start fresh
      setFadeStarted(false);
      setSleepLatencyMs(null);
      setElapsed(0);

      const ambient = createAmbientAudio(audioType);
      ambient.gainNode.gain.setValueAtTime(
        isMuted ? 0 : volume,
        ambient.audioContext.currentTime,
      );
      ambient.start();
      ambientRef.current = ambient;

      playStartRef.current = Date.now();
      setIsPlaying(true);

      // Track elapsed time for display
      elapsedIntervalRef.current = setInterval(() => {
        setElapsed((prev) => prev + 1);
      }, 1000);
    }
  }, [isPlaying, isMuted, volume, audioType]);

  const handleVolumeChange = useCallback((value: number[]) => {
    setVolume(value[0] / 100);
    setIsMuted(false);
  }, []);

  const toggleMute = useCallback(() => setIsMuted((m) => !m), []);

  // ── Time formatting ─────────────────────────────────────────────────────────
  function fmtTime(sec: number): string {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  }

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <Card className="glass-card p-5 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center shrink-0">
            <Moon className="h-5 w-5 text-primary" />
          </div>
          <div>
            <p className="text-sm font-semibold leading-tight">{title}</p>
            <p className="text-[11px] text-muted-foreground mt-0.5">
              {eegConnected ? "Auto mode — EEG fade" : "Timer mode"}
            </p>
          </div>
        </div>

        {/* Mode badge */}
        <Badge
          variant="secondary"
          className={`text-[10px] gap-1 ${eegConnected ? "border-primary/40 text-primary" : ""}`}
        >
          {eegConnected ? (
            <>
              <BrainCircuit className="h-3 w-3" />
              EEG Auto
            </>
          ) : (
            <>
              <Clock className="h-3 w-3" />
              Timer
            </>
          )}
        </Badge>
      </div>

      {/* Elapsed time display */}
      <div className="flex items-center justify-between text-[11px] text-muted-foreground font-mono">
        <span>{fmtTime(elapsed)}</span>
        <span>{isPlaying ? "playing" : "stopped"}</span>
      </div>

      {/* Transport + volume row */}
      <div className="flex items-center gap-3">
        {/* Play / pause */}
        <Button
          variant="ghost"
          size="icon"
          className="h-10 w-10 rounded-full bg-primary/10 hover:bg-primary/20 shrink-0"
          onClick={togglePlayPause}
          disabled={fadeStarted && isPlaying}
          aria-label={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? (
            <Pause className="h-5 w-5 text-primary" />
          ) : (
            <Play className="h-5 w-5 text-primary ml-0.5" />
          )}
        </Button>

        {/* Volume mute toggle */}
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 shrink-0"
          onClick={toggleMute}
          aria-label={isMuted ? "Unmute" : "Mute"}
        >
          {isMuted ? (
            <VolumeX className="h-4 w-4 text-muted-foreground" />
          ) : (
            <Volume2 className="h-4 w-4" />
          )}
        </Button>

        {/* Volume slider */}
        <Slider
          className="flex-1"
          min={0}
          max={100}
          step={1}
          value={[isMuted ? 0 : Math.round(volume * 100)]}
          onValueChange={handleVolumeChange}
          aria-label="Volume"
        />
      </div>

      {/* Timer preset selector (only shown in timer mode) */}
      {!eegConnected && (
        <div className="space-y-2">
          <p className="text-[11px] text-muted-foreground uppercase tracking-wider">
            Fade-out after
          </p>
          <div className="flex gap-2">
            {TIMER_PRESETS.map((preset) => (
              <button
                key={preset.label}
                onClick={() => {
                  setSelectedPreset(preset);
                  // If a timer was already running, restart it with new preset
                  if (isPlaying && timerRef.current) {
                    clearTimeout(timerRef.current);
                    setFadeStarted(false);
                  }
                }}
                className={`flex-1 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                  selectedPreset.ms === preset.ms
                    ? "bg-primary/15 border-primary/40 text-primary"
                    : "border-border/40 text-muted-foreground hover:border-border"
                }`}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Fade-in-progress indicator */}
      {fadeStarted && isPlaying && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground animate-pulse">
          <Moon className="h-3.5 w-3.5 text-primary" />
          Fading out gently over 90 seconds...
        </div>
      )}

      {/* Morning summary — sleep latency from this session */}
      {sleepLatencyMs !== null && (
        <div className="flex items-center gap-2 p-3 rounded-xl border border-primary/20 bg-primary/5 text-xs">
          <Check className="h-4 w-4 text-primary shrink-0" />
          <span>
            You fell asleep{" "}
            <span className="font-semibold text-primary">
              {formatLatency(sleepLatencyMs)}
            </span>{" "}
            into last night's story.
          </span>
        </div>
      )}

      {/* Last recorded latency (different session, same story) */}
      {sleepLatencyMs === null && lastRecord && (
        <div className="flex items-center gap-2 p-3 rounded-xl border border-border/30 bg-muted/10 text-[11px] text-muted-foreground">
          <Clock className="h-3.5 w-3.5 shrink-0" />
          <span>
            Last time you fell asleep{" "}
            <span className="font-medium text-foreground/80">
              {formatLatency(lastRecord.latencyMs)}
            </span>{" "}
            into this story.
          </span>
        </div>
      )}
    </Card>
  );
}
