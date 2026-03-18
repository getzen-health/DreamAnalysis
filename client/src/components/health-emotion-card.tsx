/**
 * HealthEmotionCard — passive emotion estimate from Apple Watch / Google Health data.
 *
 * Uses HR, HRV, sleep hours, and step count (already synced by use-health-sync) to
 * call POST /health-emotion/estimate on the ML backend and display a continuously
 * updated emotion state without any user action.
 *
 * Renders nothing when health data is unavailable (web platform, no permissions).
 */

import { useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { Heart, Activity, Moon, Footprints } from "lucide-react";
import { estimateHealthEmotion, type HealthEmotionResult } from "@/lib/ml-api";
import type { BiometricPayload } from "@/lib/health-sync";
import { resolveUrl } from "@/lib/queryClient";
import { getParticipantId } from "@/lib/participant";

// ── Helpers ───────────────────────────────────────────────────────────────────

const EMOTION_COLORS: Record<string, string> = {
  calm:      "text-sky-400",
  relaxed:   "text-cyan-400",
  happy:     "text-amber-400",
  excited:   "text-orange-400",
  energised: "text-lime-400",
  alert:     "text-yellow-400",
  neutral:   "text-muted-foreground",
  tense:     "text-orange-500",
  stressed:  "text-rose-400",
  drained:   "text-slate-400",
  active:    "text-lime-400",
  unknown:   "text-muted-foreground",
};

const EMOTION_ICONS: Record<string, string> = {
  calm:      "🌊",
  relaxed:   "🌿",
  happy:     "☀️",
  excited:   "⚡",
  energised: "✨",
  alert:     "👁️",
  neutral:   "⚖️",
  tense:     "🌡️",
  stressed:  "🔴",
  drained:   "🌑",
  active:    "🏃",
  unknown:   "❓",
};

function StressGauge({ value }: { value: number }) {
  // value 0–1. Colour shifts green → amber → rose.
  const pct = Math.round(value * 100);
  const color =
    value < 0.35 ? "#0891b2"   // ocean blue
    : value < 0.60 ? "#d4a017" // golden honey
    : "#e879a8";               // warm coral

  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-muted/40 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-[11px] font-mono text-muted-foreground/70 w-7 text-right shrink-0">
        {pct}%
      </span>
    </div>
  );
}

function CalmGauge({ value }: { value: number }) {
  // value 0–1 where 1 = perfectly calm.
  const calm = 1 - value;           // invert stress → calm
  const pct  = Math.round(calm * 100);
  const color =
    calm > 0.65 ? "#0891b2"
    : calm > 0.40 ? "#d4a017"
    : "#e879a8";

  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-muted/40 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-[11px] font-mono text-muted-foreground/70 w-7 text-right shrink-0">
        {pct}%
      </span>
    </div>
  );
}

function formatSyncTime(d: Date | null): string {
  if (!d) return "never";
  const diffMin = Math.round((Date.now() - d.getTime()) / 60_000);
  if (diffMin < 1) return "just now";
  if (diffMin === 1) return "1 min ago";
  if (diffMin < 60) return `${diffMin} min ago`;
  const h = Math.round(diffMin / 60);
  return h === 1 ? "1h ago" : `${h}h ago`;
}

// ── Component ─────────────────────────────────────────────────────────────────

interface HealthEmotionCardProps {
  payload: BiometricPayload;
  lastSyncAt: Date | null;
}

export function HealthEmotionCard({ payload, lastSyncAt }: HealthEmotionCardProps) {
  // Build the request object from the BiometricPayload fields that the backend uses
  const hr = payload.current_heart_rate ?? payload.resting_heart_rate;
  const hrv = payload.hrv_rmssd;     // may be undefined
  const resp = payload.respiratory_rate;
  // steps_last_hour proxy — use steps_today / hours_since_wake, capped at 2h
  let stepsLastHour: number | undefined;
  if (payload.steps_today != null && payload.hours_since_wake != null && payload.hours_since_wake > 0) {
    const hoursAwake = Math.min(payload.hours_since_wake, 16);
    stepsLastHour = Math.round(payload.steps_today / hoursAwake);
  }
  const sleep = payload.sleep_total_hours;

  const queryKey = [
    "health-emotion",
    Math.round((hr ?? 0) * 10),        // bucket changes by 0.1 BPM
    Math.round((hrv ?? 0) * 10),
    Math.round((sleep ?? 0) * 10),
    Math.round((stepsLastHour ?? 0) / 50) * 50,  // bucket every 50 steps
  ];

  const { data, isLoading } = useQuery<HealthEmotionResult>({
    queryKey,
    queryFn: () =>
      estimateHealthEmotion({
        hr_bpm: hr!,
        hrv_rmssd_ms: hrv,
        respiratory_rate: resp,
        steps_last_hour: stepsLastHour,
        sleep_hours: sleep,
        timestamp: new Date().toISOString(),
      }),
    enabled: hr != null && hr > 20,
    staleTime: 15 * 60 * 1000,   // sync cadence is 15 min; don't over-fetch
    retry: 1,
  });

  // Track last saved emotion+confidence to avoid duplicate saves on re-render
  const lastSavedRef = useRef<string | null>(null);

  useEffect(() => {
    if (!data) return;
    const key = `${data.emotion}:${data.confidence}`;
    if (lastSavedRef.current === key) return;
    lastSavedRef.current = key;

    // Persist to user_readings for model retraining (fire-and-forget)
    fetch(resolveUrl("/api/readings"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        userId: getParticipantId(),
        source: "health",
        emotion: data.emotion,
        valence: data.valence ?? null,
        arousal: data.arousal ?? null,
        stress: data.stress ?? null,
        confidence: data.confidence,
        modelType: "health-heuristic",
        features: {
          hr_bpm: hr,
          hrv_rmssd_ms: hrv ?? null,
          respiratory_rate: resp ?? null,
          steps_last_hour: stepsLastHour ?? null,
          sleep_hours: sleep ?? null,
        },
      }),
    }).catch(() => {
      // Silent — storage failure is not user-facing
    });
  }, [data, hr, hrv, resp, stepsLastHour, sleep]);

  if (!hr || hr <= 20) return null;

  return (
    <div className="rounded-2xl p-4 bg-card/70 border border-border/60 shadow-sm space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[0.08em]">
          From Your Watch
        </p>
        <span className="text-[10px] text-muted-foreground/40">
          {formatSyncTime(lastSyncAt)}
        </span>
      </div>

      {isLoading ? (
        <div className="space-y-2 animate-pulse">
          <div className="h-5 w-28 rounded bg-muted/30" />
          <div className="h-2 w-full rounded bg-muted/20" />
          <div className="h-3 w-48 rounded bg-muted/20" />
        </div>
      ) : data ? (
        <>
          {/* Emotion label + icon */}
          <div className="flex items-center gap-2">
            <span className="text-xl leading-none" aria-hidden="true">
              {EMOTION_ICONS[data.emotion] ?? EMOTION_ICONS.unknown}
            </span>
            <div>
              <p className={`text-[16px] font-bold capitalize leading-tight ${EMOTION_COLORS[data.emotion] ?? "text-foreground"}`}>
                {data.emotion}
              </p>
              <p className="text-[10px] text-muted-foreground/60 leading-tight">
                {Math.round(data.confidence * 100)}% confidence · passive estimate
              </p>
            </div>
          </div>

          {/* "Your watch says…" human-readable sentence */}
          <p className="text-[12px] text-muted-foreground leading-snug">
            {data.watch_says}
          </p>

          {/* Stress / Calm gauge */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-muted-foreground/60 uppercase tracking-wide">Stress</span>
              <span className="text-[10px] text-muted-foreground/60 uppercase tracking-wide">Calm</span>
            </div>
            <StressGauge value={data.stress} />
            <CalmGauge value={data.stress} />
          </div>

          {/* Raw metric chips */}
          <div className="flex flex-wrap gap-2">
            <div className="flex items-center gap-1 px-2 py-1 rounded-lg bg-muted/25 border border-border/30">
              <Heart className="h-3 w-3 text-rose-400 shrink-0" aria-hidden="true" />
              <span className="text-[11px] font-mono text-foreground/80">
                {Math.round(hr)} BPM
              </span>
            </div>
            {hrv != null && hrv > 0 && (
              <div className="flex items-center gap-1 px-2 py-1 rounded-lg bg-muted/25 border border-border/30">
                <Activity className="h-3 w-3 text-cyan-400 shrink-0" aria-hidden="true" />
                <span className="text-[11px] font-mono text-foreground/80">
                  {Math.round(hrv)} ms HRV
                </span>
              </div>
            )}
            {sleep != null && sleep > 0 && (
              <div className="flex items-center gap-1 px-2 py-1 rounded-lg bg-muted/25 border border-border/30">
                <Moon className="h-3 w-3 text-violet-400 shrink-0" aria-hidden="true" />
                <span className="text-[11px] font-mono text-foreground/80">
                  {sleep.toFixed(1)}h sleep
                </span>
              </div>
            )}
            {stepsLastHour != null && stepsLastHour > 0 && (
              <div className="flex items-center gap-1 px-2 py-1 rounded-lg bg-muted/25 border border-border/30">
                <Footprints className="h-3 w-3 text-lime-400 shrink-0" aria-hidden="true" />
                <span className="text-[11px] font-mono text-foreground/80">
                  ~{stepsLastHour.toLocaleString()} steps/h
                </span>
              </div>
            )}
          </div>
        </>
      ) : (
        <p className="text-[12px] text-muted-foreground/60">
          Unable to estimate — check ML backend connection.
        </p>
      )}
    </div>
  );
}
