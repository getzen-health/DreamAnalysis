/**
 * MoodTrends -- Emotion timeline page.
 *
 * Shows feelings over time as emoji-driven visual entries, NOT numeric scores.
 *
 * Sections:
 * 1. Hero — current emotion (large emoji + name + recency + source)
 * 2. Top 5 Emotions — horizontal stacked bar distribution
 * 3. Emotion Timeline — scrollable list of every entry, newest first
 * 4. Time-of-Day Pattern — dominant emotion per period (morning/afternoon/evening)
 * 5. Emotion Frequency — horizontal bars showing how often each emotion appeared
 *
 * Data: /api/brain/history/:userId?days=30, useCurrentEmotion() hook
 */

import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { pageTransition, cardVariants } from "@/lib/animations";
import { getParticipantId } from "@/lib/participant";
import { useCurrentEmotion } from "@/hooks/use-current-emotion";
import { sbGetGeneric } from "@/lib/supabase-store";
import { Sun, Sunset, Moon } from "lucide-react";

/* ---------- constants ---------- */

const EMOTION_COLORS: Record<string, string> = {
  happy: "#0891b2",
  sad: "#6366f1",
  angry: "#ea580c",
  fear: "#7c3aed",
  neutral: "#94a3b8",
  focused: "#0891b2",
  stressed: "#e879a8",
  relaxed: "#4ade80",
  peaceful: "#4ade80",
  anxious: "#d4a017",
  grateful: "#0891b2",
  tired: "#94a3b8",
  surprised: "#d4a017",
  surprise: "#d4a017",
};

const EMOTION_EMOJI: Record<string, string> = {
  happy: "\u{1F60A}",
  sad: "\u{1F622}",
  angry: "\u{1F620}",
  fear: "\u{1F628}",
  neutral: "\u{1F610}",
  focused: "\u{1F3AF}",
  stressed: "\u{1F630}",
  relaxed: "\u{1F60C}",
  peaceful: "\u{1F9D8}",
  anxious: "\u{1F61F}",
  grateful: "\u{1F64F}",
  tired: "\u{1F634}",
  surprised: "\u{1F632}",
  surprise: "\u{1F632}",
};

type TimeRange = "today" | "week" | "month";

/* ---------- types ---------- */

interface HistoryEntry {
  stress: number;
  happiness: number;
  focus: number;
  dominantEmotion: string;
  valence: number | null;
  timestamp: string;
}

/* ---------- helpers ---------- */

function getEmoji(emotion: string): string {
  return EMOTION_EMOJI[emotion.toLowerCase()] ?? "\u{1F610}";
}

function getColor(emotion: string): string {
  return EMOTION_COLORS[emotion.toLowerCase()] ?? "#94a3b8";
}

function filterByRange(entries: HistoryEntry[], range: TimeRange): HistoryEntry[] {
  const now = Date.now();
  return entries.filter((e) => {
    const ts = new Date(e.timestamp).getTime();
    switch (range) {
      case "today":
        return new Date(ts).toDateString() === new Date().toDateString();
      case "week":
        return now - ts < 7 * 86_400_000;
      case "month":
      default:
        return true;
    }
  });
}

function formatRelativeTime(isoTimestamp: string): string {
  const diff = Date.now() - new Date(isoTimestamp).getTime();
  const minutes = Math.floor(diff / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function formatTime(isoTimestamp: string): string {
  return new Date(isoTimestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function guessSource(entry: HistoryEntry): string {
  // Best-effort guess from available data
  if (entry.focus > 0.7) return "EEG";
  if (entry.valence !== null) return "voice";
  return "check-in";
}

/** Top N emotions by frequency */
function computeDistribution(
  entries: HistoryEntry[],
  topN: number,
): { emotion: string; count: number; pct: number }[] {
  const counts = new Map<string, number>();
  for (const e of entries) {
    if (!e.dominantEmotion) continue;
    const key = e.dominantEmotion.toLowerCase();
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const total = Array.from(counts.values()).reduce((a, b) => a + b, 0);
  if (total === 0) return [];

  return Array.from(counts.entries())
    .sort(([, a], [, b]) => b - a)
    .slice(0, topN)
    .map(([emotion, count]) => ({
      emotion,
      count,
      pct: Math.round((count / total) * 100),
    }));
}

/** Dominant emotion per time-of-day bucket */
function computePeriodDominants(
  entries: HistoryEntry[],
): { period: string; emotion: string; icon: "sun" | "sunset" | "moon" }[] {
  const buckets: Record<string, Map<string, number>> = {
    Morning: new Map(),
    Afternoon: new Map(),
    Evening: new Map(),
  };

  for (const e of entries) {
    if (!e.dominantEmotion) continue;
    const hour = new Date(e.timestamp).getHours();
    const key = e.dominantEmotion.toLowerCase();
    let bucket: string;
    if (hour >= 5 && hour < 12) bucket = "Morning";
    else if (hour >= 12 && hour < 18) bucket = "Afternoon";
    else bucket = "Evening";

    const map = buckets[bucket];
    map.set(key, (map.get(key) ?? 0) + 1);
  }

  const icons: Record<string, "sun" | "sunset" | "moon"> = {
    Morning: "sun",
    Afternoon: "sunset",
    Evening: "moon",
  };

  return Object.entries(buckets)
    .filter(([, map]) => map.size > 0)
    .map(([period, map]) => {
      let maxEmotion = "neutral";
      let maxCount = 0;
      for (const [em, count] of map) {
        if (count > maxCount) {
          maxCount = count;
          maxEmotion = em;
        }
      }
      return { period, emotion: maxEmotion, icon: icons[period] };
    });
}

/** Frequency of each emotion for horizontal bar chart */
function computeFrequency(
  entries: HistoryEntry[],
): { emotion: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const e of entries) {
    if (!e.dominantEmotion) continue;
    const key = e.dominantEmotion.toLowerCase();
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return Array.from(counts.entries())
    .sort(([, a], [, b]) => b - a)
    .map(([emotion, count]) => ({ emotion, count }));
}

/* ---------- sub-components ---------- */

function PeriodIcon({ name }: { name: string }) {
  switch (name) {
    case "sun":
      return <Sun className="h-5 w-5 text-amber-400" />;
    case "sunset":
      return <Sunset className="h-5 w-5 text-orange-400" />;
    case "moon":
      return <Moon className="h-5 w-5 text-indigo-400" />;
    default:
      return null;
  }
}

/* ---------- main component ---------- */

export default function MoodTrends() {
  const userId = getParticipantId();
  const [range, setRange] = useState<TimeRange>("week");
  const { emotion: currentEmotion } = useCurrentEmotion();

  const { data } = useQuery<HistoryEntry[]>({
    queryKey: [`/api/brain/history/${userId}?days=90`],
    queryFn: async () => {
      let all: HistoryEntry[] = [];
      // 1. Express API
      try {
        const res = await fetch(`/api/brain/history/${userId}?days=90`);
        if (res.ok) {
          const json = await res.json();
          if (Array.isArray(json)) all = json;
        }
      } catch { /* API unavailable */ }
      // 2. Supabase fallback
      try {
        const { getSupabase } = await import("@/lib/supabase-browser");
        const sb = await getSupabase();
        const since = new Date(Date.now() - 90 * 86400000).toISOString();
        const { data: rows } = await sb.from("emotion_history").select("*")
          .eq("user_id", userId).gte("created_at", since)
          .order("created_at", { ascending: false }).limit(2000);
        if (rows) {
          for (const r of rows as any[]) {
            all.push({
              stress: r.stress ?? 0,
              happiness: r.mood ?? 0,
              focus: r.focus ?? 0,
              dominantEmotion: r.dominant_emotion ?? "neutral",
              timestamp: r.created_at,
            } as HistoryEntry);
          }
        }
      } catch { /* Supabase unavailable */ }
      // 3. Supabase-backed local cache fallback
      const cached = sbGetGeneric<any[]>("ndw_emotion_history");
      if (Array.isArray(cached)) all.push(...cached);
      // Deduplicate by timestamp (within 3s)
      all.sort((a: any, b: any) => new Date(a.timestamp ?? a.created_at ?? 0).getTime() - new Date(b.timestamp ?? b.created_at ?? 0).getTime());
      const deduped: HistoryEntry[] = [];
      for (const entry of all) {
        const ts = new Date(entry.timestamp ?? (entry as any).created_at ?? 0).getTime();
        const lastTs = deduped.length > 0 ? new Date(deduped[deduped.length - 1].timestamp ?? 0).getTime() : 0;
        if (ts - lastTs > 3000 || deduped.length === 0) deduped.push(entry);
      }
      return deduped;
    },
    retry: false,
    staleTime: 60_000,
  });

  const filtered = useMemo(
    () => filterByRange(data ?? [], range),
    [data, range],
  );

  const distribution = useMemo(
    () => computeDistribution(filtered, 5),
    [filtered],
  );

  const periodDominants = useMemo(
    () => computePeriodDominants(filtered),
    [filtered],
  );

  const frequency = useMemo(
    () => computeFrequency(filtered),
    [filtered],
  );

  // Timeline entries: every individual entry, newest first
  const timelineEntries = useMemo(() => {
    return [...filtered]
      .filter((e) => e.dominantEmotion)
      .sort(
        (a, b) =>
          new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
      );
  }, [filtered]);

  const maxFreq = frequency.length > 0 ? frequency[0].count : 1;

  // Current emotion hero
  const heroEmotion = currentEmotion?.emotion?.toLowerCase() ?? null;
  const heroColor = heroEmotion ? getColor(heroEmotion) : "#94a3b8";

  const isEmpty = timelineEntries.length === 0 && !heroEmotion;

  return (
    <div className="max-w-2xl mx-auto px-4 py-6 space-y-5 pb-4">
      {/* Header */}
      <motion.div
        initial={pageTransition.initial}
        animate={pageTransition.animate}
        transition={pageTransition.transition}
      >
        <h1 className="text-xl font-bold tracking-tight text-foreground">
          How You Feel
        </h1>
        <p className="text-sm mt-1 text-muted-foreground">
          Your emotions over time
        </p>
      </motion.div>

      {/* Section 1: Hero — Current Emotion */}
      <motion.div
        className="rounded-2xl p-6 border border-border overflow-hidden relative"
        style={{
          background: heroEmotion
            ? `linear-gradient(135deg, ${heroColor}15, ${heroColor}08, transparent)`
            : undefined,
          boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
        }}
        initial={{ opacity: 0, y: 12, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      >
        {heroEmotion ? (
          <div className="flex flex-col items-center text-center">
            <motion.span
              className="text-6xl"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 260, damping: 20 }}
            >
              {getEmoji(heroEmotion)}
            </motion.span>
            <span
              className="text-2xl font-bold mt-3 capitalize"
              style={{ color: heroColor }}
            >
              {heroEmotion}
            </span>
            <span className="text-xs text-muted-foreground mt-2">
              {formatRelativeTime(currentEmotion!.timestamp)} via{" "}
              {currentEmotion!.source === "eeg"
                ? "EEG"
                : currentEmotion!.source === "voice"
                  ? "voice check-in"
                  : "manual check-in"}
            </span>
          </div>
        ) : (
          <div className="text-center py-4">
            <span className="text-5xl block mb-3 opacity-30">
              {"\u{1F610}"}
            </span>
            <p className="text-sm text-muted-foreground">
              Complete a check-in to see how you feel
            </p>
          </div>
        )}
      </motion.div>

      {/* Time range tabs */}
      <div className="flex gap-2">
        {(["today", "week", "month"] as TimeRange[]).map((r) => (
          <button
            key={r}
            onClick={() => setRange(r)}
            className={`flex-1 py-2 rounded-xl text-xs font-semibold transition-colors ${
              range === r
                ? "bg-primary text-primary-foreground"
                : "bg-muted/40 text-muted-foreground hover:bg-muted/60"
            }`}
          >
            {r === "today" ? "Today" : r === "week" ? "Week" : "Month"}
          </button>
        ))}
      </div>

      {/* Section 2: Top 5 Emotions — Stacked Bar */}
      {distribution.length > 0 && (
        <motion.div
          className="rounded-2xl p-4 border border-border bg-card"
          style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
          custom={1}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <span className="text-sm font-semibold text-foreground mb-3 block">
            Your Emotions
          </span>

          {/* Stacked bar */}
          <div className="flex w-full h-6 rounded-full overflow-hidden mb-3">
            {distribution.map((d, i) => (
              <motion.div
                key={d.emotion}
                className="h-full"
                style={{ backgroundColor: getColor(d.emotion) }}
                initial={{ width: 0 }}
                animate={{ width: `${d.pct}%` }}
                transition={{
                  duration: 0.6,
                  delay: i * 0.08,
                  ease: [0.22, 1, 0.36, 1],
                }}
              />
            ))}
          </div>

          {/* Legend */}
          <div className="flex flex-wrap gap-x-4 gap-y-1.5">
            {distribution.map((d) => (
              <div key={d.emotion} className="flex items-center gap-1.5">
                <span className="text-sm">{getEmoji(d.emotion)}</span>
                <span className="text-xs text-foreground capitalize">
                  {d.emotion}
                </span>
                <span className="text-[10px] text-muted-foreground">
                  {d.pct}%
                </span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Section 3: Emotion Timeline */}
      {timelineEntries.length > 0 && (
        <motion.div
          className="rounded-2xl p-4 border border-border bg-card"
          style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
          custom={2}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <span className="text-sm font-semibold text-foreground mb-3 block">
            Timeline
          </span>
          <div className="space-y-0 max-h-[360px] overflow-y-auto pr-1">
            {timelineEntries.map((entry, i) => {
              const em = entry.dominantEmotion.toLowerCase();
              const color = getColor(em);
              return (
                <motion.div
                  key={`${entry.timestamp}-${i}`}
                  className="flex items-center gap-3 py-2.5 border-l-2 pl-3"
                  style={{ borderColor: color }}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{
                    delay: Math.min(i * 0.03, 0.6),
                    duration: 0.3,
                    ease: "easeOut",
                  }}
                >
                  <span className="text-xl flex-shrink-0">
                    {getEmoji(em)}
                  </span>
                  <div className="flex-1 min-w-0">
                    <span className="text-sm font-medium text-foreground capitalize">
                      {em}
                    </span>
                  </div>
                  <span className="text-[11px] text-muted-foreground flex-shrink-0">
                    {formatTime(entry.timestamp)}
                  </span>
                  <span
                    className="text-[10px] px-1.5 py-0.5 rounded-full flex-shrink-0 font-medium"
                    style={{
                      backgroundColor: `${color}20`,
                      color,
                    }}
                  >
                    {guessSource(entry)}
                  </span>
                </motion.div>
              );
            })}
          </div>
        </motion.div>
      )}

      {/* Section 4: Time-of-Day Pattern */}
      {periodDominants.length > 0 && (
        <motion.div
          className="rounded-2xl p-4 border border-border bg-card"
          style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
          custom={3}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <span className="text-sm font-semibold text-foreground mb-3 block">
            Time of Day
          </span>
          <div className="grid grid-cols-3 gap-3">
            {periodDominants.map((p) => {
              const color = getColor(p.emotion);
              return (
                <div
                  key={p.period}
                  className="flex flex-col items-center gap-1.5 rounded-xl p-3"
                  style={{ backgroundColor: `${color}10` }}
                >
                  <PeriodIcon name={p.icon} />
                  <span className="text-[11px] text-muted-foreground">
                    {p.period}
                  </span>
                  <span className="text-2xl">{getEmoji(p.emotion)}</span>
                  <span
                    className="text-xs font-medium capitalize"
                    style={{ color }}
                  >
                    {p.emotion}
                  </span>
                </div>
              );
            })}
          </div>
        </motion.div>
      )}

      {/* Section 5: Emotion Frequency */}
      {frequency.length > 0 && (
        <motion.div
          className="rounded-2xl p-4 border border-border bg-card"
          style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
          custom={4}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <span className="text-sm font-semibold text-foreground mb-3 block">
            How Often
          </span>
          <div className="space-y-2.5">
            {frequency.map((f, i) => {
              const color = getColor(f.emotion);
              const widthPct = Math.max((f.count / maxFreq) * 100, 8);
              return (
                <div key={f.emotion} className="flex items-center gap-2.5">
                  <span className="text-lg flex-shrink-0 w-7 text-center">
                    {getEmoji(f.emotion)}
                  </span>
                  <span className="text-xs text-foreground capitalize w-16 flex-shrink-0">
                    {f.emotion}
                  </span>
                  <div className="flex-1 bg-muted/30 rounded-full h-3">
                    <motion.div
                      className="h-3 rounded-full"
                      style={{ backgroundColor: color }}
                      initial={{ width: 0 }}
                      animate={{ width: `${widthPct}%` }}
                      transition={{
                        duration: 0.7,
                        delay: i * 0.06,
                        ease: [0.22, 1, 0.36, 1],
                      }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground w-6 text-right flex-shrink-0">
                    {f.count}
                  </span>
                </div>
              );
            })}
          </div>
        </motion.div>
      )}

      {/* Empty state */}
      {isEmpty && (
        <div
          className="rounded-2xl p-8 border border-border bg-card text-center"
          style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
        >
          <span className="text-5xl block mb-3 opacity-30">
            {"\u{1F610}"}
          </span>
          <p className="text-sm text-muted-foreground">
            No emotion data yet. Complete a voice check-in to start tracking how
            you feel.
          </p>
        </div>
      )}
    </div>
  );
}
