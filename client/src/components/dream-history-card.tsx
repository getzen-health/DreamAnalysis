import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { BookOpen, Moon, Sparkles, Search, ChevronDown, ChevronUp, AlertTriangle } from "lucide-react";
import { resolveUrl } from "@/lib/queryClient";
import {
  isNightmare,
  isLucid,
  hasInsight,
  entryQualityBand,
  bandColorClass,
  formatEntryDate,
  formatEntryDatetime,
  truncateDreamText,
  topThemes,
  applyFilter,
  searchDreams,
  sortNewest,
  computeStats,
  type DreamEntry,
  type DreamFilter,
} from "@/lib/dream-history";

interface Props {
  userId: string;
}

const FILTER_LABELS: Record<DreamFilter, string> = {
  all:          "All",
  nightmares:   "Nightmares",
  lucid:        "Lucid",
  "with-insight": "With Insight",
};

// ── Single expanded dream detail ─────────────────────────────────────────────

function DreamDetail({ entry }: { entry: DreamEntry }) {
  const band = entryQualityBand(entry);

  return (
    <div className="mt-3 space-y-2 text-xs border-t border-white/10 pt-3">
      {/* Full dream text */}
      <p className="text-muted-foreground leading-relaxed whitespace-pre-line">
        {entry.dreamText}
      </p>

      {/* Emotional arc */}
      {entry.emotionalArc && (
        <div className="flex gap-2">
          <span className="text-muted-foreground/60 shrink-0">Arc:</span>
          <span className="text-muted-foreground italic">{entry.emotionalArc}</span>
        </div>
      )}

      {/* Key insight */}
      {entry.keyInsight && (
        <div className="bg-secondary/10 border border-secondary/20 rounded px-2 py-1.5">
          <p className="text-secondary">&ldquo;{entry.keyInsight}&rdquo;</p>
        </div>
      )}

      {/* Symbols */}
      {(entry.symbols ?? []).length > 0 && (
        <div className="flex flex-wrap gap-1">
          {(entry.symbols ?? []).map((s) => (
            <span key={s} className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary/80">
              {s}
            </span>
          ))}
        </div>
      )}

      {/* Stats row */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-muted-foreground/70">
        {entry.sleepQuality != null && (
          <span>Sleep quality: {entry.sleepQuality}%</span>
        )}
        {entry.lucidityScore != null && entry.lucidityScore > 0 && (
          <span>Lucidity: {entry.lucidityScore}%</span>
        )}
        {entry.sleepDuration != null && (
          <span>Duration: {entry.sleepDuration.toFixed(1)} h</span>
        )}
        {band && (
          <span className={bandColorClass(band)}>Quality: {band}</span>
        )}
        {entry.threatSimulationIndex != null && (
          <span>Threat index: {(entry.threatSimulationIndex * 100).toFixed(0)}%</span>
        )}
      </div>

      {/* Timestamp */}
      <p className="text-[10px] text-muted-foreground/40">{formatEntryDatetime(entry.timestamp)}</p>
    </div>
  );
}

// ── Single dream row ─────────────────────────────────────────────────────────

function DreamRow({ entry }: { entry: DreamEntry }) {
  const [expanded, setExpanded] = useState(false);
  const nightmare = isNightmare(entry);
  const lucid = isLucid(entry);
  const themes = topThemes(entry);

  return (
    <div
      className={`rounded-lg p-3 border transition-colors cursor-pointer
        ${nightmare
          ? "border-red-500/20 bg-red-500/5 hover:bg-red-500/10"
          : lucid
          ? "border-cyan-500/20 bg-cyan-500/5 hover:bg-cyan-500/10"
          : "border-white/10 bg-white/5 hover:bg-white/8"}`}
      onClick={() => setExpanded((p) => !p)}
    >
      <div className="flex items-start gap-2">
        {/* Icon */}
        <div className="mt-0.5 shrink-0">
          {nightmare ? (
            <AlertTriangle className="h-3.5 w-3.5 text-red-400" />
          ) : lucid ? (
            <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
          ) : (
            <Moon className="h-3.5 w-3.5 text-secondary/70" />
          )}
        </div>

        {/* Main content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[11px] text-muted-foreground/60">
              {formatEntryDate(entry.timestamp)}
            </span>
            <div className="flex items-center gap-1">
              {nightmare && (
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-red-500/15 text-red-400">nightmare</span>
              )}
              {lucid && (
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-cyan-500/15 text-cyan-400">lucid</span>
              )}
              {expanded ? (
                <ChevronUp className="h-3 w-3 text-muted-foreground/40" />
              ) : (
                <ChevronDown className="h-3 w-3 text-muted-foreground/40" />
              )}
            </div>
          </div>

          <p className="text-xs text-foreground/90 leading-relaxed mt-0.5">
            {expanded ? entry.dreamText : truncateDreamText(entry.dreamText)}
          </p>

          {themes.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-1.5">
              {themes.map((t) => (
                <span key={t} className="text-[10px] px-1.5 py-0.5 rounded-full bg-secondary/15 text-secondary capitalize">
                  {t}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {expanded && <DreamDetail entry={entry} />}
    </div>
  );
}

// ── Main card ────────────────────────────────────────────────────────────────

export function DreamHistoryCard({ userId }: Props) {
  const [filter, setFilter] = useState<DreamFilter>("all");
  const [query, setQuery] = useState("");

  const { data: rawData, isLoading } = useQuery<DreamEntry[]>({
    queryKey: ["dream-analysis", userId],
    queryFn: async () => {
      const res = await fetch(resolveUrl(`/api/dream-analysis/${userId}`));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    },
    staleTime: 2 * 60 * 1000,
  });

  // Defensive: API may return non-array on error or partial mock
  const raw: DreamEntry[] = Array.isArray(rawData) ? rawData : [];
  const sorted = sortNewest(raw);
  const stats = computeStats(sorted);

  const visible = searchDreams(applyFilter(sorted, filter), query);

  return (
    <Card className="glass-card p-5 hover-glow border-secondary/20">
      <CardHeader className="p-0 pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <BookOpen className="h-4 w-4 text-secondary" />
          Dream Journal
          {!isLoading && (
            <span className="ml-auto text-[10px] text-muted-foreground">
              {stats.total} {stats.total === 1 ? "entry" : "entries"} · {stats.nightmares} nightmare{stats.nightmares !== 1 ? "s" : ""} · {stats.lucid} lucid
            </span>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent className="p-0 space-y-3">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground/50 pointer-events-none" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search dreams, themes, insights…"
            className="pl-8 h-8 text-xs bg-white/5 border-white/10 focus:border-secondary/40"
          />
        </div>

        {/* Filter tabs */}
        <div className="flex flex-wrap gap-1.5">
          {(Object.keys(FILTER_LABELS) as DreamFilter[]).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`text-[10px] px-2.5 py-1 rounded-full border transition-colors
                ${filter === f
                  ? "bg-secondary/20 border-secondary/40 text-secondary"
                  : "border-white/10 text-muted-foreground hover:border-white/20"}`}
            >
              {FILTER_LABELS[f]}
              {f !== "all" && (
                <span className="ml-1 opacity-60">
                  {f === "nightmares" ? stats.nightmares
                   : f === "lucid" ? stats.lucid
                   : stats.withInsight}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Entries */}
        {isLoading ? (
          <p className="text-xs text-muted-foreground py-4 text-center">Loading dream history…</p>
        ) : visible.length === 0 ? (
          <p className="text-xs text-muted-foreground py-4 text-center">
            {raw.length === 0 ? "No dream entries yet." : "No entries match your filter."}
          </p>
        ) : (
          <div className="space-y-2 max-h-[480px] overflow-y-auto pr-1">
            {visible.map((entry) => (
              <DreamRow key={entry.id} entry={entry} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
