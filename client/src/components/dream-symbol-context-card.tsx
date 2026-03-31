import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Sparkles } from "lucide-react";
import { resolveUrl } from "@/lib/queryClient";
import {
  buildSymbolContextMap,
  symbolMood,
  sortedSymbols,
  symbolSummary,
  MOOD_COLOR,
  MOOD_BG,
  MOOD_LABEL,
  type DreamEntryForSymbol,
  type SymbolSortKey,
} from "@/lib/dream-symbol-context";

interface Props {
  userId: string;
}

const SORT_OPTIONS: { key: SymbolSortKey; label: string }[] = [
  { key: "frequency", label: "Most common" },
  { key: "recent",    label: "Recent" },
  { key: "darkest",   label: "Darkest" },
  { key: "brightest", label: "Brightest" },
];

function SymbolRow({ ctx }: { ctx: ReturnType<typeof sortedSymbols>[0] }) {
  const [expanded, setExpanded] = useState(false);
  const mood  = symbolMood(ctx);
  const summary = symbolSummary(ctx);

  return (
    <div
      className={`rounded-lg p-2.5 border cursor-pointer transition-colors ${MOOD_BG[mood]}`}
      onClick={() => setExpanded((p) => !p)}
    >
      <div className="flex items-center gap-2">
        {/* Symbol name + count */}
        <span className="text-xs font-medium capitalize flex-1">{ctx.symbol}</span>
        <span className="text-[10px] text-muted-foreground/60">×{ctx.count}</span>
        <span className={`text-[10px] px-1.5 py-0.5 rounded-full border ${MOOD_BG[mood]} ${MOOD_COLOR[mood]}`}>
          {MOOD_LABEL[mood]}
        </span>
      </div>

      {/* Top themes as mini-chips */}
      {ctx.topThemes.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-1.5">
          {ctx.topThemes.map((t) => (
            <span key={t} className="text-[10px] px-1.5 py-0.5 rounded-full bg-white/8 text-muted-foreground/70 capitalize">
              {t}
            </span>
          ))}
        </div>
      )}

      {/* Expanded detail */}
      {expanded && (
        <div className="mt-2 space-y-1.5 border-t border-white/10 pt-2">
          <p className="text-[11px] text-muted-foreground/70 leading-relaxed">{summary}</p>

          {ctx.topArcs.length > 0 && (
            <div>
              <p className="text-[10px] text-muted-foreground/50 mb-1">Common arcs:</p>
              {ctx.topArcs.map((arc) => (
                <p key={arc} className="text-[10px] text-muted-foreground italic">"{arc}"</p>
              ))}
            </div>
          )}

          <div className="flex gap-4 text-[10px] text-muted-foreground/50">
            {ctx.avgTsi > 0 && (
              <span>Threat: {Math.round(ctx.avgTsi * 100)}%</span>
            )}
            {ctx.avgLucidity > 0 && (
              <span>Lucidity: {Math.round(ctx.avgLucidity)}%</span>
            )}
            <span>Last seen: {new Date(ctx.lastSeen).toLocaleDateString("en-US", { month: "short", day: "numeric" })}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export function DreamSymbolContextCard({ userId }: Props) {
  const [sortBy, setSortBy] = useState<SymbolSortKey>("frequency");

  const { data: rawData, isLoading } = useQuery<DreamEntryForSymbol[]>({
    queryKey: ["dream-analysis", userId],
    queryFn: async () => {
      const res = await fetch(resolveUrl(`/api/dream-analysis/${userId}`));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    },
    staleTime: 2 * 60 * 1000,
  });

  const raw: DreamEntryForSymbol[] = Array.isArray(rawData) ? rawData : [];
  const contextMap = buildSymbolContextMap(raw);
  const symbols = sortedSymbols(contextMap, sortBy, 8);

  return (
    <Card className="glass-card p-5 hover-glow border-secondary/20">
      <CardHeader className="p-0 pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-secondary" />
          Personal Dream Dictionary
          {!isLoading && contextMap.size > 0 && (
            <span className="ml-auto text-[10px] text-muted-foreground">
              {contextMap.size} symbol{contextMap.size !== 1 ? "s" : ""}
            </span>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent className="p-0 space-y-3">
        {/* Sort tabs */}
        {symbols.length > 0 && (
          <div className="flex gap-1.5 flex-wrap">
            {SORT_OPTIONS.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setSortBy(key)}
                className={`text-[10px] px-2 py-1 rounded-full border transition-colors
                  ${sortBy === key
                    ? "bg-secondary/20 border-secondary/40 text-secondary"
                    : "border-white/10 text-muted-foreground hover:border-white/20"}`}
              >
                {label}
              </button>
            ))}
          </div>
        )}

        {/* Symbol list */}
        {isLoading ? (
          <p className="text-xs text-muted-foreground py-4 text-center">Loading symbol data…</p>
        ) : symbols.length === 0 ? (
          <p className="text-xs text-muted-foreground py-4 text-center">
            No symbols recorded yet — they appear after AI dream analysis.
          </p>
        ) : (
          <div className="space-y-2">
            {symbols.map((ctx) => (
              <SymbolRow key={ctx.symbol} ctx={ctx} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
