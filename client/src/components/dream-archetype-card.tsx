import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Layers } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import {
  aggregateArchetypes,
  type DreamForArchetype,
  type AggregatedArchetype,
} from "@/lib/dream-archetype";

interface Props {
  userId: string;
}

function ArchetypeBar({ item, maxScore }: { item: AggregatedArchetype; maxScore: number }) {
  const pct = maxScore > 0 ? Math.round((item.score / maxScore) * 100) : 0;
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <span className="text-sm" aria-hidden>{item.icon}</span>
        <span className="text-xs font-medium flex-1">{item.label}</span>
        <span className="text-[10px] text-muted-foreground/60">
          {item.dreamCount} dream{item.dreamCount !== 1 ? "s" : ""}
        </span>
        <span className="text-[10px] text-muted-foreground/50 w-8 text-right">
          {Math.round(item.prevalence * 100)}%
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-white/8 overflow-hidden">
        <div
          className="h-full rounded-full bg-secondary/60 transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="text-[10px] text-muted-foreground/50 leading-snug">{item.description}</p>
    </div>
  );
}

export function DreamArchetypeCard({ userId }: Props) {
  const { data: rawData, isLoading } = useQuery<DreamForArchetype[]>({
    queryKey: ["dream-analysis", userId],
    queryFn: async () => {
      const res = await apiRequest("GET", `/api/dream-analysis/${userId}`); return res.json();
    },
    staleTime: 2 * 60 * 1000,
  });

  const raw = Array.isArray(rawData) ? rawData : [];
  const archetypes = aggregateArchetypes(raw, 5);
  const maxScore = archetypes[0]?.score ?? 1;

  return (
    <Card className="glass-card p-5 hover-glow border-secondary/20">
      <CardHeader className="p-0 pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Layers className="h-4 w-4 text-secondary" />
          Dream Archetypes
          {!isLoading && raw.length > 0 && (
            <span className="ml-auto text-[10px] text-muted-foreground">
              {raw.length} dream{raw.length !== 1 ? "s" : ""}
            </span>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent className="p-0 space-y-4">
        {isLoading ? (
          <p className="text-xs text-muted-foreground py-4 text-center">Analysing dream archetypes…</p>
        ) : archetypes.length === 0 || archetypes.every((a) => a.score === 0) ? (
          <p className="text-xs text-muted-foreground py-4 text-center">
            No archetype data yet — archetypes appear after dreams with symbols or themes are analysed.
          </p>
        ) : (
          <div className="space-y-3">
            {archetypes.map((item) => (
              <ArchetypeBar key={item.archetype} item={item} maxScore={maxScore} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
