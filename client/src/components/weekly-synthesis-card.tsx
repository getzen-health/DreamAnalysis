import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Loader2, BookOpen, RefreshCw } from "lucide-react";
import { resolveUrl } from "@/lib/queryClient";
import type { WeeklySynthesisResponse } from "@/lib/weekly-synthesis";

interface Props {
  userId: string;
}

export function WeeklySynthesisCard({ userId }: Props) {
  const [state, setState] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [data, setData] = useState<WeeklySynthesisResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState("");

  async function generate() {
    setState("loading");
    setErrorMsg("");
    try {
      const res = await fetch(resolveUrl(`/api/dream-weekly-synthesis/${userId}`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.message ?? `HTTP ${res.status}`);
      }
      const json: WeeklySynthesisResponse = await res.json();
      setData(json);
      setState("done");
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : "Generation failed");
      setState("error");
    }
  }

  const formattedDate = data?.generatedAt
    ? new Date(data.generatedAt).toLocaleString("en-US", {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      })
    : null;

  return (
    <Card className="glass-card p-5 hover-glow border-secondary/20">
      <CardHeader className="p-0 pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <BookOpen className="h-4 w-4 text-secondary" />
          Weekly Dream Synthesis
          {state === "done" && data && (
            <span className="ml-auto text-[10px] text-muted-foreground">
              {data.dreamCount} dream{data.dreamCount !== 1 ? "s" : ""} · {data.nightmareCount} nightmare{data.nightmareCount !== 1 ? "s" : ""}
            </span>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent className="p-0 space-y-3">
        {state === "idle" && (
          <>
            <p className="text-xs text-muted-foreground">
              Generate a personalised narrative that synthesises this week's dream themes, emotional arcs, and insights.
            </p>
            <Button
              size="sm"
              variant="outline"
              className="w-full border-secondary/30 text-secondary hover:bg-secondary/10"
              onClick={generate}
            >
              Generate weekly report
            </Button>
          </>
        )}

        {state === "loading" && (
          <div className="flex items-center gap-2 py-4 justify-center text-muted-foreground text-xs">
            <Loader2 className="w-4 h-4 animate-spin" />
            Synthesising this week's dreams…
          </div>
        )}

        {state === "error" && (
          <>
            <p className="text-xs text-destructive">{errorMsg}</p>
            <Button size="sm" variant="ghost" className="w-full text-xs" onClick={generate}>
              Try again
            </Button>
          </>
        )}

        {state === "done" && data && (
          <>
            {/* Top themes chips */}
            {data.topThemes.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {data.topThemes.map((t) => (
                  <span
                    key={t}
                    className="text-[10px] px-2 py-0.5 rounded-full bg-secondary/15 text-secondary capitalize"
                  >
                    {t}
                  </span>
                ))}
              </div>
            )}

            {/* Synthesis text */}
            <div className="text-xs text-muted-foreground leading-relaxed whitespace-pre-line border-l-2 border-secondary/30 pl-3">
              {data.synthesis}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between pt-1">
              {formattedDate && (
                <p className="text-[10px] text-muted-foreground/50">Generated {formattedDate}</p>
              )}
              <button
                onClick={generate}
                className="text-[10px] text-muted-foreground/50 hover:text-muted-foreground flex items-center gap-1 transition-colors"
              >
                <RefreshCw className="w-2.5 h-2.5" /> Regenerate
              </button>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
