// client/src/components/morning-briefing-card.tsx
import { Sun, Loader2, Sparkles, TrendingUp } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import type { BriefingResponse } from "@/lib/insight-engine";

interface Props {
  loading: boolean;
  briefing: BriefingResponse | null;
  onGenerate: () => void;
}

export function MorningBriefingCard({ loading, briefing, onGenerate }: Props) {
  return (
    <Card className="glass-card p-5">
      <div className="flex items-center gap-2 mb-4">
        <Sun className="h-4 w-4 text-primary" />
        <h3 className="text-sm font-semibold">Morning Briefing</h3>
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground py-4">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span>Generating your briefing...</span>
        </div>
      )}

      {!loading && !briefing && (
        <div className="space-y-3">
          <p className="text-xs text-muted-foreground">
            Get a personalized morning synthesis of your sleep, HRV, and emotional patterns.
          </p>
          <Button onClick={onGenerate} className="w-full" size="sm">
            <Sparkles className="h-3.5 w-3.5 mr-2" />
            Good Morning — Generate Briefing
          </Button>
        </div>
      )}

      {!loading && briefing && (
        <div className="space-y-4">
          {/* State summary */}
          <p className="text-sm leading-relaxed text-foreground/90">{briefing.stateSummary}</p>

          {/* 3 actions */}
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground">Today's priorities</p>
            {briefing.actions.map((action, i) => (
              <div key={i} className="flex items-start gap-2.5">
                <div className="w-5 h-5 rounded-full bg-primary/10 text-primary text-xs font-bold flex items-center justify-center shrink-0 mt-0.5">
                  {i + 1}
                </div>
                <p className="text-sm">{action}</p>
              </div>
            ))}
          </div>

          {/* Forecast */}
          <div className="bg-muted/30 rounded-lg p-3 border border-border/20">
            <div className="flex items-center gap-2 mb-1">
              <TrendingUp className="h-3.5 w-3.5 text-primary" />
              <span className="text-xs font-medium">
                Forecast: {briefing.forecast.label} ({(briefing.forecast.probability * 100).toFixed(0)}%)
              </span>
            </div>
            <p className="text-xs text-muted-foreground">{briefing.forecast.reason}</p>
          </div>
        </div>
      )}
    </Card>
  );
}
