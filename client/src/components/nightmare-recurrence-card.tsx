import { Link } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingDown, TrendingUp, Minus, HelpCircle, ShieldCheck } from "lucide-react";
import {
  type NightmareRecurrenceData,
  type NightmareTrend,
  trendLabel,
  irtEffectivenessLabel,
  shouldSuggestIrt,
  formatShortDate,
} from "@/lib/nightmare-recurrence";

interface Props {
  data: NightmareRecurrenceData;
}

const TREND_META: Record<
  NightmareTrend,
  { icon: React.FC<{ className?: string }>; color: string; bg: string; border: string }
> = {
  improving: {
    icon: TrendingDown,
    color: "text-emerald-400",
    bg: "bg-emerald-500/10",
    border: "border-emerald-500/30",
  },
  worsening: {
    icon: TrendingUp,
    color: "text-destructive",
    bg: "bg-destructive/10",
    border: "border-destructive/30",
  },
  stable: {
    icon: Minus,
    color: "text-amber-400",
    bg: "bg-amber-500/10",
    border: "border-amber-500/30",
  },
  unknown: {
    icon: HelpCircle,
    color: "text-muted-foreground",
    bg: "bg-muted/20",
    border: "border-muted/30",
  },
};

export function NightmareRecurrenceCard({ data }: Props) {
  if (data.recentNightmares === 0 && data.irtSessionCount === 0) return null;

  const meta = TREND_META[data.trend] ?? TREND_META["unknown"];
  const TrendIcon = meta.icon;
  const effectivenessMsg = irtEffectivenessLabel(data.postIrtNightmares, data.irtSessionCount);
  const showIrtCta = shouldSuggestIrt(data);

  return (
    <Card className={`glass-card p-5 hover-glow border ${meta.border}`}>
      <CardHeader className="p-0 pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <TrendIcon className={`h-4 w-4 ${meta.color}`} />
          Nightmare Recurrence
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0 space-y-3">
        {/* Trend badge */}
        <div className={`rounded-md px-3 py-2 ${meta.bg} flex items-center gap-2`}>
          <TrendIcon className={`h-3.5 w-3.5 ${meta.color} shrink-0`} />
          <p className={`text-xs font-medium ${meta.color}`}>{trendLabel(data.trend)}</p>
        </div>

        {/* 7-day window comparison */}
        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-lg bg-muted/30 py-2 text-center">
            <p className="text-base font-semibold text-foreground">{data.recentNightmares}</p>
            <p className="text-[9px] text-muted-foreground uppercase tracking-wide">Last 7 days</p>
          </div>
          <div className="rounded-lg bg-muted/30 py-2 text-center">
            <p className="text-base font-semibold text-foreground">{data.olderNightmares}</p>
            <p className="text-[9px] text-muted-foreground uppercase tracking-wide">Prior 7 days</p>
          </div>
        </div>

        {/* IRT effectiveness */}
        {effectivenessMsg && (
          <div
            className={`rounded-md px-3 py-2 flex items-start gap-2 ${
              data.postIrtNightmares === 0
                ? "bg-emerald-500/10 border border-emerald-500/20"
                : "bg-muted/20"
            }`}
          >
            <ShieldCheck
              className={`h-3.5 w-3.5 shrink-0 mt-0.5 ${
                data.postIrtNightmares === 0 ? "text-emerald-400" : "text-muted-foreground"
              }`}
            />
            <div>
              <p
                className={`text-xs ${
                  data.postIrtNightmares === 0 ? "text-emerald-400 font-medium" : "text-muted-foreground"
                }`}
              >
                {effectivenessMsg}
              </p>
              {data.lastIrtDate && (
                <p className="text-[10px] text-muted-foreground/60 mt-0.5">
                  Last IRT session: {formatShortDate(data.lastIrtDate)}
                </p>
              )}
            </div>
          </div>
        )}

        {/* Last nightmare date */}
        {data.lastNightmareDate && (
          <p className="text-[10px] text-muted-foreground/60">
            Most recent nightmare: {formatShortDate(data.lastNightmareDate)}
          </p>
        )}

        {/* IRT CTA */}
        {showIrtCta && (
          <Link href="/study/morning">
            <a className="block w-full text-center text-[11px] font-medium px-3 py-2 rounded-md bg-amber-500/15 border border-amber-500/30 text-amber-300 hover:bg-amber-500/25 transition-colors">
              Start IRT session in morning form →
            </a>
          </Link>
        )}
      </CardContent>
    </Card>
  );
}
