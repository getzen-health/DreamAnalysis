/**
 * PredictiveAlertCard — Displays a predictive alert on the Today page.
 *
 * Warning alerts get an amber left border accent with AlertTriangle icon.
 * Positive alerts get an emerald left border accent with TrendingUp icon.
 * Returns nothing when alert is null (predictions are bonus content).
 */

import { AlertTriangle, TrendingUp, Eye } from "lucide-react";
import type { PredictiveAlert } from "@/lib/predictive-alerts";

export interface PredictiveAlertCardProps {
  alert: PredictiveAlert | null;
}

export function PredictiveAlertCard({ alert }: PredictiveAlertCardProps) {
  if (!alert) return null;

  const isWarning = alert.type === "warning";
  const borderColor = isWarning ? "border-l-amber-500" : "border-l-emerald-500";
  const Icon = isWarning ? AlertTriangle : TrendingUp;
  const iconColor = isWarning ? "text-amber-500" : "text-emerald-500";

  return (
    <div
      data-testid="predictive-alert-card"
      className={`rounded-[14px] bg-card border border-border border-l-4 ${borderColor} p-5`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <Eye className="h-4 w-4 text-muted-foreground" />
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Tomorrow's Forecast
        </span>
      </div>

      {/* Headline + Icon */}
      <div className="flex items-start gap-3 mb-2">
        <Icon className={`h-5 w-5 mt-0.5 shrink-0 ${iconColor}`} />
        <h3 data-testid="predictive-headline" className="text-sm font-semibold text-foreground leading-snug">
          {alert.headline}
        </h3>
      </div>

      {/* Body */}
      <p data-testid="predictive-body" className="text-xs text-muted-foreground leading-relaxed ml-8 mb-3">
        {alert.body}
      </p>

      {/* Action pill */}
      {alert.action && (
        <div className="ml-8 mb-3">
          <span
            data-testid="predictive-action"
            className={`inline-block text-xs font-medium px-3 py-1 rounded-full ${
              isWarning
                ? "bg-amber-500/10 text-amber-400"
                : "bg-emerald-500/10 text-emerald-400"
            }`}
          >
            {alert.action}
          </span>
        </div>
      )}

      {/* Factors tags */}
      {alert.factors.length > 0 && (
        <div data-testid="predictive-factors" className="ml-8 flex flex-wrap gap-1.5">
          {alert.factors.map((factor, i) => (
            <span
              key={i}
              className="text-[10px] text-muted-foreground bg-muted/50 px-2 py-0.5 rounded-full"
            >
              {factor}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
