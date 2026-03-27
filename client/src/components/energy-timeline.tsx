/**
 * EnergyTimeline — Bevel-style hourly energy forecast for the remaining day.
 * Shows color-coded bars for each hour with peak focus windows highlighted.
 */

import { motion } from "framer-motion";
import { Zap, Star } from "lucide-react";
import { predictEnergy, type HourlyEnergy } from "@/lib/energy-predictor";

interface Props {
  sleepHours?: number;
  recovery?: number;
  wakeHour?: number;
}

function formatHour(h: number): string {
  const hour = ((h % 24) + 24) % 24;
  if (hour === 0) return "12a";
  if (hour === 12) return "12p";
  if (hour < 12) return `${hour}a`;
  return `${hour - 12}p`;
}

function getBarColor(energy: number, isPeak: boolean): string {
  if (isPeak) return "bg-emerald-400";
  if (energy >= 0.7) return "bg-emerald-500/70";
  if (energy >= 0.5) return "bg-amber-500/70";
  if (energy >= 0.3) return "bg-amber-600/50";
  return "bg-gray-600/40";
}

export function EnergyTimeline({ sleepHours = 7, recovery = 50, wakeHour = 7 }: Props) {
  const forecast = predictEnergy(sleepHours, recovery, wakeHour);

  if (forecast.hours.length === 0) return null;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-amber-400" />
          <h3 className="text-sm font-medium">Your Energy Today</h3>
        </div>
        <div className="flex items-center gap-1 text-xs text-muted-foreground">
          <Star className="w-3 h-3 text-emerald-400" />
          <span>Peak: {forecast.peakWindow}</span>
        </div>
      </div>

      <div className="flex items-end gap-[3px] h-20 px-1">
        {forecast.hours.map((hour, i) => (
          <EnergyBar key={hour.hour} hour={hour} index={i} />
        ))}
      </div>

      {/* Hour labels — show every 2-3 hours */}
      <div className="flex gap-[3px] px-1">
        {forecast.hours.map((hour) => (
          <div
            key={hour.hour}
            className="flex-1 text-center"
          >
            <span className={`text-[9px] ${hour.isCurrent ? "text-primary font-bold" : "text-muted-foreground/50"}`}>
              {hour.hour % 2 === 0 ? formatHour(hour.hour) : ""}
            </span>
          </div>
        ))}
      </div>

      {/* Current energy label */}
      <div className="text-center">
        <span className="text-xs text-muted-foreground">
          Current: {Math.round(forecast.currentEnergy * 100)}% energy
        </span>
      </div>
    </div>
  );
}

function EnergyBar({ hour, index }: { hour: HourlyEnergy; index: number }) {
  const heightPct = Math.max(8, hour.energy * 100);
  const color = getBarColor(hour.energy, hour.isPeak);

  return (
    <motion.div
      className="flex-1 flex flex-col justify-end relative"
      initial={{ scaleY: 0 }}
      animate={{ scaleY: 1 }}
      transition={{ delay: index * 0.03, duration: 0.3 }}
      style={{ originY: 1, height: "100%" }}
    >
      <div
        className={`
          w-full rounded-t-sm transition-colors
          ${color}
          ${hour.isCurrent ? "ring-1 ring-primary ring-offset-1 ring-offset-background" : ""}
        `}
        style={{ height: `${heightPct}%` }}
        title={`${formatHour(hour.hour)}: ${Math.round(hour.energy * 100)}% — ${hour.label}`}
      />
      {hour.isPeak && (
        <div className="absolute -top-3 left-1/2 -translate-x-1/2">
          <Star className="w-2.5 h-2.5 text-emerald-400 fill-emerald-400" />
        </div>
      )}
    </motion.div>
  );
}
