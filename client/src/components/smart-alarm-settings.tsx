/**
 * smart-alarm-settings.tsx — Settings card for the smart sleep alarm.
 *
 * Lets the user configure:
 *   - Target wake time (time picker)
 *   - Early-wake window (15 / 30 / 45 min)
 *   - Toggle alarm on/off
 */

import { Card } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { AlarmClock } from "lucide-react";

const WINDOW_OPTIONS = [15, 30, 45] as const;

export interface SmartAlarmSettingsProps {
  enabled: boolean;
  onEnabledChange: (enabled: boolean) => void;
  targetWakeTime: string;            // "HH:MM" format
  onTargetWakeTimeChange: (time: string) => void;
  windowMinutes: number;             // 15 | 30 | 45
  onWindowMinutesChange: (minutes: number) => void;
}

export default function SmartAlarmSettings({
  enabled,
  onEnabledChange,
  targetWakeTime,
  onTargetWakeTimeChange,
  windowMinutes,
  onWindowMinutesChange,
}: SmartAlarmSettingsProps) {
  return (
    <Card
      data-testid="smart-alarm-settings"
      className="rounded-[14px] bg-card border border-border p-5 space-y-4"
    >
      {/* Header row with toggle */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <AlarmClock className="h-4 w-4 text-muted-foreground" />
          <h3 className="text-sm font-medium">Smart Alarm</h3>
        </div>
        <div className="flex items-center gap-2">
          <Label htmlFor="smart-alarm-toggle" className="text-xs text-muted-foreground">
            {enabled ? "On" : "Off"}
          </Label>
          <Switch
            id="smart-alarm-toggle"
            data-testid="smart-alarm-toggle"
            checked={enabled}
            onCheckedChange={onEnabledChange}
          />
        </div>
      </div>

      {/* Description */}
      <p className="text-xs text-muted-foreground leading-relaxed">
        Wake during light sleep (N1/N2) or after a REM cycle within your chosen
        window for better alertness and no sleep inertia.
      </p>

      {/* Controls — only active when enabled */}
      <div className={enabled ? "space-y-4" : "space-y-4 opacity-50 pointer-events-none"}>
        {/* Target wake time */}
        <div className="flex items-center gap-3">
          <label
            htmlFor="smart-alarm-time"
            className="text-xs text-muted-foreground shrink-0"
          >
            Wake at
          </label>
          <input
            id="smart-alarm-time"
            data-testid="smart-alarm-time"
            type="time"
            value={targetWakeTime}
            onChange={(e) => onTargetWakeTimeChange(e.target.value)}
            className="rounded-md border border-border bg-background px-3 py-1.5 text-sm font-mono text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </div>

        {/* Window selector */}
        <div className="flex items-center gap-3">
          <span className="text-xs text-muted-foreground shrink-0">Window</span>
          <div className="flex gap-2" data-testid="smart-alarm-window-selector">
            {WINDOW_OPTIONS.map((opt) => (
              <button
                key={opt}
                data-testid={`smart-alarm-window-${opt}`}
                onClick={() => onWindowMinutesChange(opt)}
                className={`px-3 py-1 text-xs font-mono rounded-md border transition-colors ${
                  windowMinutes === opt
                    ? "border-primary bg-primary/10 text-primary font-medium"
                    : "border-border text-muted-foreground hover:border-primary/40"
                }`}
              >
                {opt}m
              </button>
            ))}
          </div>
        </div>
      </div>
    </Card>
  );
}
