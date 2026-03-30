/**
 * RealityTestSettings — configuration card for daytime reality testing reminders.
 *
 * Toggle on/off, adjust frequency (3–10/day), set active hours, preview today's schedule.
 * Persists config via loadRealityTestConfig / saveRealityTestConfig.
 */

import { useState, useEffect, useMemo } from "react";
import { Card } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Eye, Clock, Shuffle } from "lucide-react";
import {
  type RealityTestConfig,
  loadRealityTestConfig,
  saveRealityTestConfig,
  scheduleRealityTests,
} from "@/lib/reality-testing";

function pad2(n: number): string {
  return String(n).padStart(2, "0");
}

function formatTime(hour: number, minute: number): string {
  const ampm = hour >= 12 ? "PM" : "AM";
  const h = hour % 12 || 12;
  return `${h}:${pad2(minute)} ${ampm}`;
}

export function RealityTestSettings() {
  const [config, setConfig] = useState<RealityTestConfig>(loadRealityTestConfig);

  // Persist on every change
  useEffect(() => {
    saveRealityTestConfig(config);
  }, [config]);

  // Compute today's schedule preview
  const schedule = useMemo(
    () => (config.enabled ? scheduleRealityTests(config) : []),
    [config],
  );

  const updateConfig = (partial: Partial<RealityTestConfig>) => {
    setConfig((prev) => ({ ...prev, ...partial }));
  };

  return (
    <Card
      className="rounded-[14px] bg-card border border-border p-5 space-y-5"
      data-testid="reality-test-settings"
    >
      {/* Header + toggle */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Eye className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-medium">Reality Testing Reminders</h3>
        </div>
        <Switch
          checked={config.enabled}
          onCheckedChange={(checked) => updateConfig({ enabled: checked })}
          data-testid="reality-test-toggle"
        />
      </div>

      {config.enabled && (
        <>
          {/* Frequency slider */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs text-muted-foreground">
                Tests per day
              </Label>
              <span
                className="text-xs font-mono font-medium text-foreground"
                data-testid="frequency-value"
              >
                {config.frequency}
              </span>
            </div>
            <Slider
              min={3}
              max={10}
              step={1}
              value={[config.frequency]}
              onValueChange={([v]) => updateConfig({ frequency: v })}
              data-testid="frequency-slider"
            />
            <p className="text-[10px] text-muted-foreground">
              3 = gentle, 10 = intensive training
            </p>
          </div>

          {/* Active hours */}
          <div className="space-y-3">
            <div className="flex items-center gap-1.5">
              <Clock className="h-3.5 w-3.5 text-muted-foreground" />
              <Label className="text-xs text-muted-foreground">Active hours</Label>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <label className="text-[11px] text-muted-foreground">From</label>
                <select
                  value={config.startHour}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    updateConfig({ startHour: v });
                  }}
                  className="rounded-md border border-border bg-background px-2 py-1 text-xs font-mono text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                  data-testid="start-hour-select"
                >
                  {Array.from({ length: 24 }, (_, i) => (
                    <option key={i} value={i} disabled={i >= config.endHour}>
                      {pad2(i)}:00
                    </option>
                  ))}
                </select>
              </div>
              <span className="text-xs text-muted-foreground">to</span>
              <div className="flex items-center gap-2">
                <label className="text-[11px] text-muted-foreground">Until</label>
                <select
                  value={config.endHour}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    updateConfig({ endHour: v });
                  }}
                  className="rounded-md border border-border bg-background px-2 py-1 text-xs font-mono text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                  data-testid="end-hour-select"
                >
                  {Array.from({ length: 24 }, (_, i) => i + 1).map((h) => (
                    <option key={h} value={h} disabled={h <= config.startHour}>
                      {pad2(h)}:00
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Randomize toggle */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Shuffle className="h-3.5 w-3.5 text-muted-foreground" />
              <Label className="text-xs text-muted-foreground">
                Randomize timing
              </Label>
            </div>
            <Switch
              checked={config.randomize}
              onCheckedChange={(checked) => updateConfig({ randomize: checked })}
              data-testid="randomize-toggle"
            />
          </div>

          {/* Today's schedule preview */}
          {schedule.length > 0 && (
            <div className="space-y-2" data-testid="schedule-preview">
              <p className="text-[11px] text-muted-foreground font-medium uppercase tracking-wider">
                Today's schedule
              </p>
              <div className="grid gap-1.5">
                {schedule.map((slot, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-2 rounded-md bg-muted/30 px-3 py-1.5"
                    data-testid="schedule-slot"
                  >
                    <span className="text-xs font-mono text-muted-foreground w-16 shrink-0">
                      {formatTime(slot.hour, slot.minute)}
                    </span>
                    <span className="text-xs text-foreground truncate">
                      {slot.test.prompt}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </Card>
  );
}
