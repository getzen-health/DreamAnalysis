/**
 * InterventionTriggerSettings — settings panel for EEG-triggered interventions.
 *
 * Provides a master toggle, individual toggles for breathing/music/break
 * suggestions, and a stress threshold slider. Persists to localStorage.
 *
 * @see Issue #504
 */

import { useState, useCallback } from "react";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import {
  Wind,
  Music,
  Coffee,
  Zap,
} from "lucide-react";
import {
  loadTriggerConfig,
  saveTriggerConfig,
  type TriggerConfig,
} from "@/lib/eeg-intervention-trigger";

// ── Component ────────────────────────────────────────────────────────────────

export function InterventionTriggerSettings() {
  const [config, setConfig] = useState<TriggerConfig>(() => loadTriggerConfig());

  const update = useCallback((patch: Partial<TriggerConfig>) => {
    setConfig((prev) => {
      const next = { ...prev, ...patch };
      saveTriggerConfig(next);
      return next;
    });
  }, []);

  return (
    <div className="space-y-5">
      {/* Master toggle */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <Zap className="h-4 w-4 text-amber-400" />
          <Label htmlFor="intervention-master" className="text-sm font-medium">
            Enable automatic interventions
          </Label>
        </div>
        <Switch
          id="intervention-master"
          checked={config.enabled}
          onCheckedChange={(checked) => update({ enabled: checked })}
        />
      </div>

      {config.enabled && (
        <div className="space-y-4 pl-1">
          {/* Stress threshold slider */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs text-muted-foreground">
                Stress threshold
              </Label>
              <span className="text-xs font-mono text-muted-foreground">
                {config.stressThreshold.toFixed(1)}
              </span>
            </div>
            <Slider
              min={0.5}
              max={0.9}
              step={0.05}
              value={[config.stressThreshold]}
              onValueChange={([v]) => update({ stressThreshold: v })}
              className="w-full"
            />
            <p className="text-[10px] text-muted-foreground leading-relaxed">
              Higher = less sensitive. Breathing exercise triggers when stress exceeds this for {config.stressDuration}s.
            </p>
          </div>

          {/* Individual toggles */}
          <div className="space-y-3">
            {/* Breathing */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <Wind className="h-3.5 w-3.5 text-cyan-400" />
                <div>
                  <Label htmlFor="auto-breathing" className="text-xs">
                    Breathing exercise
                  </Label>
                  <p className="text-[10px] text-muted-foreground">
                    Suggest breathing when sustained stress detected
                  </p>
                </div>
              </div>
              <Switch
                id="auto-breathing"
                checked={config.autoBreathing}
                onCheckedChange={(checked) => update({ autoBreathing: checked })}
              />
            </div>

            {/* Music change */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <Music className="h-3.5 w-3.5 text-indigo-400" />
                <div>
                  <Label htmlFor="auto-music" className="text-xs">
                    Music change
                  </Label>
                  <p className="text-[10px] text-muted-foreground">
                    Switch to calm music when alpha drops and beta stays high
                  </p>
                </div>
              </div>
              <Switch
                id="auto-music"
                checked={config.autoMusicChange}
                onCheckedChange={(checked) => update({ autoMusicChange: checked })}
              />
            </div>

            {/* Break reminders */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <Coffee className="h-3.5 w-3.5 text-amber-400" />
                <div>
                  <Label htmlFor="auto-break" className="text-xs">
                    Break reminders
                  </Label>
                  <p className="text-[10px] text-muted-foreground">
                    Suggest a break on eye fatigue or long sessions (25+ min)
                  </p>
                </div>
              </div>
              <Switch
                id="auto-break"
                checked={config.autoBreakReminder}
                onCheckedChange={(checked) => update({ autoBreakReminder: checked })}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
