import { useState, useEffect } from "react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  getNotificationPreferences,
  saveNotificationPreferences,
  getPresetForFrequency,
  type NotificationFrequency,
  type NotificationPreferences,
} from "@/lib/notification-strategy";

const FREQUENCY_OPTIONS: {
  value: NotificationFrequency;
  label: string;
  description: string;
}[] = [
  {
    value: "minimal",
    label: "Minimal",
    description: "Morning brief only, 1x/day",
  },
  {
    value: "balanced",
    label: "Balanced",
    description: "Morning brief + session reminders + weekly insight",
  },
  {
    value: "engaged",
    label: "Engaged",
    description: "All notifications, up to 3x/day",
  },
];

const QUIET_HOURS = Array.from({ length: 24 }, (_, i) => {
  const label =
    i === 0 ? "12 AM" : i < 12 ? `${i} AM` : i === 12 ? "12 PM" : `${i - 12} PM`;
  return { value: i, label };
});

export function NotificationPrefsSheet({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [prefs, setPrefs] = useState<NotificationPreferences>(getNotificationPreferences);

  // Reload on open in case something changed externally
  useEffect(() => {
    if (open) {
      setPrefs(getNotificationPreferences());
    }
  }, [open]);

  function update(patch: Partial<NotificationPreferences>) {
    setPrefs((prev) => {
      const next = { ...prev, ...patch };
      saveNotificationPreferences(next);
      return next;
    });
  }

  function handleFrequencyChange(value: string) {
    const frequency = value as NotificationFrequency;
    const enabledTypes = getPresetForFrequency(frequency);
    const next = { ...prefs, frequency, enabledTypes };
    setPrefs(next);
    saveNotificationPreferences(next);
  }

  function toggleType(key: keyof NotificationPreferences["enabledTypes"]) {
    const enabledTypes = { ...prefs.enabledTypes, [key]: !prefs.enabledTypes[key] };
    update({ enabledTypes });
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="bottom"
        className="max-h-[85vh] overflow-y-auto rounded-t-2xl"
      >
        <SheetHeader className="text-left mb-4">
          <SheetTitle>Notification Preferences</SheetTitle>
          <SheetDescription>
            Choose how often you hear from us. We never guilt-trip.
          </SheetDescription>
        </SheetHeader>

        {/* Frequency picker */}
        <div className="mb-6">
          <p className="text-sm font-medium mb-3">Frequency</p>
          <RadioGroup
            value={prefs.frequency}
            onValueChange={handleFrequencyChange}
            className="space-y-3"
          >
            {FREQUENCY_OPTIONS.map((opt) => (
              <label
                key={opt.value}
                className="flex items-start gap-3 p-3 rounded-lg border border-border cursor-pointer hover:bg-muted/30 transition-colors"
              >
                <RadioGroupItem value={opt.value} className="mt-0.5" />
                <div>
                  <span className="text-sm font-medium">{opt.label}</span>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {opt.description}
                  </p>
                </div>
              </label>
            ))}
          </RadioGroup>
        </div>

        {/* Quiet hours */}
        <div className="mb-6">
          <p className="text-sm font-medium mb-3">Quiet Hours</p>
          <div className="flex items-center gap-3">
            <div className="flex-1">
              <Label className="text-xs text-muted-foreground">From</Label>
              <select
                value={prefs.quietHoursStart}
                onChange={(e) =>
                  update({ quietHoursStart: Number(e.target.value) })
                }
                className="w-full mt-1 rounded-md border border-border bg-background px-3 py-2 text-sm"
              >
                {QUIET_HOURS.map((h) => (
                  <option key={h.value} value={h.value}>
                    {h.label}
                  </option>
                ))}
              </select>
            </div>
            <span className="text-muted-foreground mt-5">to</span>
            <div className="flex-1">
              <Label className="text-xs text-muted-foreground">Until</Label>
              <select
                value={prefs.quietHoursEnd}
                onChange={(e) =>
                  update({ quietHoursEnd: Number(e.target.value) })
                }
                className="w-full mt-1 rounded-md border border-border bg-background px-3 py-2 text-sm"
              >
                {QUIET_HOURS.map((h) => (
                  <option key={h.value} value={h.value}>
                    {h.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Individual toggles */}
        <div className="mb-6">
          <p className="text-sm font-medium mb-3">Notification Types</p>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm">Morning Brief</p>
                <p className="text-xs text-muted-foreground">
                  Daily wellness summary
                </p>
              </div>
              <Switch
                checked={prefs.enabledTypes.morningBrief}
                onCheckedChange={() => toggleType("morningBrief")}
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm">Session Reminder</p>
                <p className="text-xs text-muted-foreground">
                  Neurofeedback/meditation reminder
                </p>
              </div>
              <Switch
                checked={prefs.enabledTypes.sessionReminder}
                onCheckedChange={() => toggleType("sessionReminder")}
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm">Weekly Insight</p>
                <p className="text-xs text-muted-foreground">
                  Weekly progress summary
                </p>
              </div>
              <Switch
                checked={prefs.enabledTypes.weeklyInsight}
                onCheckedChange={() => toggleType("weeklyInsight")}
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm">Streak Encouragement</p>
                <p className="text-xs text-muted-foreground">
                  Voice check-in streak updates
                </p>
              </div>
              <Switch
                checked={prefs.enabledTypes.streakEncouragement}
                onCheckedChange={() => toggleType("streakEncouragement")}
              />
            </div>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}
