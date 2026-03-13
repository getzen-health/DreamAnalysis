/**
 * NotificationSettings — user interface for managing smart daily notifications.
 *
 * Lets the user enable/disable morning brain reports and evening wind-down
 * prompts, configure send times, set quiet hours, and send test notifications.
 *
 * Data flow: fetches prefs from GET /notifications/preferences/{userId},
 * persists changes via PUT /notifications/preferences/{userId}.
 */

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Bell, BellOff, Moon, Sun, Clock, TestTube } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { useToast } from "@/hooks/use-toast";
import { getMLApiUrl } from "@/lib/ml-api";

// ── Types ──────────────────────────────────────────────────────────────────────

interface NotificationPrefs {
  user_id: string;
  enabled: boolean;
  morning_report_enabled: boolean;
  evening_winddown_enabled: boolean;
  post_session_enabled: boolean;
  weekly_summary_enabled: boolean;
  quiet_hours_start: number;
  quiet_hours_end: number;
  morning_hour: number;
  evening_hour: number;
  timezone_offset_hours: number;
  skip_weekends_morning: boolean;
  min_stress_for_evening: number;
  updated_at: number;
}

interface NotificationPrefsUpdate {
  enabled?: boolean;
  morning_report_enabled?: boolean;
  evening_winddown_enabled?: boolean;
  post_session_enabled?: boolean;
  weekly_summary_enabled?: boolean;
  quiet_hours_start?: number;
  quiet_hours_end?: number;
  morning_hour?: number;
  evening_hour?: number;
  skip_weekends_morning?: boolean;
  min_stress_for_evening?: number;
}

interface TestNotificationResult {
  test: boolean;
  type: string;
  title: string;
  body: string;
  route: string;
  skip?: boolean;
  skip_reason?: string | null;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function formatHour(hour: number): string {
  if (hour === 0) return "12:00 AM";
  if (hour < 12) return `${hour}:00 AM`;
  if (hour === 12) return "12:00 PM";
  return `${hour - 12}:00 PM`;
}

// ── API calls ──────────────────────────────────────────────────────────────────

async function fetchPrefs(userId: string): Promise<NotificationPrefs> {
  const res = await fetch(
    `${getMLApiUrl()}/api/notifications/preferences/${encodeURIComponent(userId)}`
  );
  if (!res.ok) throw new Error(`Failed to load preferences: ${res.status}`);
  return res.json() as Promise<NotificationPrefs>;
}

async function putPrefs(
  userId: string,
  updates: NotificationPrefsUpdate
): Promise<NotificationPrefs> {
  const res = await fetch(
    `${getMLApiUrl()}/api/notifications/preferences/${encodeURIComponent(userId)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updates),
    }
  );
  if (!res.ok) throw new Error(`Failed to update preferences: ${res.status}`);
  const body = await res.json() as { preferences: NotificationPrefs };
  return body.preferences;
}

async function sendTestNotification(
  userId: string,
  type: "morning_report" | "evening_winddown"
): Promise<TestNotificationResult> {
  const res = await fetch(
    `${getMLApiUrl()}/api/notifications/test/${encodeURIComponent(userId)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notification_type: type }),
    }
  );
  if (!res.ok) throw new Error(`Test notification failed: ${res.status}`);
  return res.json() as Promise<TestNotificationResult>;
}

// ── Section component ─────────────────────────────────────────────────────────

function SettingRow({
  label,
  description,
  children,
}: {
  label: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4 py-3 border-b border-border/40 last:border-0">
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium leading-tight">{label}</p>
        {description && (
          <p className="text-xs text-muted-foreground mt-0.5">{description}</p>
        )}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

interface NotificationSettingsProps {
  userId: string;
}

export function NotificationSettings({ userId }: NotificationSettingsProps) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [testResult, setTestResult] = useState<TestNotificationResult | null>(null);

  const { data: prefs, isLoading, isError } = useQuery<NotificationPrefs>({
    queryKey: ["notification-prefs", userId],
    queryFn: () => fetchPrefs(userId),
    staleTime: 60_000,
    retry: 1,
  });

  const mutation = useMutation<NotificationPrefs, Error, NotificationPrefsUpdate>({
    mutationFn: (updates) => putPrefs(userId, updates),
    onSuccess: (updated) => {
      queryClient.setQueryData(["notification-prefs", userId], updated);
    },
    onError: (err) => {
      toast({
        title: "Failed to save",
        description: err.message,
        variant: "destructive",
      });
    },
  });

  const testMutation = useMutation<
    TestNotificationResult,
    Error,
    "morning_report" | "evening_winddown"
  >({
    mutationFn: (type) => sendTestNotification(userId, type),
    onSuccess: (result) => {
      setTestResult(result);
      toast({
        title: result.skip ? "Notification would be skipped" : "Test notification generated",
        description: result.skip
          ? result.skip_reason ?? "Stress below threshold"
          : result.title,
      });
    },
    onError: (err) => {
      toast({
        title: "Test failed",
        description: err.message,
        variant: "destructive",
      });
    },
  });

  const update = (updates: NotificationPrefsUpdate) => mutation.mutate(updates);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="p-6 flex items-center justify-center">
          <div className="h-5 w-5 rounded-full border-2 border-primary border-t-transparent animate-spin" />
        </CardContent>
      </Card>
    );
  }

  if (isError || !prefs) {
    return (
      <Card>
        <CardContent className="p-6">
          <p className="text-sm text-muted-foreground">
            Could not load notification settings.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Master toggle */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            {prefs.enabled ? (
              <Bell className="h-4 w-4 text-primary" />
            ) : (
              <BellOff className="h-4 w-4 text-muted-foreground" />
            )}
            Smart Notifications
            {prefs.enabled && (
              <Badge className="ml-auto text-xs bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                Active
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <SettingRow
            label="Enable notifications"
            description="Master switch — turns off all notifications when disabled"
          >
            <Switch
              checked={prefs.enabled}
              onCheckedChange={(v) => update({ enabled: v })}
            />
          </SettingRow>
        </CardContent>
      </Card>

      {/* Morning report */}
      <Card className={prefs.enabled ? "" : "opacity-60 pointer-events-none"}>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Sun className="h-4 w-4 text-amber-400" />
            Morning Brain Report
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <SettingRow
            label="Morning report"
            description={`Sent at ${formatHour(prefs.morning_hour)} with your brain scores`}
          >
            <Switch
              checked={prefs.morning_report_enabled}
              onCheckedChange={(v) => update({ morning_report_enabled: v })}
            />
          </SettingRow>

          <SettingRow
            label={`Send time: ${formatHour(prefs.morning_hour)}`}
            description="Adjust when you want the morning report"
          >
            <div className="w-40">
              <Slider
                min={5}
                max={11}
                step={1}
                value={[prefs.morning_hour]}
                onValueChange={([v]) => update({ morning_hour: v })}
                className="w-full"
              />
            </div>
          </SettingRow>

          <SettingRow
            label="Skip weekends"
            description="Skip morning reports on Saturday and Sunday"
          >
            <Switch
              checked={prefs.skip_weekends_morning}
              onCheckedChange={(v) => update({ skip_weekends_morning: v })}
            />
          </SettingRow>

          <div className="pt-2">
            <Button
              variant="outline"
              size="sm"
              className="gap-2"
              onClick={() => testMutation.mutate("morning_report")}
              disabled={testMutation.isPending}
            >
              <TestTube className="h-3.5 w-3.5" />
              {testMutation.isPending ? "Generating…" : "Preview morning report"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Evening wind-down */}
      <Card className={prefs.enabled ? "" : "opacity-60 pointer-events-none"}>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Moon className="h-4 w-4 text-indigo-400" />
            Evening Wind-Down
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <SettingRow
            label="Evening wind-down"
            description={`Sent at ${formatHour(prefs.evening_hour)} if stress was elevated`}
          >
            <Switch
              checked={prefs.evening_winddown_enabled}
              onCheckedChange={(v) => update({ evening_winddown_enabled: v })}
            />
          </SettingRow>

          <SettingRow
            label={`Send time: ${formatHour(prefs.evening_hour)}`}
            description="Adjust when you want the evening prompt"
          >
            <div className="w-40">
              <Slider
                min={18}
                max={23}
                step={1}
                value={[prefs.evening_hour]}
                onValueChange={([v]) => update({ evening_hour: v })}
                className="w-full"
              />
            </div>
          </SettingRow>

          <SettingRow
            label={`Min stress to trigger: ${(prefs.min_stress_for_evening * 100).toFixed(0)}%`}
            description="Evening prompt only sends when today's stress exceeds this level"
          >
            <div className="w-40">
              <Slider
                min={0}
                max={80}
                step={5}
                value={[Math.round(prefs.min_stress_for_evening * 100)]}
                onValueChange={([v]) =>
                  update({ min_stress_for_evening: v / 100 })
                }
                className="w-full"
              />
            </div>
          </SettingRow>

          <div className="pt-2">
            <Button
              variant="outline"
              size="sm"
              className="gap-2"
              onClick={() => testMutation.mutate("evening_winddown")}
              disabled={testMutation.isPending}
            >
              <TestTube className="h-3.5 w-3.5" />
              {testMutation.isPending ? "Generating…" : "Preview evening prompt"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Quiet hours */}
      <Card className={prefs.enabled ? "" : "opacity-60 pointer-events-none"}>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Clock className="h-4 w-4 text-muted-foreground" />
            Quiet Hours
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <SettingRow
            label={`No notifications ${formatHour(prefs.quiet_hours_start)} – ${formatHour(prefs.quiet_hours_end)}`}
            description="Notifications are suppressed during this window"
          >
            <span className="text-xs text-muted-foreground">
              {prefs.quiet_hours_start}:00 – {prefs.quiet_hours_end}:00
            </span>
          </SettingRow>

          <SettingRow
            label={`Start: ${formatHour(prefs.quiet_hours_start)}`}
          >
            <div className="w-40">
              <Slider
                min={20}
                max={23}
                step={1}
                value={[prefs.quiet_hours_start]}
                onValueChange={([v]) => update({ quiet_hours_start: v })}
                className="w-full"
              />
            </div>
          </SettingRow>

          <SettingRow
            label={`End: ${formatHour(prefs.quiet_hours_end)}`}
          >
            <div className="w-40">
              <Slider
                min={5}
                max={10}
                step={1}
                value={[prefs.quiet_hours_end]}
                onValueChange={([v]) => update({ quiet_hours_end: v })}
                className="w-full"
              />
            </div>
          </SettingRow>
        </CardContent>
      </Card>

      {/* Other notification types */}
      <Card className={prefs.enabled ? "" : "opacity-60 pointer-events-none"}>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Other Notifications</CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <SettingRow
            label="Post-session insights"
            description="Immediate insight after a voice or EEG session completes"
          >
            <Switch
              checked={prefs.post_session_enabled}
              onCheckedChange={(v) => update({ post_session_enabled: v })}
            />
          </SettingRow>

          <SettingRow
            label="Weekly summary"
            description="Sunday morning summary of your week's brain patterns"
          >
            <Switch
              checked={prefs.weekly_summary_enabled}
              onCheckedChange={(v) => update({ weekly_summary_enabled: v })}
            />
          </SettingRow>
        </CardContent>
      </Card>

      {/* Test result preview */}
      {testResult && !testResult.skip && (
        <Card className="border-primary/30 bg-primary/5">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-primary">Preview</CardTitle>
          </CardHeader>
          <CardContent className="pt-0 space-y-1">
            <p className="text-sm font-semibold">{testResult.title}</p>
            <p className="text-xs text-muted-foreground">{testResult.body}</p>
            <p className="text-xs text-muted-foreground">
              Route: <code className="text-primary">{testResult.route}</code>
            </p>
          </CardContent>
        </Card>
      )}

      {testResult?.skip && (
        <Card className="border-amber-500/30 bg-amber-500/5">
          <CardContent className="p-4">
            <p className="text-sm text-amber-400">
              Notification would be skipped — {testResult.skip_reason}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
