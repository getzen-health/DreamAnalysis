/**
 * Sleep Insights — circadian rhythm chart, smart alarm suggestion, and sleep hygiene tips.
 * Uses sleep data from Health Connect via useHealthSync.
 */
import { useState, useEffect } from "react";
import { Clock, Lightbulb } from "lucide-react";

interface SleepDay {
  date: string;
  totalHours: number;
  deepHours: number;
  remHours: number;
  efficiency: number;
}

interface SleepInsightProps {
  sleepHours: number | null;
  deepHours: number | null;
  remHours: number | null;
  efficiency: number | null;
}

function getSleepHistory(): SleepDay[] {
  try {
    const raw = localStorage.getItem("ndw_sleep_history");
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

function saveSleepDay(day: SleepDay) {
  try {
    const history = getSleepHistory();
    const idx = history.findIndex(d => d.date === day.date);
    if (idx >= 0) history[idx] = day;
    else history.push(day);
    // Keep last 14 days
    const recent = history.slice(-14);
    localStorage.setItem("ndw_sleep_history", JSON.stringify(recent));
  } catch { /* ignore */ }
}

function getSmartAlarmSuggestion(avgSleep: number): string {
  // Average sleep cycle is ~90 minutes
  if (avgSleep < 6) {
    return "You're averaging under 6 hours. Try going to bed 30 minutes earlier tonight.";
  }
  if (avgSleep < 7) {
    const cycles = Math.round(avgSleep / 1.5);
    return `You're getting ~${cycles} sleep cycles. Adding one more (90 min earlier bedtime) could boost your focus.`;
  }
  if (avgSleep <= 9) {
    return "Your sleep duration is in the healthy range. Keep this consistent schedule.";
  }
  return "You're sleeping over 9 hours — could indicate sleep quality issues. Focus on deep sleep.";
}

function getSleepTips(efficiency: number | null, deepHours: number | null, remHours: number | null): string[] {
  const tips: string[] = [];
  if (efficiency !== null && efficiency < 80) {
    tips.push("Your sleep efficiency is below 80% — try avoiding screens 1 hour before bed");
  }
  if (deepHours !== null && deepHours < 1) {
    tips.push("Low deep sleep — regular exercise and cooler room temperature can help");
  }
  if (remHours !== null && remHours < 1) {
    tips.push("Low REM sleep — reduce alcohol and caffeine after 2 PM");
  }
  if (tips.length === 0) {
    tips.push("Your sleep looks healthy — maintain your current bedtime routine");
  }
  return tips;
}

export function SleepInsights({ sleepHours, deepHours, remHours, efficiency }: SleepInsightProps) {
  const [history, setHistory] = useState<SleepDay[]>([]);

  useEffect(() => {
    if (sleepHours !== null && sleepHours > 0) {
      const today = new Date().toISOString().slice(0, 10);
      saveSleepDay({
        date: today,
        totalHours: sleepHours,
        deepHours: deepHours || 0,
        remHours: remHours || 0,
        efficiency: efficiency || 0,
      });
    }
    setHistory(getSleepHistory());
  }, [sleepHours, deepHours, remHours, efficiency]);

  if (sleepHours === null || sleepHours === 0) return null;

  const avgSleep = history.length > 0
    ? history.reduce((a, b) => a + b.totalHours, 0) / history.length
    : sleepHours;
  const alarmTip = getSmartAlarmSuggestion(avgSleep);
  const tips = getSleepTips(efficiency, deepHours, remHours);

  return (
    <div style={{
      background: "var(--card)", border: "1px solid var(--border)",
      borderRadius: 14, padding: "12px 14px", marginBottom: 14,
    }}>
      <div style={{
        fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
        textTransform: "uppercase" as const, letterSpacing: "0.5px", marginBottom: 10,
      }}>
        Sleep Insights
      </div>

      {/* Circadian mini chart — 7-day bar chart */}
      {history.length >= 3 && (
        <div style={{ marginBottom: 10 }}>
          <div style={{ display: "flex", alignItems: "flex-end", gap: 3, height: 40, marginBottom: 4 }}>
            {history.slice(-7).map((day) => {
              const pct = Math.min(100, (day.totalHours / 10) * 100);
              const deepPct = day.deepHours / Math.max(day.totalHours, 0.1) * 100;
              return (
                <div key={day.date} style={{ flex: 1, display: "flex", flexDirection: "column" as const, alignItems: "center", height: "100%", justifyContent: "flex-end" }}>
                  <div style={{
                    width: "100%", borderRadius: 3, overflow: "hidden",
                    height: `${pct}%`, minHeight: 4, position: "relative" as const,
                    background: "hsl(270 40% 60% / 0.3)",
                  }}>
                    <div style={{
                      position: "absolute" as const, bottom: 0, width: "100%",
                      height: `${deepPct}%`, background: "hsl(270 40% 60%)",
                      borderRadius: "0 0 3px 3px",
                    }} />
                  </div>
                </div>
              );
            })}
          </div>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            {history.slice(-7).map((day) => (
              <div key={day.date} style={{ flex: 1, textAlign: "center" as const, fontSize: 8, color: "var(--muted-foreground)" }}>
                {new Date(day.date + "T12:00").toLocaleDateString("en", { weekday: "narrow" })}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Smart alarm suggestion */}
      <div style={{
        display: "flex", alignItems: "center", gap: 8,
        padding: "8px 0", borderTop: history.length >= 3 ? "1px solid var(--border)" : "none",
      }}>
        <Clock style={{ width: 14, height: 14, color: "hsl(200 70% 55%)", flexShrink: 0 }} />
        <div style={{ fontSize: 11, color: "var(--foreground)", lineHeight: 1.4 }}>{alarmTip}</div>
      </div>

      {/* Sleep hygiene tips */}
      {tips.map((tip, i) => (
        <div key={i} style={{
          display: "flex", alignItems: "center", gap: 8,
          padding: "6px 0", borderTop: "1px solid var(--border)",
        }}>
          <Lightbulb style={{ width: 14, height: 14, color: "hsl(38 85% 52%)", flexShrink: 0 }} />
          <div style={{ fontSize: 11, color: "var(--muted-foreground)", lineHeight: 1.4 }}>{tip}</div>
        </div>
      ))}
    </div>
  );
}
