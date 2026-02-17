import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Moon, TrendingUp, Brain, Activity, Sparkles } from "lucide-react";

/* ---------- simulated multi-night data ---------- */
function generateNightlyData() {
  const nights = [];
  for (let i = 6; i >= 0; i--) {
    const date = new Date();
    date.setDate(date.getDate() - i);
    nights.push({
      day: date.toLocaleDateString("en-US", { weekday: "short" }),
      remMinutes: Math.round(60 + Math.random() * 60),
      deepMinutes: Math.round(40 + Math.random() * 50),
      lightMinutes: Math.round(120 + Math.random() * 80),
      dreamEpisodes: Math.round(1 + Math.random() * 4),
      avgIntensity: Math.round(30 + Math.random() * 50),
      avgLucidity: Math.round(5 + Math.random() * 30),
      sleepScore: Math.round(55 + Math.random() * 35),
    });
  }
  return nights;
}

function generateHypnogram() {
  // Simulated sleep architecture for last night (30-min intervals, 8 hours)
  const stages = ["Wake", "N1", "N2", "N3", "REM"];
  const stageValues: Record<string, number> = { Wake: 4, REM: 3, N1: 2, N2: 1, N3: 0 };
  const typical = [
    "Wake", "N1", "N2", "N3", "N3", "N2", "REM",
    "N2", "N3", "N3", "N2", "REM", "REM",
    "N2", "N2", "REM",
  ];
  const now = new Date();
  now.setHours(23, 0, 0, 0);
  return typical.map((stage, i) => {
    const t = new Date(now.getTime() + i * 30 * 60000);
    return {
      time: t.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" }),
      stage: stageValues[stage],
      label: stage,
    };
  });
}

function generateRemCycles() {
  return Array.from({ length: 5 }, (_, i) => ({
    cycle: `Cycle ${i + 1}`,
    duration: Math.round(8 + Math.random() * 20),
    intensity: Math.round(30 + i * 8 + Math.random() * 20),
    lucidity: Math.round(5 + i * 5 + Math.random() * 15),
  }));
}

/* ========== Component ========== */
export default function DreamPatterns() {
  const [nightlyData] = useState(generateNightlyData);
  const [hypnogram] = useState(generateHypnogram);
  const [remCycles] = useState(generateRemCycles);

  const totalDreams = nightlyData.reduce((s, n) => s + n.dreamEpisodes, 0);
  const avgRem = Math.round(nightlyData.reduce((s, n) => s + n.remMinutes, 0) / nightlyData.length);
  const avgScore = Math.round(nightlyData.reduce((s, n) => s + n.sleepScore, 0) / nightlyData.length);

  const STAGE_NAMES = ["N3 (Deep)", "N2 (Light)", "N1", "REM", "Wake"];

  return (
    <main className="p-6 space-y-6 max-w-5xl">
      {/* Summary Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: "Dreams Detected", value: totalDreams, sub: "last 7 nights", color: "text-secondary" },
          { label: "Avg REM", value: `${avgRem}min`, sub: "per night", color: "text-primary" },
          { label: "Sleep Score", value: avgScore, sub: "7-day avg", color: "text-accent" },
          { label: "REM Cycles", value: remCycles.length, sub: "last night", color: "text-foreground" },
        ].map((stat) => (
          <Card key={stat.label} className="glass-card p-4 hover-glow text-center">
            <p className={`text-2xl font-semibold ${stat.color}`}>{stat.value}</p>
            <p className="text-xs text-muted-foreground mt-1">{stat.label}</p>
            <p className="text-[10px] text-muted-foreground/60">{stat.sub}</p>
          </Card>
        ))}
      </div>

      {/* Sleep Architecture (Hypnogram) */}
      <Card className="glass-card p-5 hover-glow">
        <div className="flex items-center gap-2 mb-4">
          <Brain className="h-4 w-4 text-secondary" />
          <h3 className="text-sm font-medium">Sleep Architecture — Last Night</h3>
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={hypnogram}>
            <defs>
              <linearGradient id="hypnoGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="hsl(262, 45%, 65%)" stopOpacity={0.3} />
                <stop offset="100%" stopColor="hsl(262, 45%, 65%)" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="time"
              tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={[0, 4]}
              ticks={[0, 1, 2, 3, 4]}
              tickFormatter={(v: number) => STAGE_NAMES[v] || ""}
              tick={{ fontSize: 9, fill: "hsl(220, 12%, 42%)" }}
              width={70}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                background: "hsl(220, 22%, 9%)",
                border: "1px solid hsl(220, 18%, 20%)",
                borderRadius: 8,
                fontSize: 12,
              }}
              labelStyle={{ color: "hsl(38, 20%, 92%)" }}
              formatter={(value: number) => [STAGE_NAMES[value] || "Unknown", "Stage"]}
            />
            <Area
              type="stepAfter"
              dataKey="stage"
              stroke="hsl(262, 45%, 65%)"
              fill="url(#hypnoGrad)"
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </Card>

      {/* Weekly Trends */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Dream Episodes & Intensity */}
        <Card className="glass-card p-5 hover-glow">
          <div className="flex items-center gap-2 mb-4">
            <Moon className="h-4 w-4 text-secondary" />
            <h3 className="text-sm font-medium">Dream Detection — 7 Days</h3>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={nightlyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 15%)" opacity={0.4} />
              <XAxis
                dataKey="day"
                tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis hide />
              <Tooltip
                contentStyle={{
                  background: "hsl(220, 22%, 9%)",
                  border: "1px solid hsl(220, 18%, 20%)",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                labelStyle={{ color: "hsl(38, 20%, 92%)" }}
              />
              <Bar
                dataKey="dreamEpisodes"
                fill="hsl(262, 45%, 65%)"
                radius={[4, 4, 0, 0]}
                name="Dreams Detected"
              />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* REM vs Deep vs Light */}
        <Card className="glass-card p-5 hover-glow">
          <div className="flex items-center gap-2 mb-4">
            <Activity className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-medium">Sleep Stages — 7 Days</h3>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={nightlyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 15%)" opacity={0.4} />
              <XAxis
                dataKey="day"
                tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis hide />
              <Tooltip
                contentStyle={{
                  background: "hsl(220, 22%, 9%)",
                  border: "1px solid hsl(220, 18%, 20%)",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                labelStyle={{ color: "hsl(38, 20%, 92%)" }}
              />
              <Bar dataKey="deepMinutes" stackId="a" fill="hsl(262, 45%, 55%)" name="Deep" radius={[0, 0, 0, 0]} />
              <Bar dataKey="lightMinutes" stackId="a" fill="hsl(220, 50%, 50%)" name="Light" radius={[0, 0, 0, 0]} />
              <Bar dataKey="remMinutes" stackId="a" fill="hsl(152, 60%, 48%)" name="REM" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <div className="flex justify-center gap-4 mt-3">
            {[
              { label: "Deep", color: "hsl(262, 45%, 55%)" },
              { label: "Light", color: "hsl(220, 50%, 50%)" },
              { label: "REM", color: "hsl(152, 60%, 48%)" },
            ].map((item) => (
              <div key={item.label} className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-sm" style={{ background: item.color }} />
                <span className="text-[10px] text-muted-foreground">{item.label}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* REM Cycle Progression */}
      <Card className="glass-card p-5 hover-glow">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-medium">REM Cycle Progression — Last Night</h3>
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={remCycles}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 15%)" opacity={0.4} />
            <XAxis
              dataKey="cycle"
              tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                background: "hsl(220, 22%, 9%)",
                border: "1px solid hsl(220, 18%, 20%)",
                borderRadius: 8,
                fontSize: 12,
              }}
              labelStyle={{ color: "hsl(38, 20%, 92%)" }}
            />
            <Line
              type="monotone"
              dataKey="intensity"
              stroke="hsl(38, 85%, 58%)"
              strokeWidth={2}
              dot={{ r: 4, fill: "hsl(38, 85%, 58%)" }}
              name="Intensity %"
            />
            <Line
              type="monotone"
              dataKey="lucidity"
              stroke="hsl(200, 70%, 55%)"
              strokeWidth={2}
              dot={{ r: 4, fill: "hsl(200, 70%, 55%)" }}
              name="Lucidity %"
            />
            <Line
              type="monotone"
              dataKey="duration"
              stroke="hsl(152, 60%, 48%)"
              strokeWidth={2}
              dot={{ r: 4, fill: "hsl(152, 60%, 48%)" }}
              name="Duration (min)"
            />
          </LineChart>
        </ResponsiveContainer>
        <div className="flex justify-center gap-4 mt-3">
          {[
            { label: "Intensity", color: "hsl(38, 85%, 58%)" },
            { label: "Lucidity", color: "hsl(200, 70%, 55%)" },
            { label: "Duration", color: "hsl(152, 60%, 48%)" },
          ].map((item) => (
            <div key={item.label} className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full" style={{ background: item.color }} />
              <span className="text-[10px] text-muted-foreground">{item.label}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* AI Analysis */}
      <div className="ai-insight-card">
        <div className="flex items-start gap-3">
          <Sparkles className="h-5 w-5 text-primary mt-0.5 shrink-0" />
          <div>
            <p className="text-sm font-medium text-foreground mb-1">Pattern Analysis</p>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Over 7 nights, your BCI detected {totalDreams} dream episodes with an average of {avgRem} minutes
              of REM sleep per night. REM cycle duration naturally increases through the night — your
              later cycles show {remCycles.length > 2 ? `${remCycles[remCycles.length - 1].duration}min vs ${remCycles[0].duration}min` : "healthy progression"},
              which is consistent with normal sleep architecture. Lucidity estimates trend upward in later
              cycles, suggesting enhanced self-awareness during extended REM periods.
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
