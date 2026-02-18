import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScoreCircle } from "@/components/score-circle";
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
import {
  Heart,
  Activity,
  Moon,
  Sparkles,
  Footprints,
  Flame,
  Wind,
  TrendingUp,
} from "lucide-react";
import {
  getHealthDailySummary,
  getHealthInsights,
  getHealthTrends,
  type HealthInsight,
  type HealthTrend,
} from "@/lib/ml-api";

/* ---------- simulated fallback data ---------- */
function generateFallbackMetrics() {
  return {
    heartRate: 68 + Math.round(Math.random() * 12),
    hrv: 35 + Math.round(Math.random() * 25),
    steps: 5000 + Math.round(Math.random() * 7000),
    sleepHours: +(6 + Math.random() * 2.5).toFixed(1),
    calories: 1800 + Math.round(Math.random() * 600),
    spO2: 95 + Math.round(Math.random() * 4),
    respiratoryRate: 12 + Math.round(Math.random() * 6),
    stressLevel: 20 + Math.round(Math.random() * 40),
  };
}

function generateFallbackWeekly() {
  return Array.from({ length: 7 }, (_, i) => {
    const d = new Date();
    d.setDate(d.getDate() - (6 - i));
    return {
      day: d.toLocaleDateString("en-US", { weekday: "short" }),
      heartRate: 62 + Math.round(Math.random() * 15),
      hrv: 30 + Math.round(Math.random() * 30),
      steps: 4000 + Math.round(Math.random() * 8000),
      sleep: +(5.5 + Math.random() * 3).toFixed(1),
      stress: 15 + Math.round(Math.random() * 45),
    };
  });
}

function generateFallbackInsights(): HealthInsight[] {
  return [
    {
      insight_type: "exercise_flow",
      title: "Exercise boosts your flow state",
      description: "On days with 8,000+ steps, your flow scores are 23% higher. Physical activity primes the brain for focused work.",
      correlation_strength: 0.72,
      evidence_count: 14,
      brain_metric: "flow_score",
      health_metric: "steps",
    },
    {
      insight_type: "hrv_creativity",
      title: "High HRV correlates with creativity",
      description: "When your HRV is above 45ms, creativity scores increase significantly. Try breathing exercises to boost HRV.",
      correlation_strength: 0.65,
      evidence_count: 11,
      brain_metric: "creativity_score",
      health_metric: "hrv_sdnn",
    },
    {
      insight_type: "sleep_memory",
      title: "Deep sleep strengthens memory encoding",
      description: "Nights with 7+ hours of sleep show 35% better memory encoding the next day. Prioritize sleep consistency.",
      correlation_strength: 0.81,
      evidence_count: 18,
      brain_metric: "encoding_score",
      health_metric: "sleep_analysis",
    },
    {
      insight_type: "heart_rate_arousal",
      title: "Elevated heart rate tracks emotional arousal",
      description: "Your resting heart rate closely tracks emotional arousal patterns. Calming techniques lower both simultaneously.",
      correlation_strength: 0.58,
      evidence_count: 22,
      brain_metric: "arousal",
      health_metric: "heart_rate",
    },
  ];
}

/* ========== Component ========== */
export default function HealthAnalytics() {
  const [metrics] = useState(generateFallbackMetrics);
  const [weekly] = useState(generateFallbackWeekly);
  const [insights, setInsights] = useState<HealthInsight[]>(generateFallbackInsights);
  const [apiConnected, setApiConnected] = useState(false);

  // Try to fetch real data from backend
  useEffect(() => {
    async function fetchReal() {
      try {
        const realInsights = await getHealthInsights("demo-user");
        if (realInsights && realInsights.length > 0) {
          setInsights(realInsights);
          setApiConnected(true);
        }
      } catch {
        // Backend not available, use fallback data
      }
    }
    fetchReal();
  }, []);

  const sleepScore = Math.round(Math.min(100, (metrics.sleepHours / 8) * 100));
  const activityScore = Math.round(Math.min(100, (metrics.steps / 10000) * 100));
  const heartScore = Math.round(Math.max(0, 100 - Math.abs(metrics.heartRate - 70) * 2));

  return (
    <main className="p-6 space-y-6 max-w-5xl">
      {/* Score Gauges */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="score-card p-4 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={heartScore}
            label="Heart"
            gradientId="grad-heart"
            colorFrom="hsl(4, 72%, 55%)"
            colorTo="hsl(25, 85%, 55%)"
            size="sm"
          />
          <p className="text-xs text-muted-foreground mt-1">{metrics.heartRate} BPM</p>
        </div>
        <div className="score-card p-4 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={sleepScore}
            label="Sleep"
            gradientId="grad-health-sleep"
            colorFrom="hsl(262, 45%, 65%)"
            colorTo="hsl(220, 50%, 50%)"
            size="sm"
          />
          <p className="text-xs text-muted-foreground mt-1">{metrics.sleepHours}h</p>
        </div>
        <div className="score-card p-4 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={activityScore}
            label="Activity"
            gradientId="grad-activity"
            colorFrom="hsl(152, 60%, 48%)"
            colorTo="hsl(38, 85%, 58%)"
            size="sm"
          />
          <p className="text-xs text-muted-foreground mt-1">{metrics.steps.toLocaleString()} steps</p>
        </div>
        <div className="score-card p-4 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={metrics.spO2}
            label="SpO2"
            gradientId="grad-spo2"
            colorFrom="hsl(200, 70%, 55%)"
            colorTo="hsl(152, 60%, 48%)"
            size="sm"
          />
          <p className="text-xs text-muted-foreground mt-1">{metrics.spO2}%</p>
        </div>
      </div>

      {/* Vital Stats Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { icon: Heart, label: "HRV", value: `${metrics.hrv}ms`, color: "text-destructive" },
          { icon: Wind, label: "Resp Rate", value: `${metrics.respiratoryRate}/min`, color: "text-primary" },
          { icon: Flame, label: "Calories", value: metrics.calories.toLocaleString(), color: "text-accent" },
          { icon: Activity, label: "Stress", value: `${metrics.stressLevel}%`, color: metrics.stressLevel > 50 ? "text-warning" : "text-success" },
        ].map((stat) => {
          const Icon = stat.icon;
          return (
            <Card key={stat.label} className="glass-card p-4 hover-glow">
              <div className="flex items-center gap-2 mb-2">
                <Icon className={`h-4 w-4 ${stat.color}`} />
                <span className="text-xs text-muted-foreground">{stat.label}</span>
              </div>
              <p className={`text-lg font-semibold font-mono ${stat.color}`}>{stat.value}</p>
            </Card>
          );
        })}
      </div>

      {/* Weekly Trends */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Heart Rate & HRV */}
        <Card className="glass-card p-5 hover-glow">
          <div className="flex items-center gap-2 mb-4">
            <Heart className="h-4 w-4 text-destructive" />
            <h3 className="text-sm font-medium">Heart Rate & HRV — 7 Days</h3>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={weekly}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 15%)" opacity={0.4} />
              <XAxis dataKey="day" tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: "hsl(220, 22%, 9%)", border: "1px solid hsl(220, 18%, 20%)", borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: "hsl(38, 20%, 92%)" }}
              />
              <Line type="monotone" dataKey="heartRate" stroke="hsl(4, 72%, 55%)" strokeWidth={2} dot={{ r: 3 }} name="HR (BPM)" />
              <Line type="monotone" dataKey="hrv" stroke="hsl(200, 70%, 55%)" strokeWidth={2} dot={{ r: 3 }} name="HRV (ms)" />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        {/* Steps */}
        <Card className="glass-card p-5 hover-glow">
          <div className="flex items-center gap-2 mb-4">
            <Footprints className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-medium">Daily Steps — 7 Days</h3>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={weekly}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 15%)" opacity={0.4} />
              <XAxis dataKey="day" tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
              <YAxis hide />
              <Tooltip
                contentStyle={{ background: "hsl(220, 22%, 9%)", border: "1px solid hsl(220, 18%, 20%)", borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: "hsl(38, 20%, 92%)" }}
              />
              <Bar dataKey="steps" fill="hsl(152, 60%, 48%)" radius={[4, 4, 0, 0]} name="Steps" />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Sleep Duration */}
        <Card className="glass-card p-5 hover-glow">
          <div className="flex items-center gap-2 mb-4">
            <Moon className="h-4 w-4 text-secondary" />
            <h3 className="text-sm font-medium">Sleep Duration — 7 Days</h3>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={weekly}>
              <defs>
                <linearGradient id="sleepHealthGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="hsl(262, 45%, 65%)" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="hsl(262, 45%, 65%)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 15%)" opacity={0.4} />
              <XAxis dataKey="day" tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
              <YAxis domain={[4, 10]} tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: "hsl(220, 22%, 9%)", border: "1px solid hsl(220, 18%, 20%)", borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: "hsl(38, 20%, 92%)" }}
              />
              <Area type="monotone" dataKey="sleep" stroke="hsl(262, 45%, 65%)" fill="url(#sleepHealthGrad)" strokeWidth={2} name="Sleep (h)" />
            </AreaChart>
          </ResponsiveContainer>
        </Card>

        {/* Stress Levels */}
        <Card className="glass-card p-5 hover-glow">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="h-4 w-4 text-warning" />
            <h3 className="text-sm font-medium">Stress Level — 7 Days</h3>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={weekly}>
              <defs>
                <linearGradient id="stressHealthGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="hsl(38, 85%, 58%)" stopOpacity={0.25} />
                  <stop offset="100%" stopColor="hsl(38, 85%, 58%)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 15%)" opacity={0.4} />
              <XAxis dataKey="day" tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: "hsl(220, 22%, 9%)", border: "1px solid hsl(220, 18%, 20%)", borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: "hsl(38, 20%, 92%)" }}
              />
              <Area type="monotone" dataKey="stress" stroke="hsl(38, 85%, 58%)" fill="url(#stressHealthGrad)" strokeWidth={2} name="Stress %" />
            </AreaChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Brain-Health Correlations */}
      <Card className="glass-card p-5 hover-glow">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-medium">Brain-Health Correlations</h3>
          </div>
          <Badge variant={apiConnected ? "default" : "secondary"} className="text-[10px]">
            {apiConnected ? "Live Data" : "Simulated"}
          </Badge>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {insights.map((insight, i) => (
            <div
              key={i}
              className="p-4 rounded-xl"
              style={{
                background: "hsl(220, 22%, 8%)",
                border: "1px solid hsl(220, 18%, 13%)",
              }}
            >
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-medium text-foreground">{insight.title}</h4>
                <span
                  className="text-[10px] font-mono px-2 py-0.5 rounded-full"
                  style={{
                    background: `hsl(152, 60%, 48%, ${insight.correlation_strength * 0.2})`,
                    color: "hsl(152, 60%, 48%)",
                  }}
                >
                  r={insight.correlation_strength.toFixed(2)}
                </span>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {insight.description}
              </p>
              <div className="flex gap-2 mt-2">
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary">
                  {insight.brain_metric.replace(/_/g, " ")}
                </span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-accent/10 text-accent">
                  {insight.health_metric.replace(/_/g, " ")}
                </span>
                {insight.evidence_count > 0 && (
                  <span className="text-[10px] text-muted-foreground/60">
                    {insight.evidence_count} data points
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </Card>
    </main>
  );
}
