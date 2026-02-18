import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScoreCircle } from "@/components/score-circle";
import {
  AreaChart,
  Area,
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
  Brain,
  Sparkles,
  Zap,
  TrendingUp,
  Radio,
} from "lucide-react";
import { useDevice } from "@/hooks/use-device";

/* ---------- types ---------- */
interface HealthPoint {
  time: string;
  stress: number;
  focus: number;
  relaxation: number;
  cogLoad: number;
}

/* ========== Component ========== */
export default function HealthAnalytics() {
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const analysis = latestFrame?.analysis;

  // Extract live metrics from ML models
  const emotions = analysis?.emotions;
  const stress = analysis?.stress;
  const attention = analysis?.attention;
  const cognitiveLoad = analysis?.cognitive_load;
  const flowState = analysis?.flow_state;
  const creativity = analysis?.creativity;
  const drowsiness = analysis?.drowsiness;
  const memoryEncoding = analysis?.memory_encoding;

  // Current metrics
  const stressIndex = isStreaming ? Math.round((stress?.stress_index ?? emotions?.stress_index ?? 0) * 100) : 0;
  const focusScore = isStreaming ? Math.round((attention?.attention_score ?? emotions?.focus_index ?? 0) * 100) : 0;
  const relaxScore = isStreaming ? Math.round((emotions?.relaxation_index ?? 0) * 100) : 0;
  const cogLoadIndex = isStreaming ? Math.round((cognitiveLoad?.load_index ?? 0) * 100) : 0;
  const flowScore = isStreaming ? Math.round((flowState?.flow_score ?? 0) * 100) : 0;
  const creativityScore = isStreaming ? Math.round((creativity?.creativity_score ?? 0) * 100) : 0;
  const drowsinessIndex = isStreaming ? Math.round((drowsiness?.drowsiness_index ?? 0) * 100) : 0;
  const memoryScore = isStreaming ? Math.round((memoryEncoding?.encoding_score ?? 0) * 100) : 0;

  // Composite scores
  const brainHealthScore = isStreaming
    ? Math.round(focusScore * 0.25 + relaxScore * 0.25 + (100 - stressIndex) * 0.25 + flowScore * 0.25)
    : 0;
  const cognitiveScore = isStreaming
    ? Math.round(focusScore * 0.3 + creativityScore * 0.25 + memoryScore * 0.25 + (100 - drowsinessIndex) * 0.2)
    : 0;
  const wellbeingScore = isStreaming
    ? Math.round(relaxScore * 0.35 + (100 - stressIndex) * 0.35 + flowScore * 0.3)
    : 0;

  // Accumulate timeline
  const [timeline, setTimeline] = useState<HealthPoint[]>([]);

  useEffect(() => {
    if (!isStreaming || !analysis) return;
    const now = new Date().toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
    setTimeline((prev) => [
      ...prev.slice(-60),
      {
        time: now,
        stress: stressIndex,
        focus: focusScore,
        relaxation: relaxScore,
        cogLoad: cogLoadIndex,
      },
    ]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame?.timestamp]);

  // Derive insights from current live data
  const insights = [];
  if (isStreaming) {
    if (flowScore > 60) {
      insights.push({
        title: "Flow State Active",
        description: `Your flow score is ${flowScore}%. This is optimal for deep work and creative problem solving. Minimize interruptions.`,
        strength: flowScore / 100,
        brain: "flow_score",
        health: "focus",
      });
    }
    if (stressIndex > 50) {
      insights.push({
        title: "Elevated Stress Detected",
        description: `Stress index at ${stressIndex}%. Consider a breathing exercise or short break. High stress reduces cognitive performance.`,
        strength: stressIndex / 100,
        brain: "stress_index",
        health: "relaxation",
      });
    }
    if (creativityScore > 50) {
      insights.push({
        title: "Creative State Detected",
        description: `Creativity score at ${creativityScore}%. Your theta-alpha ratio suggests heightened divergent thinking. Good time for brainstorming.`,
        strength: creativityScore / 100,
        brain: "creativity",
        health: "cognitive",
      });
    }
    if (memoryScore > 60) {
      insights.push({
        title: "Strong Memory Encoding",
        description: `Memory encoding score at ${memoryScore}%. Your brain is actively consolidating information. Great time for learning.`,
        strength: memoryScore / 100,
        brain: "memory",
        health: "encoding",
      });
    }
    if (drowsinessIndex > 60) {
      insights.push({
        title: "Drowsiness Alert",
        description: `Drowsiness index at ${drowsinessIndex}%. Consider a short break or physical activity to restore alertness.`,
        strength: drowsinessIndex / 100,
        brain: "drowsiness",
        health: "alertness",
      });
    }
    if (insights.length === 0) {
      insights.push({
        title: "Balanced Brain State",
        description: "Your neural patterns are balanced. All metrics within normal ranges. Keep up the good work!",
        strength: 0.5,
        brain: "overall",
        health: "balance",
      });
    }
  }

  return (
    <main className="p-6 space-y-6 max-w-5xl">
      {/* Connection Banner */}
      {!isStreaming && (
        <div className="p-4 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning flex items-center gap-3">
          <Radio className="h-4 w-4 shrink-0" />
          Connect your Muse 2 from the sidebar to see live health analytics.
        </div>
      )}

      {/* Score Gauges */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="score-card p-4 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={brainHealthScore}
            label="Brain Health"
            gradientId="grad-brain-health"
            colorFrom="hsl(152, 60%, 48%)"
            colorTo="hsl(200, 70%, 55%)"
            size="sm"
          />
          <p className="text-xs text-muted-foreground mt-1">Focus + Relaxation + Low Stress</p>
        </div>
        <div className="score-card p-4 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={cognitiveScore}
            label="Cognitive"
            gradientId="grad-cognitive"
            colorFrom="hsl(262, 45%, 65%)"
            colorTo="hsl(220, 50%, 50%)"
            size="sm"
          />
          <p className="text-xs text-muted-foreground mt-1">Focus + Creativity + Memory</p>
        </div>
        <div className="score-card p-4 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={wellbeingScore}
            label="Wellbeing"
            gradientId="grad-wellbeing"
            colorFrom="hsl(38, 85%, 58%)"
            colorTo="hsl(25, 85%, 55%)"
            size="sm"
          />
          <p className="text-xs text-muted-foreground mt-1">Relaxation + Low Stress + Flow</p>
        </div>
      </div>

      {/* Vital Stats Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { icon: Activity, label: "Stress", value: `${stressIndex}%`, color: stressIndex > 50 ? "text-warning" : "text-success" },
          { icon: Brain, label: "Focus", value: `${focusScore}%`, color: "text-primary" },
          { icon: Zap, label: "Flow", value: `${flowScore}%`, color: "text-accent" },
          { icon: Heart, label: "Relaxation", value: `${relaxScore}%`, color: "text-success" },
          { icon: TrendingUp, label: "Creativity", value: `${creativityScore}%`, color: "text-secondary" },
          { icon: Brain, label: "Memory", value: `${memoryScore}%`, color: "text-primary" },
          { icon: Activity, label: "Cog Load", value: `${cogLoadIndex}%`, color: cogLoadIndex > 70 ? "text-warning" : "text-foreground" },
          { icon: Activity, label: "Drowsiness", value: `${drowsinessIndex}%`, color: drowsinessIndex > 60 ? "text-warning" : "text-success" },
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

      {/* Timeline */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Stress & Relaxation */}
        <Card className="glass-card p-5 hover-glow">
          <div className="flex items-center gap-2 mb-4">
            <Heart className="h-4 w-4 text-success" />
            <h3 className="text-sm font-medium">Stress vs Relaxation</h3>
            {isStreaming && (
              <span className="ml-auto text-[10px] font-mono text-primary animate-pulse">LIVE</span>
            )}
          </div>
          {timeline.length < 2 ? (
            <div className="h-[180px] flex items-center justify-center text-sm text-muted-foreground">
              {isStreaming ? "Collecting data..." : "Connect device to see trends"}
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={timeline.slice(-30)}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 15%)" opacity={0.4} />
                <XAxis dataKey="time" tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ background: "hsl(220, 22%, 9%)", border: "1px solid hsl(220, 18%, 20%)", borderRadius: 8, fontSize: 12 }}
                  labelStyle={{ color: "hsl(38, 20%, 92%)" }}
                />
                <Line type="monotone" dataKey="stress" stroke="hsl(38, 85%, 58%)" strokeWidth={2} dot={false} name="Stress %" />
                <Line type="monotone" dataKey="relaxation" stroke="hsl(152, 60%, 48%)" strokeWidth={2} dot={false} name="Relaxation %" />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>

        {/* Focus & Cognitive Load */}
        <Card className="glass-card p-5 hover-glow">
          <div className="flex items-center gap-2 mb-4">
            <Brain className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-medium">Focus & Cognitive Load</h3>
            {isStreaming && (
              <span className="ml-auto text-[10px] font-mono text-primary animate-pulse">LIVE</span>
            )}
          </div>
          {timeline.length < 2 ? (
            <div className="h-[180px] flex items-center justify-center text-sm text-muted-foreground">
              {isStreaming ? "Collecting data..." : "Connect device to see trends"}
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={timeline.slice(-30)}>
                <defs>
                  <linearGradient id="focusGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="hsl(200, 70%, 55%)" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="hsl(200, 70%, 55%)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 15%)" opacity={0.4} />
                <XAxis dataKey="time" tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "hsl(220, 12%, 42%)" }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ background: "hsl(220, 22%, 9%)", border: "1px solid hsl(220, 18%, 20%)", borderRadius: 8, fontSize: 12 }}
                  labelStyle={{ color: "hsl(38, 20%, 92%)" }}
                />
                <Area type="monotone" dataKey="focus" stroke="hsl(200, 70%, 55%)" fill="url(#focusGrad)" strokeWidth={2} dot={false} name="Focus %" />
                <Line type="monotone" dataKey="cogLoad" stroke="hsl(262, 45%, 65%)" strokeWidth={1.5} dot={false} name="Cog Load %" strokeDasharray="4 4" />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>

      {/* Brain-Health Insights */}
      {insights.length > 0 && (
        <Card className="glass-card p-5 hover-glow">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-primary" />
              <h3 className="text-sm font-medium">Brain-Health Insights</h3>
            </div>
            <Badge variant={isStreaming ? "default" : "secondary"} className="text-[10px]">
              {isStreaming ? "Live Data" : "No Data"}
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
                      background: `hsl(152, 60%, 48%, ${insight.strength * 0.2})`,
                      color: "hsl(152, 60%, 48%)",
                    }}
                  >
                    {Math.round(insight.strength * 100)}%
                  </span>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed">{insight.description}</p>
                <div className="flex gap-2 mt-2">
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary">
                    {insight.brain.replace(/_/g, " ")}
                  </span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-accent/10 text-accent">
                    {insight.health}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </main>
  );
}
