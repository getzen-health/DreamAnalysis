import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from "recharts";
import { Sparkles, Brain, Moon, Heart, TrendingUp, Loader2, Lightbulb, Zap, Bed } from "lucide-react";

export default function Insights() {
  const userId = "demo-user";

  // Pre-dream EEG patterns
  const preDreamData = [
    { time: "-30m", theta: 25, alpha: 35, beta: 20, delta: 15, gamma: 5 },
    { time: "-25m", theta: 28, alpha: 32, beta: 18, delta: 18, gamma: 4 },
    { time: "-20m", theta: 32, alpha: 28, beta: 15, delta: 22, gamma: 3 },
    { time: "-15m", theta: 35, alpha: 22, beta: 12, delta: 28, gamma: 3 },
    { time: "-10m", theta: 38, alpha: 18, beta: 14, delta: 25, gamma: 5 },
    { time: "-5m", theta: 42, alpha: 12, beta: 18, delta: 20, gamma: 8 },
    { time: "REM", theta: 45, alpha: 8, beta: 22, delta: 12, gamma: 13 },
  ];

  const dreamMoodCorrelation = [
    { metric: "Sleep Quality", positive: 85, negative: 45 },
    { metric: "Dream Vividness", positive: 72, negative: 28 },
    { metric: "Lucidity", positive: 65, negative: 15 },
    { metric: "Next-Day Energy", positive: 78, negative: 52 },
    { metric: "Next-Day Mood", positive: 82, negative: 38 },
    { metric: "Creativity", positive: 70, negative: 40 },
  ];

  const radarData = [
    { subject: "Dream Recall", A: 78, B: 55 },
    { subject: "Lucidity", A: 45, B: 30 },
    { subject: "Emotional Depth", A: 82, B: 60 },
    { subject: "Symbol Richness", A: 68, B: 45 },
    { subject: "Sleep Quality", A: 85, B: 70 },
    { subject: "Mood Impact", A: 72, B: 50 },
  ];

  const weeklyInsights = [
    {
      icon: Lightbulb,
      title: "Enhanced Dream Recall This Week",
      description: "Your dream recall has improved 23% compared to last week. The increased theta activity before sleep correlates with better dream memory formation.",
      type: "success" as const,
    },
    {
      icon: Brain,
      title: "Optimal Brain State for Creativity",
      description: "Your alpha-theta ratio during early morning hours suggests a peak creative window between 6-8 AM. Consider journaling or brainstorming during this time.",
      type: "primary" as const,
    },
    {
      icon: Heart,
      title: "Emotional Processing Through Dreams",
      description: "Dreams with water symbolism appeared 4 times this week, correlating with periods of emotional processing. Your stress levels decreased 15% following these dreams.",
      type: "secondary" as const,
    },
    {
      icon: Bed,
      title: "Sleep Architecture Improvement",
      description: "Your deep sleep (N3) duration increased by 12 minutes on average. This correlates with reduced daytime fatigue and improved focus scores.",
      type: "success" as const,
    },
  ];

  const colorMap = {
    success: "bg-success/10 border-success/30 text-success",
    primary: "bg-primary/10 border-primary/30 text-primary",
    secondary: "bg-secondary/10 border-secondary/30 text-secondary",
    warning: "bg-warning/10 border-warning/30 text-warning",
  };

  return (
    <main className="p-4 md:p-6 space-y-6">
      {/* AI Weekly Insights */}
      <Card className="glass-card p-6 rounded-xl hover-glow">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-futuristic font-semibold flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-secondary" />
            AI Weekly Insights
          </h3>
          <Badge variant="outline" className="border-secondary/30 text-secondary">
            Week of {new Date().toLocaleDateString("en-US", { month: "short", day: "numeric" })}
          </Badge>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {weeklyInsights.map((insight, i) => {
            const Icon = insight.icon;
            return (
              <div key={i} className={`flex items-start gap-3 p-4 rounded-lg border ${colorMap[insight.type]}`}>
                <Icon className="h-5 w-5 mt-0.5 flex-shrink-0" />
                <div>
                  <h4 className="font-semibold text-sm mb-1">{insight.title}</h4>
                  <p className="text-xs text-foreground/70">{insight.description}</p>
                </div>
              </div>
            );
          })}
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pre-Dream EEG Patterns */}
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-lg font-futuristic font-semibold mb-4 flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            Your Brain Before Dreaming
          </h3>
          <p className="text-xs text-foreground/50 mb-4">EEG band power changes in the 30 minutes before REM onset</p>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={preDreamData}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
              <XAxis dataKey="time" tick={{ fontSize: 10 }} stroke="hsl(var(--foreground))" opacity={0.5} />
              <YAxis tick={{ fontSize: 10 }} stroke="hsl(var(--foreground))" opacity={0.5} />
              <Tooltip contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8 }} />
              <Line type="monotone" dataKey="theta" stroke="hsl(195, 100%, 50%)" strokeWidth={2} name="Theta" />
              <Line type="monotone" dataKey="alpha" stroke="hsl(120, 100%, 55%)" strokeWidth={2} name="Alpha" />
              <Line type="monotone" dataKey="beta" stroke="hsl(45, 100%, 50%)" strokeWidth={2} name="Beta" />
              <Line type="monotone" dataKey="delta" stroke="hsl(270, 70%, 65%)" strokeWidth={2} name="Delta" />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        {/* Dream Profile Radar */}
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-lg font-futuristic font-semibold mb-4 flex items-center gap-2">
            <Moon className="h-5 w-5 text-secondary" />
            Dream Profile
          </h3>
          <p className="text-xs text-foreground/50 mb-4">This week (blue) vs. average (gray)</p>
          <ResponsiveContainer width="100%" height={250}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="hsl(var(--border))" opacity={0.3} />
              <PolarAngleAxis dataKey="subject" tick={{ fontSize: 10, fill: "hsl(var(--foreground))", opacity: 0.7 }} />
              <PolarRadiusAxis tick={{ fontSize: 8 }} domain={[0, 100]} />
              <Radar name="This Week" dataKey="A" stroke="hsl(195, 100%, 50%)" fill="hsl(195, 100%, 50%)" fillOpacity={0.3} />
              <Radar name="Average" dataKey="B" stroke="hsl(var(--foreground))" fill="hsl(var(--foreground))" fillOpacity={0.1} />
            </RadarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Dream-Mood Connection */}
      <Card className="glass-card p-6 rounded-xl hover-glow">
        <h3 className="text-lg font-futuristic font-semibold mb-4 flex items-center gap-2">
          <Heart className="h-5 w-5 text-success" />
          Dream-Mood Connection
        </h3>
        <p className="text-xs text-foreground/50 mb-4">How dream quality affects next-day metrics (positive vs negative dreams)</p>
        <div className="space-y-3">
          {dreamMoodCorrelation.map((item, i) => (
            <div key={i} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="text-foreground/80">{item.metric}</span>
                <div className="flex gap-4">
                  <span className="text-xs text-success font-mono">+{item.positive}%</span>
                  <span className="text-xs text-destructive font-mono">{item.negative}%</span>
                </div>
              </div>
              <div className="flex gap-1 h-2">
                <div className="bg-success/60 rounded-l-full h-full transition-all duration-500" style={{ width: `${item.positive}%` }} />
                <div className="bg-destructive/40 rounded-r-full h-full transition-all duration-500" style={{ width: `${item.negative}%` }} />
              </div>
            </div>
          ))}
        </div>
        <div className="flex gap-4 mt-4 justify-center">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-success/60" />
            <span className="text-xs text-foreground/60">Positive Dreams</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-destructive/40" />
            <span className="text-xs text-foreground/60">Negative Dreams</span>
          </div>
        </div>
      </Card>
    </main>
  );
}
