import { useState, useEffect, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import { Sparkles, Brain, Moon, Heart, Lightbulb, Bed, Radio } from "lucide-react";
import { useDevice } from "@/hooks/use-device";

/* ---------- types ---------- */
interface BandHistoryPoint {
  time: string;
  theta: number;
  alpha: number;
  beta: number;
  delta: number;
  gamma: number;
}

/* ========== Component ========== */
export default function Insights() {
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const analysis = latestFrame?.analysis;

  const bandPowers = analysis?.band_powers ?? {};
  const emotions = analysis?.emotions;
  const dreamDetection = analysis?.dream_detection;
  const sleepStaging = analysis?.sleep_staging;
  const flowState = analysis?.flow_state;
  const creativity = analysis?.creativity;
  const attention = analysis?.attention;
  const stress = analysis?.stress;
  const meditation = analysis?.meditation;
  const memoryEncoding = analysis?.memory_encoding;

  // Accumulate band power history
  const [bandHistory, setBandHistory] = useState<BandHistoryPoint[]>([]);

  useEffect(() => {
    if (!isStreaming || !bandPowers.alpha) return;
    const now = new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    setBandHistory((prev) => [
      ...prev.slice(-30),
      {
        time: now,
        theta: Math.round((bandPowers.theta ?? 0) * 100),
        alpha: Math.round((bandPowers.alpha ?? 0) * 100),
        beta: Math.round((bandPowers.beta ?? 0) * 100),
        delta: Math.round((bandPowers.delta ?? 0) * 100),
        gamma: Math.round((bandPowers.gamma ?? 0) * 100),
      },
    ]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame?.timestamp]);

  // Live brain profile radar — throttled to 8s
  const [radarData, setRadarData] = useState<{ subject: string; value: number }[]>([]);
  const radarTimerRef = useRef(0);
  const RADAR_THROTTLE = 8_000;

  useEffect(() => {
    if (!isStreaming) { setRadarData([]); return; }
    const now = Date.now();
    if (now - radarTimerRef.current < RADAR_THROTTLE && radarData.length > 0) return;
    radarTimerRef.current = now;
    setRadarData([
      { subject: "Focus", value: Math.round((attention?.attention_score ?? 0) * 100) },
      { subject: "Creativity", value: Math.round((creativity?.creativity_score ?? 0) * 100) },
      { subject: "Relaxation", value: Math.round((emotions?.relaxation_index ?? 0) * 100) },
      { subject: "Memory", value: Math.round((memoryEncoding?.encoding_score ?? 0) * 100) },
      { subject: "Flow", value: Math.round((flowState?.flow_score ?? 0) * 100) },
      { subject: "Meditation", value: Math.round((meditation?.meditation_score ?? 0) * 100) },
    ]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame?.timestamp]);

  // Generate dynamic insights from live data — throttled to 12s
  type InsightItem = { icon: typeof Lightbulb; title: string; description: string; type: "success" | "primary" | "secondary" | "warning" };
  const [weeklyInsights, setWeeklyInsights] = useState<InsightItem[]>([]);
  const insightTimerRef = useRef(0);
  const INSIGHT_THROTTLE = 12_000;

  useEffect(() => {
    if (!isStreaming) { setWeeklyInsights([]); return; }
    const now = Date.now();
    if (now - insightTimerRef.current < INSIGHT_THROTTLE && weeklyInsights.length > 0) return;
    insightTimerRef.current = now;

    const insights: InsightItem[] = [];
    const focusScore = (attention?.attention_score ?? 0) * 100;
    const creativityScore = (creativity?.creativity_score ?? 0) * 100;
    const stressIndex = (stress?.stress_index ?? emotions?.stress_index ?? 0) * 100;
    const relaxIndex = (emotions?.relaxation_index ?? 0) * 100;
    const flowScore = (flowState?.flow_score ?? 0) * 100;
    const dreamProb = (dreamDetection?.probability ?? 0) * 100;
    const meditationScore = (meditation?.meditation_score ?? 0) * 100;

    if (focusScore > 60) {
      insights.push({ icon: Brain, title: "High Focus State Detected", description: `Your attention score is ${Math.round(focusScore)}%. Prefrontal beta activity indicates strong concentration. This is ideal for analytical tasks and deep work.`, type: "primary" });
    } else if (focusScore < 30) {
      insights.push({ icon: Brain, title: "Low Focus — Consider a Break", description: `Attention score at ${Math.round(focusScore)}%. Your brain may benefit from a short break or change of activity to restore focus.`, type: "warning" });
    }
    if (creativityScore > 50) {
      insights.push({ icon: Lightbulb, title: "Creative State Active", description: `Creativity at ${Math.round(creativityScore)}%. Your theta-alpha ratio suggests heightened divergent thinking. Great time for brainstorming or creative work.`, type: "success" });
    }
    if (stressIndex > 50) {
      insights.push({ icon: Heart, title: "Elevated Stress Detected", description: `Stress index at ${Math.round(stressIndex)}% while relaxation is ${Math.round(relaxIndex)}%. Consider a breathing exercise — deep breaths can shift your neural balance within minutes.`, type: "warning" });
    } else if (relaxIndex > 60) {
      insights.push({ icon: Heart, title: "Calm & Balanced", description: `Relaxation at ${Math.round(relaxIndex)}% with low stress (${Math.round(stressIndex)}%). Your autonomic nervous system is in a parasympathetic state — great for recovery.`, type: "success" });
    }
    if (flowScore > 60) {
      insights.push({ icon: Sparkles, title: "Flow State Achieved", description: `Flow score at ${Math.round(flowScore)}%. You're in the zone — high focus with moderate arousal and low stress. Protect this state by minimizing interruptions.`, type: "success" });
    }
    if (dreamProb > 40) {
      insights.push({ icon: Moon, title: "Dream-Like Brain Patterns", description: `Dream probability at ${Math.round(dreamProb)}%. Your theta-dominant pattern resembles REM-like activity. This may indicate a hypnagogic state.`, type: "secondary" });
    }
    if (meditationScore > 50) {
      insights.push({ icon: Bed, title: "Deep Meditative State", description: `Meditation score at ${Math.round(meditationScore)}%. Strong alpha coherence with low beta. Your brain is in a restful yet aware state.`, type: "primary" });
    }
    if (insights.length === 0) {
      insights.push({ icon: Brain, title: "Balanced Neural Activity", description: "All brain metrics are within normal ranges. Your neural patterns show a healthy balance of activity across frequency bands.", type: "primary" });
    }
    setWeeklyInsights(insights.slice(0, 4));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame?.timestamp]);

  const colorMap = {
    success: "bg-success/10 border-success/30 text-success",
    primary: "bg-primary/10 border-primary/30 text-primary",
    secondary: "bg-secondary/10 border-secondary/30 text-secondary",
    warning: "bg-warning/10 border-warning/30 text-warning",
  };

  return (
    <main className="p-4 md:p-6 space-y-6">
      {/* Connection Banner */}
      {!isStreaming && (
        <div className="p-4 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning flex items-center gap-3">
          <Radio className="h-4 w-4 shrink-0" />
          Connect your Muse 2 from the sidebar to see live brain insights.
        </div>
      )}

      {/* AI Insights */}
      {weeklyInsights.length > 0 && (
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-secondary" />
              AI Brain Insights
            </h3>
            {isStreaming && (
              <Badge variant="outline" className="border-primary/30 text-primary animate-pulse">
                LIVE
              </Badge>
            )}
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
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* EEG Band Powers Over Time */}
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            Brain Wave Trends
            {isStreaming && (
              <span className="ml-auto text-[10px] font-mono text-primary animate-pulse">LIVE</span>
            )}
          </h3>
          {bandHistory.length < 2 ? (
            <div className="h-[250px] flex items-center justify-center text-sm text-muted-foreground">
              {isStreaming ? "Collecting EEG data..." : "Connect device to see brain wave trends"}
            </div>
          ) : (
            <>
              <p className="text-xs text-foreground/50 mb-4">EEG band power changes during this session</p>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={bandHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 18%, 22%)" opacity={0.6} />
                  <XAxis dataKey="time" tick={{ fontSize: 10, fill: "hsl(220, 12%, 52%)" }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 10, fill: "hsl(220, 12%, 52%)" }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: "hsl(220, 22%, 9%)", border: "1px solid hsl(220, 18%, 20%)", borderRadius: 8, fontSize: 12 }} labelStyle={{ color: "hsl(38, 20%, 92%)" }} />
                  <Line type="monotone" dataKey="theta" stroke="hsl(195, 100%, 50%)" strokeWidth={2.5} dot={false} name="Theta" />
                  <Line type="monotone" dataKey="alpha" stroke="hsl(152, 60%, 48%)" strokeWidth={2.5} dot={false} name="Alpha" />
                  <Line type="monotone" dataKey="beta" stroke="hsl(38, 85%, 58%)" strokeWidth={2.5} dot={false} name="Beta" />
                  <Line type="monotone" dataKey="delta" stroke="hsl(262, 45%, 65%)" strokeWidth={2.5} dot={false} name="Delta" />
                </LineChart>
              </ResponsiveContainer>
            </>
          )}
        </Card>

        {/* Brain Profile Radar */}
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-secondary" />
            Brain Profile
            {isStreaming && (
              <span className="ml-auto text-[10px] font-mono text-primary animate-pulse">LIVE</span>
            )}
          </h3>
          {radarData.length === 0 ? (
            <div className="h-[250px] flex items-center justify-center text-sm text-muted-foreground">
              Connect device to see brain profile
            </div>
          ) : (
            <>
              <p className="text-xs text-foreground/50 mb-4">Current cognitive capabilities</p>
              <ResponsiveContainer width="100%" height={250}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="hsl(220, 18%, 25%)" opacity={0.6} />
                  <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fill: "hsl(220, 12%, 60%)" }} />
                  <PolarRadiusAxis tick={{ fontSize: 8, fill: "hsl(220, 12%, 45%)" }} domain={[0, 100]} />
                  <Radar name="Current" dataKey="value" stroke="hsl(152, 60%, 48%)" fill="hsl(152, 60%, 48%)" fillOpacity={0.25} strokeWidth={2} />
                </RadarChart>
              </ResponsiveContainer>
            </>
          )}
        </Card>
      </div>

      {/* Live Metric Summary — uses same radar throttle so numbers stay stable */}
      {isStreaming && radarData.length > 0 && (
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Heart className="h-5 w-5 text-success" />
            Current Brain State Summary
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {radarData.map((m) => (
              <div key={m.subject} className="text-center">
                <p className="text-xs text-muted-foreground">{m.subject}</p>
                <p className="text-xl font-mono font-bold text-primary">{m.value}%</p>
              </div>
            ))}
          </div>
        </Card>
      )}
    </main>
  );
}
