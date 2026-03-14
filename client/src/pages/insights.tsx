import { useState, useEffect, useRef } from "react";
import { Link } from "wouter";
import { getParticipantId } from "@/lib/participant";
import { resolveUrl } from "@/lib/queryClient";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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
import {
  Sparkles,
  Brain,
  Moon,
  Heart,
  Lightbulb,
  Bed,
  Radio,
  Wind,
  Target,
  Activity,
  Mic,
} from "lucide-react";
import { useDevice } from "@/hooks/use-device";
import { useQuery } from "@tanstack/react-query";

interface BandHistoryPoint {
  time: string;
  calm: number;   // alpha %
  alert: number;  // beta %
  creative: number; // theta %
}

function getBrainStateNarrative(
  attention: Record<string, unknown> | undefined,
  emotions: Record<string, unknown> | undefined,
  stress: Record<string, unknown> | undefined,
  flowState: Record<string, unknown> | undefined,
  creativity: Record<string, unknown> | undefined,
  meditation: Record<string, unknown> | undefined,
  bandPowers: Record<string, number>
): { headline: string; story: string; state: "peak" | "relaxed" | "stressed" | "creative" | "balanced" } {
  const focusScore = ((attention?.attention_score as number) ?? 0) * 100;
  const stressIndex = ((stress?.stress_index as number) ?? (emotions?.stress_index as number) ?? 0) * 100;
  const relaxIndex = ((emotions?.relaxation_index as number) ?? 0) * 100;
  const flowScore = ((flowState?.flow_score as number) ?? 0) * 100;
  const creativityScore = ((creativity?.creativity_score as number) ?? 0) * 100;
  const meditationScore = ((meditation?.meditation_score as number) ?? 0) * 100;

  const alpha = (bandPowers.alpha ?? 0) * 100;
  const beta = (bandPowers.beta ?? 0) * 100;
  const theta = (bandPowers.theta ?? 0) * 100;

  if (flowScore > 60 && focusScore > 60 && stressIndex < 40) {
    return {
      headline: "You're in a flow state",
      story: `Your brain is firing on all cylinders right now. High focus (${Math.round(focusScore)}%), low stress (${Math.round(stressIndex)}%), and that rare combination of engaged beta activity with calm alpha underneath — this is the neuroscience of being "in the zone." Protect this moment: silence your phone, close unnecessary tabs, and stay with what you're doing. Flow states typically last 90–120 minutes before the brain needs recovery.`,
      state: "peak",
    };
  }
  if (meditationScore > 60 || (alpha > 25 && stressIndex < 30)) {
    return {
      headline: "Your brain is deeply calm",
      story: `Strong alpha waves (${Math.round(alpha)}%) dominate your current EEG, indicating your brain has shifted into a restful but aware state. Your prefrontal cortex is quiet, stress markers are low (${Math.round(stressIndex)}%), and your parasympathetic nervous system is active. This is the ideal state for reflection, reading, or creative incubation — let ideas arrive rather than chasing them.`,
      state: "relaxed",
    };
  }
  if (creativityScore > 55 || theta > 20) {
    return {
      headline: "You're in a creative state",
      story: `Elevated theta activity (${Math.round(theta)}%) with moderate alpha suggests your brain is in a semi-relaxed, deeply creative mode. This pattern typically appears during moments of insight, daydreaming, and creative problem-solving. The prefrontal cortex is relaxed enough to let associations flow freely. If you have a creative challenge, now is the time to work on it.`,
      state: "creative",
    };
  }
  if (stressIndex > 55) {
    return {
      headline: "Your brain is working hard",
      story: `High-beta activity (${Math.round(beta)}%) and elevated stress index (${Math.round(stressIndex)}%) indicate your nervous system is in an activated state. This isn't necessarily bad — it means your brain is engaged and alert. But sustained high-beta without recovery depletes the prefrontal cortex. Consider a 5-minute breathing pause to lower cortisol before continuing. Your relaxation score of ${Math.round(relaxIndex)}% tells you how much recovery capacity you have left.`,
      state: "stressed",
    };
  }
  return {
    headline: "Your brain is in a balanced state",
    story: `Your EEG shows a healthy distribution across all frequency bands — alpha (${Math.round(alpha)}% calm), beta (${Math.round(beta)}% alert), theta (${Math.round(theta)}% creative). No single state dominates, which often means your brain is in a receptive, learning-friendly mode. This is a good time for general tasks, social interaction, or taking in new information.`,
    state: "balanced",
  };
}

function getRecommendedActions(
  stressIndex: number,
  focusScore: number,
  creativityScore: number,
  flowScore: number
): { icon: typeof Target; label: string; description: string }[] {
  const actions: { icon: typeof Target; label: string; description: string }[] = [];

  if (stressIndex > 50) {
    actions.push({
      icon: Wind,
      label: "4-7-8 Breathing",
      description: "2 minutes of this breathing pattern will reduce cortisol within one cycle.",
    });
  }
  if (flowScore > 60) {
    actions.push({
      icon: Target,
      label: "Deep Work Session",
      description: "You're in flow — start a 25-min Pomodoro on your most important task.",
    });
  }
  if (focusScore < 40) {
    actions.push({
      icon: Activity,
      label: "Movement Break",
      description: "5 minutes of walking increases prefrontal blood flow and sharpens attention.",
    });
  }
  if (creativityScore > 50) {
    actions.push({
      icon: Lightbulb,
      label: "Brainstorm Now",
      description: "Your theta-dominant state is ideal for creative thinking — capture ideas freely.",
    });
  }
  if (actions.length < 2) {
    actions.push({
      icon: Brain,
      label: "Log This Session",
      description: "Recording how you feel right now helps calibrate your personal brain baseline.",
    });
    actions.push({
      icon: Moon,
      label: "Plan Tomorrow",
      description: "Writing a 3-item tomorrow list offloads from prefrontal cortex — better sleep.",
    });
  }
  return actions.slice(0, 3);
}

const CURRENT_USER = getParticipantId();

function voiceNarrative(voice: Record<string, unknown>): { headline: string; story: string } {
  const emotion = (voice.emotion as string) ?? "neutral";
  const valence = (voice.valence as number) ?? 0;
  const arousal = (voice.arousal as number) ?? 0.5;
  const stress = (voice.stress_from_watch as number) ?? null;

  const moodLabel =
    valence > 0.3 ? "positive and energised"
    : valence < -0.2 ? "heavy or low"
    : "neutral and steady";

  const energyLabel = arousal > 0.6 ? "high energy" : arousal < 0.35 ? "low energy" : "moderate energy";

  const stressPart = stress !== null
    ? stress > 6 ? " Stress markers are elevated — a short breathing exercise may help."
    : stress < 3 ? " Stress reads low, which is a great sign."
    : ""
    : "";

  return {
    headline: `Voice check-in: ${emotion} — ${moodLabel}`,
    story: `Your most recent voice snapshot shows a ${moodLabel} emotional state with ${energyLabel}. Detected emotion: ${emotion}.${stressPart} This page already works from voice and health signals; EEG can add deeper live neural detail later.`,
  };
}

export default function Insights() {
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";

  const { data: latestVoice } = useQuery<Record<string, unknown> | null>({
    queryKey: ["voice-insights-fallback", CURRENT_USER],
    queryFn: async () => {
      const res = await fetch(resolveUrl(`/api/ml/voice-watch/latest/${CURRENT_USER}`));
      if (!res.ok) return null;
      const data = await res.json();
      if (!data || Array.isArray(data) || typeof data !== "object") return null;
      return data as Record<string, unknown>;
    },
    staleTime: 60_000,
    retry: false,
    enabled: !isStreaming,
  });
  const analysis = latestFrame?.analysis;

  const bandPowers = (analysis?.band_powers as Record<string, number>) ?? {};
  const emotions = analysis?.emotions as Record<string, unknown> | undefined;
  const dreamDetection = analysis?.dream_detection as Record<string, unknown> | undefined;
  const sleepStaging = analysis?.sleep_staging as Record<string, unknown> | undefined;
  const flowState = analysis?.flow_state as Record<string, unknown> | undefined;
  const creativity = analysis?.creativity as Record<string, unknown> | undefined;
  const attention = analysis?.attention as Record<string, unknown> | undefined;
  const stress = analysis?.stress as Record<string, unknown> | undefined;
  const meditation = analysis?.meditation as Record<string, unknown> | undefined;
  const memoryEncoding = analysis?.memory_encoding as Record<string, unknown> | undefined;

  // Rolling band power history (calm/alert/creative in % of total)
  const [bandHistory, setBandHistory] = useState<BandHistoryPoint[]>([]);

  useEffect(() => {
    if (!isStreaming || !bandPowers.alpha) return;
    const total =
      (bandPowers.delta ?? 0) +
      (bandPowers.theta ?? 0) +
      (bandPowers.alpha ?? 0) +
      (bandPowers.beta ?? 0) +
      (bandPowers.gamma ?? 0) +
      0.001;
    const now = new Date().toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    setBandHistory((prev) => [
      ...prev.slice(-40),
      {
        time: now,
        calm: Math.round(Math.min(60, (bandPowers.alpha / total) * 100)),
        alert: Math.round(Math.min(60, (bandPowers.beta / total) * 100)),
        creative: Math.round(Math.min(60, (bandPowers.theta / total) * 100)),
      },
    ]);
  }, [latestFrame?.timestamp]);

  // Brain profile radar — throttled to 8s
  const [radarData, setRadarData] = useState<{ subject: string; value: number }[]>([]);
  const radarTimerRef = useRef(0);

  useEffect(() => {
    if (!isStreaming) {
      setRadarData([]);
      return;
    }
    const now = Date.now();
    if (now - radarTimerRef.current < 8_000 && radarData.length > 0) return;
    radarTimerRef.current = now;
    setRadarData([
      { subject: "Focus", value: Math.round(((attention?.attention_score as number) ?? 0) * 100) },
      { subject: "Creative", value: Math.round(((creativity?.creativity_score as number) ?? 0) * 100) },
      { subject: "Calm", value: Math.round(((emotions?.relaxation_index as number) ?? 0) * 100) },
      { subject: "Memory", value: Math.round(((memoryEncoding?.encoding_score as number) ?? 0) * 100) },
      { subject: "Flow", value: Math.round(((flowState?.flow_score as number) ?? 0) * 100) },
      { subject: "Meditate", value: Math.round(((meditation?.meditation_score as number) ?? 0) * 100) },
    ]);
  }, [latestFrame?.timestamp]);

  // Dynamic insights — throttled to 12s
  type InsightItem = {
    icon: typeof Lightbulb;
    title: string;
    description: string;
    type: "success" | "primary" | "secondary" | "warning";
  };
  const [liveInsights, setLiveInsights] = useState<InsightItem[]>([]);
  const insightTimerRef = useRef(0);

  useEffect(() => {
    if (!isStreaming) {
      setLiveInsights([]);
      return;
    }
    const now = Date.now();
    if (now - insightTimerRef.current < 12_000 && liveInsights.length > 0) return;
    insightTimerRef.current = now;

    const items: InsightItem[] = [];
    const focusScore = ((attention?.attention_score as number) ?? 0) * 100;
    const creativityScore = ((creativity?.creativity_score as number) ?? 0) * 100;
    const stressIndex = ((stress?.stress_index as number) ?? (emotions?.stress_index as number) ?? 0) * 100;
    const relaxIndex = ((emotions?.relaxation_index as number) ?? 0) * 100;
    const flowScore = ((flowState?.flow_score as number) ?? 0) * 100;
    const dreamProb = ((dreamDetection?.probability as number) ?? 0) * 100;
    const meditationScore = ((meditation?.meditation_score as number) ?? 0) * 100;

    if (focusScore > 60)
      items.push({ icon: Brain, title: "High Focus Detected", description: `Attention at ${Math.round(focusScore)}%. Your prefrontal beta is elevated — ideal for deep, analytical work. This is the brain in executive mode.`, type: "primary" });
    else if (focusScore < 30)
      items.push({ icon: Brain, title: "Low Focus Right Now", description: `Attention at ${Math.round(focusScore)}%. Your prefrontal cortex may need stimulation — try standing up, cold water, or switching tasks briefly.`, type: "warning" });

    if (creativityScore > 50)
      items.push({ icon: Lightbulb, title: "Creative State Active", description: `Creativity at ${Math.round(creativityScore)}%. Elevated theta-alpha ratio signals your brain is in a diffuse, associative mode — the source of original ideas.`, type: "success" });

    if (stressIndex > 50)
      items.push({ icon: Heart, title: "Elevated Stress Markers", description: `Stress at ${Math.round(stressIndex)}%. High-beta frontal activity and reduced alpha suggest sympathetic nervous system activation. A 2-minute breathing exercise can shift this quickly.`, type: "warning" });
    else if (relaxIndex > 60)
      items.push({ icon: Heart, title: "Calm & Recovered", description: `Relaxation at ${Math.round(relaxIndex)}%. Low stress (${Math.round(stressIndex)}%) and strong alpha indicate your autonomic nervous system has found its baseline. Good time for reflection.`, type: "success" });

    if (flowScore > 60)
      items.push({ icon: Sparkles, title: "Flow State Active", description: `Flow at ${Math.round(flowScore)}%. High focus, moderate arousal, and low stress — the signature of peak performance. Minimize interruptions to sustain this.`, type: "success" });

    if (dreamProb > 40)
      items.push({ icon: Moon, title: "Hypnagogic-Like Patterns", description: `Dream probability at ${Math.round(dreamProb)}%. Theta-dominant patterns similar to REM are emerging — this may indicate a hypnagogic (pre-sleep) or deep meditative state.`, type: "secondary" });

    if (meditationScore > 50)
      items.push({ icon: Bed, title: "Deep Meditative State", description: `Meditation at ${Math.round(meditationScore)}%. Coherent alpha with quiet beta is the signature of restful awareness. Your brain is recovering and integrating.`, type: "primary" });

    if (items.length === 0)
      items.push({ icon: Brain, title: "Balanced Neural Activity", description: "All brain metrics are within typical resting ranges. No dominant state detected — your neural patterns show a healthy equilibrium across all frequency bands.", type: "primary" });

    setLiveInsights(items.slice(0, 4));
  }, [latestFrame?.timestamp]);

  // Narrative
  const narrative = isStreaming
    ? getBrainStateNarrative(attention, emotions, stress, flowState, creativity, meditation, bandPowers)
    : null;

  const stressIndex = ((stress?.stress_index as number) ?? (emotions?.stress_index as number) ?? 0) * 100;
  const focusScore = ((attention?.attention_score as number) ?? 0) * 100;
  const creativityScore = ((creativity?.creativity_score as number) ?? 0) * 100;
  const flowScore = ((flowState?.flow_score as number) ?? 0) * 100;
  const recommendedActions = isStreaming
    ? getRecommendedActions(stressIndex, focusScore, creativityScore, flowScore)
    : [];

  // Voice-derived insights and actions (when no EEG)
  const voiceInsightItems: { icon: typeof Lightbulb; title: string; description: string; type: "success" | "primary" | "secondary" | "warning" }[] =
    !isStreaming && latestVoice
      ? (() => {
          const items: { icon: typeof Lightbulb; title: string; description: string; type: "success" | "primary" | "secondary" | "warning" }[] = [];
          const vValence = (latestVoice.valence as number) ?? 0;
          const vStress = (latestVoice.stress_from_watch as number) ?? null;
          const vEmotion = (latestVoice.emotion as string) ?? "neutral";
          // stress_from_watch is 0-1 scale
          if (vStress !== null && vStress > 0.6)
            items.push({ icon: Heart, title: "Elevated Voice Stress", description: `Stress markers at ${Math.round(vStress * 100)}%. Pitch variability and faster speech tempo suggest sympathetic activation. A 2-minute breathing exercise can shift this quickly.`, type: "warning" });
          else if (vStress !== null && vStress < 0.3)
            items.push({ icon: Heart, title: "Calm Vocal State", description: `Stress at ${Math.round(vStress * 100)}%. Steady pitch and relaxed tempo indicate your nervous system is regulated — good state for focused or reflective work.`, type: "success" });
          if (vValence > 0.3)
            items.push({ icon: Sparkles, title: "Positive Emotional Tone", description: `Voice detected ${vEmotion} with positive valence (${Math.round(vValence * 100)}%). Positive affect broadens attention scope and enhances creative thinking.`, type: "success" });
          else if (vValence < -0.2)
            items.push({ icon: Brain, title: "Negative Emotional Tone", description: `Voice detected ${vEmotion} state. Negative valence can narrow attention — be mindful of decisions made in this state.`, type: "warning" });
          if (items.length < 2)
            items.push({ icon: Brain, title: "Voice Baseline Captured", description: "Daily voice check-ins build an emotional baseline over time. Health data can deepen readiness and recovery insights, and EEG remains an optional live layer.", type: "primary" });
          return items.slice(0, 4);
        })()
      : [];

  const voiceRecommendedActions = !isStreaming && latestVoice
    ? getRecommendedActions(
        ((latestVoice.stress_from_watch as number) ?? 0.5) * 100,
        50,
        0,
        0,
      )
    : [];

  const colorMap = {
    success: "bg-success/10 border-success/30 text-success",
    primary: "bg-primary/10 border-primary/30 text-primary",
    secondary: "bg-secondary/10 border-secondary/30 text-secondary",
    warning: "bg-warning/10 border-warning/30 text-warning",
  };

  const stateColors: Record<string, string> = {
    peak: "from-primary/20 to-success/10 border-primary/30",
    relaxed: "from-success/20 to-primary/10 border-success/30",
    creative: "from-secondary/20 to-primary/10 border-secondary/30",
    stressed: "from-warning/20 to-destructive/10 border-warning/30",
    balanced: "from-muted/20 to-primary/10 border-border/40",
  };

  return (
    <main className="p-4 md:p-6 pb-24 space-y-6">

      {/* ── Connection Banner ── */}
      {!isStreaming && latestVoice && (
        <div className="p-4 rounded-xl border border-primary/30 bg-primary/5 text-sm text-muted-foreground flex items-center gap-3">
          <Radio className="h-4 w-4 shrink-0 text-primary" />
          Showing voice-based insights. Health and watch signals can refine this further, and EEG is optional later.
        </div>
      )}

      {/* ── Voice Fallback Narrative (no EEG) ── */}
      {!isStreaming && latestVoice && (() => {
        const vn = voiceNarrative(latestVoice);
        return (
          <Card className="glass-card p-6 rounded-xl bg-gradient-to-br from-secondary/20 to-primary/10 border-secondary/30">
            <div className="flex items-start justify-between mb-3">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Voice Check-in</p>
                <h2 className="text-xl font-bold text-foreground">{vn.headline}</h2>
              </div>
              <Badge variant="outline" className="border-secondary/30 text-secondary shrink-0 ml-4">VOICE</Badge>
            </div>
            <p className="text-sm text-foreground/75 leading-relaxed">{vn.story}</p>
          </Card>
        );
      })()}

      {/* ── Brain Narrative (live EEG) ── */}
      {narrative && (
        <Card className={`glass-card p-6 rounded-xl bg-gradient-to-br ${stateColors[narrative.state]}`}>
          <div className="flex items-start justify-between mb-3">
            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Right Now</p>
              <h2 className="text-xl font-bold text-foreground">{narrative.headline}</h2>
            </div>
            <Badge variant="outline" className="border-primary/30 text-primary animate-pulse shrink-0 ml-4">
              LIVE
            </Badge>
          </div>
          <p className="text-sm text-foreground/75 leading-relaxed">{narrative.story}</p>
        </Card>
      )}

      {/* ── Offline: empty state — no voice data, no EEG ── */}
      {!isStreaming && !latestVoice && (
        <Card className="glass-card p-8 rounded-xl text-center">
          <Mic className="h-10 w-10 text-muted-foreground/30 mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">
            Complete a voice check-in to unlock your first insights
          </h3>
          <p className="text-sm text-muted-foreground mb-6 max-w-md mx-auto">
            A 30-second voice check-in analyzes your emotional tone, stress level, and energy.
            Once recorded, this page will show personalized insights, recommended actions, and trends over time.
          </p>
          <Link href="/emotions">
            <Button size="default" className="mb-6">
              <Mic className="h-4 w-4 mr-2" />
              Start Voice Check-in
            </Button>
          </Link>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm text-muted-foreground text-left max-w-lg mx-auto">
            {[
              { icon: Brain, title: "Brain State Narrative", desc: "A plain-English story of what your brain is doing — not just numbers, but what they mean." },
              { icon: Lightbulb, title: "AI-Generated Insights", desc: "Personalized observations about your focus, creativity, stress, and flow states." },
              { icon: Target, title: "Recommended Actions", desc: "Specific things you can do right now based on your current state." },
              { icon: Activity, title: "Trends Over Time", desc: "Charts showing how your calm, alert, and creative signals shift over sessions." },
            ].map(({ icon: Icon, title, desc }) => (
              <div key={title} className="flex gap-3">
                <Icon className="h-4 w-4 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium text-foreground">{title}</p>
                  <p className="text-xs mt-0.5">{desc}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* ── AI Insight Cards ── */}
      {(liveInsights.length > 0 || voiceInsightItems.length > 0) && (
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <div className="flex items-center justify-between mb-5">
            <h3 className="text-base font-semibold flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-secondary" />
              {isStreaming ? "AI Brain Insights" : "Voice Insights"}
            </h3>
            <Badge variant="outline" className={`text-[10px] ${isStreaming ? "border-primary/30 text-primary animate-pulse" : "border-secondary/30 text-secondary"}`}>
              {isStreaming ? "LIVE" : "VOICE"}
            </Badge>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {(isStreaming ? liveInsights : voiceInsightItems).map((insight, i) => {
              const Icon = insight.icon;
              return (
                <div
                  key={i}
                  className={`flex items-start gap-3 p-4 rounded-xl border ${colorMap[insight.type]}`}
                >
                  <Icon className="h-4 w-4 mt-0.5 flex-shrink-0" />
                  <div>
                    <h4 className="font-semibold text-sm mb-1">{insight.title}</h4>
                    <p className="text-xs text-foreground/70 leading-relaxed">{insight.description}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {/* ── Recommended Actions ── */}
      {(recommendedActions.length > 0 || voiceRecommendedActions.length > 0) && (
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-base font-semibold mb-4 flex items-center gap-2">
            <Target className="h-4 w-4 text-primary" />
            Recommended Right Now
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {(isStreaming ? recommendedActions : voiceRecommendedActions).map(({ icon: Icon, label, description }, i) => (
              <div
                key={i}
                className="flex flex-col gap-2 p-4 rounded-xl border border-border/30 bg-card/30 hover:bg-card/50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <div className="w-7 h-7 rounded-lg bg-primary/15 flex items-center justify-center">
                    <Icon className="h-3.5 w-3.5 text-primary" />
                  </div>
                  <span className="text-sm font-medium">{label}</span>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed">{description}</p>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* ── Charts Row ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Band Wave Trends — simplified: only 3 meaningful lines */}
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-base font-semibold flex items-center gap-2">
              <Brain className="h-4 w-4 text-primary" />
              Brain Wave Trends
            </h3>
            {isStreaming && (
              <span className="text-[10px] font-mono text-primary animate-pulse">LIVE</span>
            )}
          </div>
          <p className="text-xs text-muted-foreground mb-4">
            % of total EEG power per signal — calm (α), alert (β), creative (θ)
          </p>
          {bandHistory.length < 2 ? (
            <div className="h-[220px] flex items-center justify-center text-sm text-muted-foreground">
              {isStreaming ? "Collecting data…" : "Live trends appear when optional EEG is connected"}
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={bandHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(220,18%,22%)" opacity={0.5} />
                <XAxis dataKey="time" tick={{ fontSize: 9, fill: "hsl(220,12%,52%)" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 9, fill: "hsl(220,12%,52%)" }} axisLine={false} tickLine={false} domain={[0, 60]} />
                <Tooltip
                  contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11 }}
                  labelStyle={{ color: "var(--popover-foreground)" }}
                />
                <Line type="monotone" dataKey="calm" stroke="hsl(152,60%,48%)" strokeWidth={2.5} dot={false} name="Calm (α)" />
                <Line type="monotone" dataKey="alert" stroke="hsl(210,80%,60%)" strokeWidth={2.5} dot={false} name="Alert (β)" />
                <Line type="monotone" dataKey="creative" stroke="hsl(262,60%,65%)" strokeWidth={2.5} dot={false} name="Creative (θ)" />
              </LineChart>
            </ResponsiveContainer>
          )}
          {/* Legend */}
          <div className="flex gap-4 mt-3 justify-center">
            {[
              { color: "hsl(152,60%,48%)", label: "Calm (α)" },
              { color: "hsl(210,80%,60%)", label: "Alert (β)" },
              { color: "hsl(262,60%,65%)", label: "Creative (θ)" },
            ].map(({ color, label }) => (
              <span key={label} className="flex items-center gap-1 text-[10px] text-muted-foreground">
                <span className="w-3 h-0.5 rounded-full inline-block" style={{ background: color }} />
                {label}
              </span>
            ))}
          </div>
        </Card>

        {/* Brain Profile Radar */}
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-base font-semibold flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-secondary" />
              Cognitive Profile
            </h3>
            {isStreaming && (
              <span className="text-[10px] font-mono text-primary animate-pulse">LIVE</span>
            )}
          </div>
          <p className="text-xs text-muted-foreground mb-4">
            Snapshot of all cognitive dimensions — updated every 8 seconds
          </p>
          {radarData.length === 0 ? (
            <div className="h-[220px] flex items-center justify-center text-sm text-muted-foreground">
              Add EEG later to see the live cognitive profile
            </div>
          ) : (
            <>
              <ResponsiveContainer width="100%" height={220}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="hsl(220,18%,25%)" opacity={0.5} />
                  <PolarAngleAxis dataKey="subject" tick={{ fontSize: 10, fill: "hsl(220,12%,60%)" }} />
                  <PolarRadiusAxis tick={{ fontSize: 7, fill: "hsl(220,12%,40%)" }} domain={[0, 100]} />
                  <Radar
                    name="Now"
                    dataKey="value"
                    stroke="hsl(152,60%,48%)"
                    fill="hsl(152,60%,48%)"
                    fillOpacity={0.2}
                    strokeWidth={2}
                  />
                </RadarChart>
              </ResponsiveContainer>
              {/* Score row */}
              <div className="grid grid-cols-3 gap-2 mt-2">
                {radarData.map((d) => (
                  <div key={d.subject} className="text-center">
                    <p className="text-[10px] text-muted-foreground">{d.subject}</p>
                    <p className="text-sm font-mono font-semibold text-primary">{d.value}%</p>
                  </div>
                ))}
              </div>
            </>
          )}
        </Card>
      </div>

      {/* ── Sleep & Dream status ── */}
      {isStreaming && (sleepStaging || dreamDetection) && (
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-base font-semibold mb-4 flex items-center gap-2">
            <Moon className="h-4 w-4 text-secondary" />
            Sleep & Dream Detection
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
            {sleepStaging && (
              <div className="p-4 rounded-xl border border-border/30 bg-card/30">
                <p className="text-xs text-muted-foreground mb-1">Sleep Stage</p>
                <p className="font-semibold capitalize">
                  {String((sleepStaging as Record<string, unknown>).stage ?? "—")}
                </p>
                <p className="text-xs text-muted-foreground mt-2">
                  {String((sleepStaging as Record<string, unknown>).stage) === "wake"
                    ? "Fully awake — normal waking EEG patterns detected."
                    : "Drowsiness or sleep-onset patterns detected. Consider a short rest."}
                </p>
              </div>
            )}
            {dreamDetection && (
              <div className="p-4 rounded-xl border border-border/30 bg-card/30">
                <p className="text-xs text-muted-foreground mb-1">Dream Probability</p>
                <p className="font-semibold">
                  {Math.round(((dreamDetection as Record<string, unknown>).probability as number ?? 0) * 100)}%
                </p>
                <p className="text-xs text-muted-foreground mt-2">
                  {((dreamDetection as Record<string, unknown>).probability as number ?? 0) > 0.4
                    ? "Theta-dominant patterns resemble REM-like activity — hypnagogic state possible."
                    : "No dream-state patterns detected. Brain is in normal waking mode."}
                </p>
              </div>
            )}
          </div>
        </Card>
      )}
    </main>
  );
}
