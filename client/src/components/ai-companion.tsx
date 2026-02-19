import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import {
  Bot,
  Send,
  Brain,
  Wind,
  Heart,
  Leaf,
  Moon,
  Zap,
  Eye,
  Activity,
} from "lucide-react";
import { useDevice } from "@/hooks/use-device";

interface ChatMessage {
  id: string;
  message: string;
  isUser: boolean;
  timestamp: Date;
}

interface AICompanionProps {
  userId: string;
}

/* ── Brain-state-aware insight generator ─────────────────────────── */

function generateInsight(analysis: Record<string, unknown>): string {
  const emotions = analysis.emotions as Record<string, unknown> | undefined;
  const sleep = analysis.sleep_staging as Record<string, unknown> | undefined;
  const flow = analysis.flow_state as Record<string, unknown> | undefined;
  const stress = analysis.stress as Record<string, unknown> | undefined;
  const attention = analysis.attention as Record<string, unknown> | undefined;
  const meditation = analysis.meditation as Record<string, unknown> | undefined;
  const drowsiness = analysis.drowsiness as Record<string, unknown> | undefined;
  const creativity = analysis.creativity as Record<string, unknown> | undefined;

  const parts: string[] = [];

  // Emotion insight
  const emotion = emotions?.emotion as string | undefined;
  const valence = emotions?.valence as number | undefined;
  const valPct = valence != null ? Math.round(valence * 100) : 0;
  if (emotion) {
    if (emotion === "relaxed" || emotion === "calm") {
      parts.push("Your brain is in a calm, relaxed state right now — great for mindfulness or creative thinking.");
    } else if (emotion === "focused" || emotion === "engaged") {
      parts.push("You're showing strong focus patterns. This is a good time for deep work or problem-solving.");
    } else if (emotion === "fearful" || emotion === "anxious" || emotion === "stressed") {
      parts.push("I'm detecting some stress in your brainwave patterns. Consider a breathing exercise or short break.");
    } else if (emotion === "happy" || emotion === "excited") {
      parts.push("Your emotional state looks positive! High valence patterns suggest you're in a good mood.");
    } else if (emotion === "sad") {
      parts.push("Your brain patterns show lower valence — this could mean quiet reflection or low energy. It's not necessarily negative; your brain may be in a restful processing mode.");
    } else if (emotion === "angry") {
      parts.push("High arousal with negative valence detected. Your beta activity is elevated — a brief mindfulness pause could help restore balance.");
    } else {
      parts.push(`Your brain is showing ${valPct >= 0 ? "positive" : "slightly negative"} valence (${valPct}%) with ${emotion} patterns.`);
    }
  }

  // Flow state
  if (flow?.in_flow) {
    parts.push("You're in a flow state — your brain is showing the ideal theta/alpha balance. Keep going!");
  }

  // Stress guidance
  const stressLevel = stress?.level as string | undefined;
  if (stressLevel === "high" || stressLevel === "moderate") {
    parts.push("Your stress markers are elevated. Try the 4-7-8 breathing technique: breathe in for 4s, hold 7s, exhale 8s.");
  }

  // Attention
  const attState = attention?.state as string | undefined;
  if (attState === "distracted") {
    parts.push("Your attention seems scattered. Try focusing on a single task for the next 5 minutes.");
  } else if (attState === "hyperfocused") {
    parts.push("You're in deep focus — be mindful of taking breaks to avoid cognitive fatigue.");
  }

  // Drowsiness
  const drowState = drowsiness?.state as string | undefined;
  if (drowState === "drowsy" || drowState === "sleepy") {
    parts.push("Drowsiness detected — consider a short 20-minute power nap or a walk outside.");
  }

  // Meditation
  const medDepth = meditation?.depth as string | undefined;
  if (medDepth === "deep" || medDepth === "transcendent") {
    parts.push("Impressive meditation depth! Your alpha/theta ratio indicates deep meditative absorption.");
  }

  // Creativity
  const creatState = creativity?.state as string | undefined;
  if (creatState === "high" || creatState === "very_high") {
    parts.push("Your creativity markers are high — alpha desynchronization suggests divergent thinking is active.");
  }

  // Sleep
  const sleepStage = sleep?.stage as string | undefined;
  if (sleepStage === "REM") {
    parts.push("REM detected — dream activity is likely occurring.");
  } else if (sleepStage && sleepStage !== "Wake") {
    parts.push(`Sleep stage: ${sleepStage}. Your brain is transitioning into sleep.`);
  }

  if (parts.length === 0) {
    parts.push("Monitoring your brain activity. Your patterns look normal and stable.");
  }

  return parts.join(" ");
}

function getQuickResponse(action: string, analysis?: Record<string, unknown>): string {
  const stress = (analysis?.stress as Record<string, unknown>)?.stress_index as number | undefined;
  const stressPct = stress != null ? (stress * 100).toFixed(0) : "unknown";

  switch (action) {
    case "breathing":
      return `Let's do a calming breathing exercise. Your current stress level is ${stressPct}%. Try this:\n\n1. Breathe in slowly for 4 seconds\n2. Hold for 7 seconds\n3. Exhale slowly for 8 seconds\n\nRepeat 4 times. I'll monitor your brainwave changes.`;
    case "meditation":
      return "Starting a guided meditation session. Close your eyes and focus on your breath. I'll track your alpha and theta waves to measure meditation depth. Let the thoughts pass like clouds.";
    case "mood":
      return analysis
        ? generateInsight(analysis)
        : "Connect your Muse 2 to get a real-time mood analysis based on your brain activity.";
    case "stress":
      return `Your current stress index is ${stressPct}%. Here are some quick relief techniques:\n\n- Progressive muscle relaxation (tense and release each muscle group)\n- Box breathing (4-4-4-4 pattern)\n- Grounding: name 5 things you see, 4 you hear, 3 you feel\n\nWhich would you like to try?`;
    default:
      return "How can I help you? I can analyze your brain state, guide meditation, or provide wellness insights.";
  }
}

/* ── Component ───────────────────────────────────────────────────── */

export function AICompanion({ userId }: AICompanionProps) {
  const [message, setMessage] = useState("");
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const device = useDevice();
  const { latestFrame, state: deviceState } = device;
  const isStreaming = deviceState === "streaming";
  const analysis = latestFrame?.analysis;

  // Auto-generate brain insight when streaming starts (once)
  const hasAutoInsightRef = useRef(false);

  useEffect(() => {
    if (!isStreaming || !analysis || hasAutoInsightRef.current) return;

    const timer = setTimeout(() => {
      if (chatHistory.length === 0) {
        hasAutoInsightRef.current = true;
        const insight = generateInsight(analysis as Record<string, unknown>);
        setChatHistory([
          {
            id: "auto-1",
            message: `I'm reading your Muse 2 brain activity live. Here's what I see:\n\n${insight}`,
            isUser: false,
            timestamp: new Date(),
          },
        ]);
      }
    }, 5000);

    return () => clearTimeout(timer);
  }, [isStreaming, analysis, chatHistory.length]);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  const addBotMessage = (text: string) => {
    setChatHistory((prev) => [
      ...prev,
      {
        id: `bot-${Date.now()}`,
        message: text,
        isUser: false,
        timestamp: new Date(),
      },
    ]);
  };

  const handleSendMessage = () => {
    if (!message.trim()) return;

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      message: message.trim(),
      isUser: true,
      timestamp: new Date(),
    };
    setChatHistory((prev) => [...prev, userMsg]);

    const lower = message.toLowerCase();
    let response: string;

    if (lower.includes("how") && (lower.includes("feel") || lower.includes("brain") || lower.includes("state"))) {
      response = analysis
        ? generateInsight(analysis as Record<string, unknown>)
        : "Connect your EEG device to get real-time brain state analysis.";
    } else if (lower.includes("stress") || lower.includes("anxious") || lower.includes("relax")) {
      response = getQuickResponse("stress", analysis as Record<string, unknown> | undefined);
    } else if (lower.includes("meditat") || lower.includes("calm") || lower.includes("mindful")) {
      response = getQuickResponse("meditation", analysis as Record<string, unknown> | undefined);
    } else if (lower.includes("breath")) {
      response = getQuickResponse("breathing", analysis as Record<string, unknown> | undefined);
    } else if (lower.includes("sleep") || lower.includes("tired") || lower.includes("drowsy")) {
      const drowsy = (analysis?.drowsiness as Record<string, unknown>)?.state as string | undefined;
      const sleep = (analysis?.sleep_staging as Record<string, unknown>)?.stage as string | undefined;
      response = `Current drowsiness: ${drowsy ?? "unknown"}, sleep stage: ${sleep ?? "unknown"}. ${
        drowsy === "drowsy" || drowsy === "sleepy"
          ? "Your brain shows signs of fatigue. A 20-minute nap or caffeine break could help."
          : "You seem alert right now. Your brain activity suggests good wakefulness."
      }`;
    } else if (lower.includes("focus") || lower.includes("attention") || lower.includes("concentrat")) {
      const att = (analysis?.attention as Record<string, unknown>)?.state as string | undefined;
      const attScore = (analysis?.attention as Record<string, unknown>)?.attention_score as number | undefined;
      response = `Attention state: ${att ?? "unknown"} (${attScore != null ? (attScore * 100).toFixed(0) + "%" : "N/A"}). ${
        att === "focused" || att === "hyperfocused"
          ? "Your brain is in a focused state — keep up the good work!"
          : "Try the Pomodoro technique: 25 minutes of focused work, then 5 minutes break."
      }`;
    } else if (lower.includes("creativ")) {
      const creat = (analysis?.creativity as Record<string, unknown>)?.state as string | undefined;
      response = `Creativity state: ${creat ?? "unknown"}. ${
        creat === "high" || creat === "very_high"
          ? "Your alpha desynchronization patterns suggest high creative potential right now!"
          : "Try free-writing or brainstorming to activate divergent thinking pathways."
      }`;
    } else {
      response = analysis
        ? `Here's your current brain snapshot:\n\n${generateInsight(analysis as Record<string, unknown>)}\n\nAsk me about stress, focus, sleep, creativity, or say "how do I feel?" for a full analysis.`
        : "I can analyze your brain activity when your Muse 2 is streaming. Try asking about your mood, stress, focus, or creativity!";
    }

    setTimeout(() => addBotMessage(response), 500);
    setMessage("");
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleQuickAction = (action: string) => {
    const response = getQuickResponse(action, analysis as Record<string, unknown> | undefined);
    addBotMessage(response);
  };

  // Live emotion metrics from Muse 2
  const emotions = analysis?.emotions;
  const stressVal = emotions?.stress_index ?? 0;
  const focusVal = emotions?.focus_index ?? 0;
  const relaxVal = emotions?.relaxation_index ?? 0;
  const valenceVal = emotions?.valence ?? 0;

  const quickActions = [
    { icon: Wind, label: "Breathing Exercise", color: "success", action: "breathing" },
    { icon: Bot, label: "Guided Meditation", color: "secondary", action: "meditation" },
    { icon: Heart, label: "Mood Check-In", color: "primary", action: "mood" },
    { icon: Leaf, label: "Stress Relief", color: "warning", action: "stress" },
  ];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Chat Interface */}
      <div className="lg:col-span-2 glass-card p-6 rounded-xl hover-glow">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold">AI Brain Companion</h3>
          <div className="flex items-center space-x-2">
            {isStreaming ? (
              <>
                <Activity className="h-3 w-3 text-primary animate-pulse" />
                <span className="text-sm font-mono text-primary">LIVE EEG</span>
              </>
            ) : (
              <>
                <div className="w-2 h-2 rounded-full bg-muted-foreground/40" />
                <span className="text-sm font-mono text-muted-foreground">NO DEVICE</span>
              </>
            )}
          </div>
        </div>

        {/* Chat Messages */}
        <div className="h-96 overflow-y-auto mb-4 space-y-4 bg-card/20 rounded-lg p-4 border border-primary/20">
          {chatHistory.length === 0 ? (
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-primary to-secondary rounded-full flex items-center justify-center flex-shrink-0">
                <Bot className="text-white text-sm" />
              </div>
              <div className="bg-card/50 rounded-lg p-3 max-w-sm">
                <p className="text-sm">
                  {isStreaming
                    ? "Hello! I'm reading your Muse 2 brain activity live. Ask me how you feel, or try a quick action on the right."
                    : "Hello! Connect your Muse 2 from the sidebar to get real-time brain insights. I can analyze your mood, stress, focus, and more."}
                </p>
                <span className="text-xs text-foreground/50">
                  {new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </span>
              </div>
            </div>
          ) : (
            chatHistory.map((chat) => (
              <div
                key={chat.id}
                className={`flex items-start space-x-3 ${chat.isUser ? "justify-end" : ""}`}
              >
                {!chat.isUser && (
                  <div className="w-8 h-8 bg-gradient-to-br from-primary to-secondary rounded-full flex items-center justify-center flex-shrink-0">
                    <Bot className="text-white text-sm" />
                  </div>
                )}
                <div
                  className={`rounded-lg p-3 max-w-sm ${
                    chat.isUser ? "bg-primary/20" : "bg-card/50"
                  }`}
                >
                  <p className="text-sm whitespace-pre-line">{chat.message}</p>
                  <span className="text-xs text-foreground/50">
                    {new Date(chat.timestamp).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </span>
                </div>
                {chat.isUser && (
                  <div className="w-8 h-8 bg-gradient-to-br from-success to-primary rounded-full flex items-center justify-center flex-shrink-0">
                    <Brain className="h-4 w-4 text-white" />
                  </div>
                )}
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Chat Input */}
        <div className="flex items-center space-x-3">
          <Input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder={
              isStreaming
                ? "Ask about your brain state, mood, focus, stress..."
                : "Connect your Muse 2 for live insights..."
            }
            className="flex-1 bg-card/50 border border-primary/30 rounded-lg !text-foreground [caret-color:var(--foreground)]"
          />
          <Button
            onClick={handleSendMessage}
            disabled={!message.trim()}
            className="bg-gradient-to-r from-primary to-secondary text-primary-foreground hover-glow"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Quick Actions & Live Metrics */}
      <div className="space-y-6">
        {/* Quick Actions */}
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-lg font-semibold mb-4">Quick Actions</h3>
          <div className="space-y-3">
            {quickActions.map((action, index) => {
              const Icon = action.icon;
              const colorClasses: Record<string, string> = {
                success: "bg-success/10 border-success/30 text-success hover:bg-success/20",
                secondary: "bg-secondary/10 border-secondary/30 text-secondary hover:bg-secondary/20",
                primary: "bg-primary/10 border-primary/30 text-primary hover:bg-primary/20",
                warning: "bg-warning/10 border-warning/30 text-warning hover:bg-warning/20",
              };
              return (
                <Button
                  key={index}
                  variant="outline"
                  className={`w-full py-3 px-4 rounded-lg transition-all flex items-center justify-between ${colorClasses[action.color]}`}
                  onClick={() => handleQuickAction(action.action)}
                >
                  <span>{action.label}</span>
                  <Icon className="h-4 w-4" />
                </Button>
              );
            })}
          </div>
        </Card>

        {/* Live Brain Metrics from Muse 2 */}
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Live Brain Metrics</h3>
            {isStreaming && (
              <span className="text-[10px] font-mono text-primary animate-pulse">LIVE</span>
            )}
          </div>
          <div className="space-y-4">
            <MetricBar label="Stress" value={stressVal * 100} color="warning" icon={Zap} />
            <MetricBar label="Focus" value={focusVal * 100} color="primary" icon={Eye} />
            <MetricBar label="Relaxation" value={relaxVal * 100} color="success" icon={Leaf} />
            <MetricBar label="Mood" value={valenceVal * 100} color="secondary" icon={Heart} />

            {/* Current states */}
            {isStreaming && analysis && (
              <div className="pt-3 mt-3 border-t border-border/30 space-y-2">
                {analysis.sleep_staging?.stage && (
                  <StateChip label="Sleep" value={analysis.sleep_staging.stage} icon={Moon} />
                )}
                {analysis.attention?.state && (
                  <StateChip label="Attention" value={analysis.attention.state} icon={Eye} />
                )}
                {analysis.flow_state?.in_flow && (
                  <StateChip label="Flow" value="In Flow" icon={Zap} />
                )}
                {analysis.meditation?.depth && (
                  <StateChip label="Meditation" value={analysis.meditation.depth} icon={Heart} />
                )}
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}

/* ── Sub-components ──────────────────────────────────────────────── */

function MetricBar({
  label,
  value,
  color,
  icon: Icon,
}: {
  label: string;
  value: number;
  color: string;
  icon: React.ComponentType<{ className?: string }>;
}) {
  const colorMap: Record<string, { bar: string; text: string }> = {
    warning: { bar: "bg-warning", text: "text-warning" },
    success: { bar: "bg-success", text: "text-success" },
    primary: { bar: "bg-primary", text: "text-primary" },
    secondary: { bar: "bg-secondary", text: "text-secondary" },
  };
  const c = colorMap[color] ?? colorMap.primary;

  return (
    <div className="flex items-center gap-3">
      <Icon className={`h-3.5 w-3.5 ${c.text} shrink-0`} />
      <span className="text-xs w-16 text-muted-foreground">{label}</span>
      <div className="flex-1 h-2 bg-card rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${c.bar}`}
          style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
        />
      </div>
      <span className={`text-xs font-mono w-8 text-right ${c.text}`}>
        {value.toFixed(0)}%
      </span>
    </div>
  );
}

function StateChip({
  label,
  value,
  icon: Icon,
}: {
  label: string;
  value: string;
  icon: React.ComponentType<{ className?: string }>;
}) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <Icon className="h-3 w-3 text-muted-foreground" />
      <span className="text-muted-foreground">{label}:</span>
      <span className="text-foreground font-medium capitalize">{value}</span>
    </div>
  );
}
