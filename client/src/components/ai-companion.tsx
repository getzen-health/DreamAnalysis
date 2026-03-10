import React, { useState, useEffect, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
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
  WifiOff,
} from "lucide-react";
import { useDevice } from "@/hooks/use-device";
import { OpenAIService, type ChatResponse } from "@/lib/openai";

interface AICompanionProps {
  userId: string;
}

// ---------------------------------------------------------------------------
// Local response engine — works without any backend API key
// ---------------------------------------------------------------------------
function generateLocalResponse(message: string, eegContext: string): string {
  const msg = message.toLowerCase();

  if (msg.includes("breath") || msg.includes("breathing")) {
    return `Let's do a **4-7-8 Breathing Exercise** together 🌬️

1. **Inhale** slowly for **4 counts** — fill your lungs completely
2. **Hold** your breath for **7 counts**
3. **Exhale** fully for **8 counts** — release every bit of air

*Start now... inhale 4…3…2…1… hold 7…6…5…4…3…2…1… exhale 8…7…6…5…4…3…2…1…*

Repeat this cycle 4 times. This technique activates your parasympathetic nervous system and lowers cortisol within minutes.

How do you feel after the first cycle? 🍃`;
  }

  if (msg.includes("medit") || msg.includes("mindful")) {
    return `Let's begin a **3-Minute Body Scan Meditation** 🧘

Find a comfortable position. Close your eyes if that feels right.

**Settle into your breath** — notice the natural rhythm. Don't change it, just observe.

**Now scan your body:**
- 👣 Feet on the ground — notice any tension and let it soften
- 🦵 Legs and hips — release tightness on each exhale
- 🫁 Stomach — let it expand freely with every breath
- 🫀 Chest and shoulders — drop them away from your ears
- 😌 Jaw, eyes, forehead — let every muscle soften

**Rest here for a moment.** Your mind may wander — that's completely natural. Just gently return attention to your breath.

When you're ready, take a deep breath and slowly open your eyes. 🌿

How was that for you?`;
  }

  if (
    msg.includes("stress") ||
    msg.includes("anxious") ||
    msg.includes("anxiety") ||
    msg.includes("overwhelm") ||
    msg.includes("stress relief")
  ) {
    const eegNote = eegContext
      ? ` Your EEG currently shows elevated beta activity — consistent with what you're feeling.`
      : "";
    return `I hear you.${eegNote} Here are some immediate techniques that work:

**5-4-3-2-1 Grounding** (do this right now):
- 👁️ 5 things you can **see**
- ✋ 4 things you can **feel** (textures, temperature)
- 👂 3 things you can **hear**
- 👃 2 things you can **smell**
- 👅 1 thing you can **taste**

This activates your sensory cortex and interrupts the stress loop within 60 seconds.

**Physiological sigh** — the fastest way to calm down:
- Double inhale through the nose (short sharp breath, then top it off)
- Long slow exhale through the mouth

**Physical resets:**
- Cold water on your wrists or face — triggers the dive reflex, slows heart rate instantly
- 5 minutes of walking — reduces cortisol by up to 50%

What's driving your stress right now? Talking it through often helps too. 🤝`;
  }

  if (
    msg.includes("mood") ||
    msg.includes("how am i") ||
    msg.includes("check-in") ||
    msg.includes("feeling") ||
    msg.includes("emotion")
  ) {
    if (eegContext) {
      return `Based on your live brain data, here's your check-in 🧠

**Current EEG snapshot:** ${eegContext}

**What this means:**
- 🟢 **Alpha (calm)** = relaxed awareness — great for reflection and learning
- 🔵 **Beta (alert)** = active thinking — good for problem-solving and focus
- 🟣 **Theta (creative)** = meditative, imaginative — the creativity gateway

Does this match how you're actually feeling? Sometimes the brain signals things our conscious mind hasn't caught up to yet.

Would you like to explore any of these states further, or try a quick exercise to shift your mental state?`;
    }
    return `Let's do a quick mood check-in 🌱

Rate yourself 1–10 on these right now:
- ⚡ **Energy** (1 = exhausted, 10 = fully alive)
- 😤 **Stress** (1 = totally calm, 10 = overwhelmed)
- 🎯 **Focus** (1 = scattered, 10 = laser-sharp)
- 😊 **Mood** (1 = very low, 10 = excellent)

Start with your own words or a voice check-in for guidance now. Add Muse 2 later if you want live brain-wave insights. 💙`;
  }

  if (msg.includes("sleep") || msg.includes("tired") || msg.includes("insomnia") || msg.includes("rest")) {
    return `Sleep is your brain's most powerful recovery tool. Here's what science recommends 😴

**Tonight:**
- 📵 Stop screens 30–60 min before bed — blue light suppresses melatonin by up to 50%
- 🌡️ Keep your room at 65–68°F (18–20°C) — your core temp must drop to initiate sleep
- 📝 Write down tomorrow's tasks — offloads your prefrontal cortex so it can rest

**Military sleep method:**
Relax your face → drop your shoulders → breathe out → relax your legs → clear your mind for 10 seconds (think of a calm lake). Most people fall asleep in under 2 minutes.

**EEG insight:** During deep N3 sleep, your brain's glymphatic system clears toxic metabolic waste — including the proteins linked to cognitive decline. Missing this is like never taking out the trash.

Are you having trouble falling asleep, staying asleep, or waking too early? 🌙`;
  }

  if (
    msg.includes("focus") ||
    msg.includes("concentrat") ||
    msg.includes("productive") ||
    msg.includes("distract")
  ) {
    return `Here are research-backed focus techniques 🎯

**Pomodoro protocol:**
- 25 min focused work → 5 min break
- After 4 cycles → 20–30 min longer break
- During breaks: look at something 20+ feet away (resets ciliary muscles)

**Your brain's natural rhythms:**
- Ultradian cycles: ~90 min peak focus, ~20 min rest
- Peak cognitive window: 2–4 hours after waking
- Work WITH these cycles, not against them

**Instant focus hacks:**
- 🎵 Brown noise or lo-fi at moderate volume (boosts abstract thinking)
- 💧 Drink water — even mild dehydration reduces cognitive performance by 10%
- 🌊 Cold water on your face before a focus session (activates the trigeminal nerve)

**EEG note:** Low beta waves (12–20 Hz) are the signature of focused calm. Anxious focus looks like high beta (20–30 Hz) — similar output, much higher cost.

What are you trying to focus on? I can give more tailored advice. 🧠`;
  }

  // Generic fallback
  return `Hello! I'm your AI wellness companion 🌿

${
  eegContext
    ? `Your brain is currently showing: ${eegContext}\n\n`
    : ""
}I can help you with:
- 🌬️ **Breathing exercises** — immediate calm
- 🧘 **Guided meditation** — deepen presence
- 💤 **Sleep optimization** — better recovery
- 🎯 **Focus techniques** — sharper thinking
- 💆 **Stress relief** — proven methods
- 😊 **Mood check-ins** — understand how you're feeling

Just ask me anything or use the quick action buttons. What would you like to explore today?`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function AICompanion({ userId }: AICompanionProps) {
  const [message, setMessage] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isOffline, setIsOffline] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();

  const device = useDevice();
  const { latestFrame, state: deviceState } = device;
  const isStreaming = deviceState === "streaming";
  const analysis = latestFrame?.analysis;

  // Load chat history — swallow errors so the component always mounts cleanly
  const { data: chatHistory = [] } = useQuery<ChatResponse[]>({
    queryKey: ["ai-chat", userId],
    queryFn: async () => {
      try {
        return await OpenAIService.getChatHistory(userId);
      } catch {
        return [];
      }
    },
    staleTime: 30_000,
  });

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, isTyping]);

  const buildEegContext = (): string => {
    if (!isStreaming || !analysis) return "";
    const parts: string[] = [];
    const emotions = analysis.emotions as Record<string, unknown> | undefined;
    if (emotions?.emotion) parts.push(`emotion: ${emotions.emotion}`);
    if (emotions?.stress_index != null)
      parts.push(`stress: ${Math.round((emotions.stress_index as number) * 100)}%`);
    if (emotions?.focus_index != null)
      parts.push(`focus: ${Math.round((emotions.focus_index as number) * 100)}%`);
    const sleep = (analysis.sleep_staging as Record<string, unknown> | undefined)?.stage;
    if (sleep) parts.push(`sleep stage: ${sleep}`);
    return parts.join(", ");
  };

  const sendMessage = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || isTyping) return;

    const optimisticUser: ChatResponse = {
      id: `optimistic-${Date.now()}`,
      message: trimmed,
      isUser: true,
      timestamp: new Date(),
    };

    queryClient.setQueryData<ChatResponse[]>(["ai-chat", userId], (prev = []) => [
      ...prev,
      optimisticUser,
    ]);
    setMessage("");
    setIsTyping(true);

    try {
      const eegContext = buildEegContext();
      const fullMessage = eegContext ? `${trimmed}\n\n[EEG context: ${eegContext}]` : trimmed;
      const currentHistory =
        queryClient.getQueryData<ChatResponse[]>(["ai-chat", userId]) ?? [];

      const aiResponse = await OpenAIService.sendChatMessage(fullMessage, userId, currentHistory);

      queryClient.setQueryData<ChatResponse[]>(["ai-chat", userId], (prev = []) => {
        const withoutOptimistic = prev.filter((m) => m.id !== optimisticUser.id);
        return [
          ...withoutOptimistic,
          { ...optimisticUser, id: `user-${Date.now()}` },
          aiResponse,
        ];
      });
      setIsOffline(false);
    } catch {
      // API unavailable — generate a local response so the user always gets feedback
      const eegContext = buildEegContext();
      const localReply: ChatResponse = {
        id: `local-${Date.now()}`,
        message: generateLocalResponse(trimmed, eegContext),
        isUser: false,
        timestamp: new Date(),
      };
      queryClient.setQueryData<ChatResponse[]>(["ai-chat", userId], (prev = []) => {
        const withoutOptimistic = prev.filter((m) => m.id !== optimisticUser.id);
        return [
          ...withoutOptimistic,
          { ...optimisticUser, id: `user-${Date.now()}` },
          localReply,
        ];
      });
      setIsOffline(true);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(message);
    }
  };

  const quickActions = [
    {
      icon: Wind,
      label: "Breathing Exercise",
      prompt: "Guide me through a calming breathing exercise right now.",
    },
    {
      icon: Bot,
      label: "Guided Meditation",
      prompt: "Start a short guided meditation session for me.",
    },
    {
      icon: Heart,
      label: "Mood Check-In",
      prompt: "How am I doing emotionally? Give me a mood check-in.",
    },
    {
      icon: Leaf,
      label: "Stress Relief",
      prompt: "What can I do right now to relieve stress?",
    },
  ];

  const emotions = analysis?.emotions as Record<string, unknown> | undefined;
  const stressVal = ((emotions?.stress_index as number) ?? 0) * 100;
  const focusVal = ((emotions?.focus_index as number) ?? 0) * 100;
  const relaxVal = ((emotions?.relaxation_index as number) ?? 0) * 100;
  const valenceVal = ((emotions?.valence as number) ?? 0) * 100;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Chat Interface */}
      <div
        className="lg:col-span-2 glass-card rounded-xl hover-glow flex flex-col"
        style={{ height: "calc(100vh - 10rem)" }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border/30">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-gradient-to-br from-primary to-secondary rounded-full flex items-center justify-center">
              <Bot className="h-5 w-5 text-white" />
            </div>
            <div>
              <h3 className="text-sm font-semibold">AI Brain Companion</h3>
              <p className="text-xs text-muted-foreground">
                {isOffline ? "Offline mode — local responses" : "Powered by GPT-5"}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {isOffline && (
              <span className="flex items-center gap-1 text-xs text-amber-500 font-mono">
                <WifiOff className="h-3 w-3" />
                OFFLINE
              </span>
            )}
            {isStreaming ? (
              <>
                <Activity className="h-3 w-3 text-primary animate-pulse" />
                <span className="text-xs font-mono text-primary">LIVE EEG</span>
              </>
            ) : (
              <span className="text-xs font-mono text-muted-foreground">NO DEVICE</span>
            )}
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {chatHistory.length === 0 && !isTyping && (
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-gradient-to-br from-primary to-secondary rounded-full flex items-center justify-center shrink-0">
                <Bot className="h-4 w-4 text-white" />
              </div>
              <div className="bg-card/50 rounded-2xl rounded-tl-sm px-4 py-3 max-w-lg">
                <p className="text-sm">
                  {isStreaming
                    ? "Hello! I'm reading your Muse 2 brain activity live. Ask me anything — breathing exercises, meditation, stress tips, or just have a conversation."
                    : "Hello! I'm your AI wellness companion. Use the quick action buttons or type anything — I'm always here to help with focus, stress, sleep, or wellness."}
                </p>
                <span className="text-[10px] text-muted-foreground mt-1 block">
                  {new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </span>
              </div>
            </div>
          )}

          {chatHistory.map((chat) => (
            <div
              key={chat.id}
              className={`flex items-end gap-2 ${chat.isUser ? "justify-end" : "justify-start"}`}
            >
              {!chat.isUser && (
                <div className="w-8 h-8 bg-gradient-to-br from-primary to-secondary rounded-full flex items-center justify-center shrink-0 mb-1">
                  <Bot className="h-4 w-4 text-white" />
                </div>
              )}
              <div
                className={`rounded-2xl px-4 py-3 max-w-lg text-sm whitespace-pre-wrap ${
                  chat.isUser
                    ? "bg-primary text-primary-foreground rounded-br-sm"
                    : "bg-card/60 rounded-bl-sm"
                }`}
              >
                {chat.message}
                <span
                  className={`text-[10px] mt-1 block ${
                    chat.isUser
                      ? "text-primary-foreground/60 text-right"
                      : "text-muted-foreground"
                  }`}
                >
                  {new Date(chat.timestamp).toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </span>
              </div>
              {chat.isUser && (
                <div className="w-8 h-8 bg-gradient-to-br from-success to-primary rounded-full flex items-center justify-center shrink-0 mb-1">
                  <Brain className="h-4 w-4 text-white" />
                </div>
              )}
            </div>
          ))}

          {isTyping && (
            <div className="flex items-end gap-2">
              <div className="w-8 h-8 bg-gradient-to-br from-primary to-secondary rounded-full flex items-center justify-center shrink-0">
                <Bot className="h-4 w-4 text-white" />
              </div>
              <div className="bg-card/60 rounded-2xl rounded-bl-sm px-4 py-3">
                <div className="flex gap-1 items-center h-4">
                  <span
                    className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"
                    style={{ animationDelay: "0ms" }}
                  />
                  <span
                    className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"
                    style={{ animationDelay: "150ms" }}
                  />
                  <span
                    className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"
                    style={{ animationDelay: "300ms" }}
                  />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="px-6 py-4 border-t border-border/30">
          <div className="flex items-end gap-3">
            <Textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Message AI companion… (Enter to send, Shift+Enter for new line)"
              className="flex-1 min-h-[44px] max-h-32 resize-none border border-primary/30 rounded-xl text-sm"
              rows={1}
              disabled={isTyping}
            />
            <Button
              onClick={() => sendMessage(message)}
              disabled={!message.trim() || isTyping}
              size="icon"
              className="h-11 w-11 rounded-xl bg-gradient-to-r from-primary to-secondary text-primary-foreground shrink-0"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
          <p className="text-[10px] text-muted-foreground mt-2">
            {isOffline
              ? "Running in offline mode — responses are locally generated."
              : isStreaming
              ? "EEG context is automatically included with each message."
              : "Voice and health context can guide responses now. EEG is optional later for live brain-aware input."}
          </p>
        </div>
      </div>

      {/* Sidebar */}
      <div className="space-y-6">
        {/* Quick Actions */}
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-sm font-semibold mb-4">Quick Actions</h3>
          <div className="space-y-2">
            {quickActions.map((action, i) => {
              const Icon = action.icon;
              return (
                <button
                  key={i}
                  onClick={() => sendMessage(action.prompt)}
                  disabled={isTyping}
                  className="w-full flex items-center gap-3 px-4 py-3 rounded-xl border border-border/30 bg-card/30 hover:bg-card/60 transition-colors text-sm text-left disabled:opacity-50 cursor-pointer"
                >
                  <Icon className="h-4 w-4 text-primary shrink-0" />
                  <span>{action.label}</span>
                </button>
              );
            })}
          </div>
        </Card>

        {/* Live Brain Metrics */}
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold">Live Brain Metrics</h3>
            {isStreaming && (
              <span className="text-[10px] font-mono text-primary animate-pulse">LIVE</span>
            )}
          </div>
          <div className="space-y-4">
            <MetricBar
              label="Stress"
              value={stressVal}
              colorClass="bg-rose-500"
              textClass="text-rose-400"
              icon={Zap}
            />
            <MetricBar
              label="Focus"
              value={focusVal}
              colorClass="bg-primary"
              textClass="text-primary"
              icon={Eye}
            />
            <MetricBar
              label="Relax"
              value={relaxVal}
              colorClass="bg-emerald-500"
              textClass="text-emerald-400"
              icon={Leaf}
            />
            <MetricBar
              label="Mood"
              value={valenceVal}
              colorClass="bg-secondary"
              textClass="text-secondary"
              icon={Heart}
            />
          </div>

          {!isStreaming && (
            <p className="text-[10px] text-muted-foreground mt-3 text-center">
              Live EEG metrics appear here if you add optional Muse later
            </p>
          )}

          {isStreaming && analysis && (
            <div className="pt-3 mt-3 border-t border-border/30 space-y-2">
              {!!(analysis.sleep_staging as Record<string, unknown> | undefined)?.stage && (
                <StateChip
                  label="Sleep"
                  value={String(
                    (analysis.sleep_staging as Record<string, unknown>).stage
                  )}
                  icon={Moon}
                />
              )}
              {!!(analysis.attention as Record<string, unknown> | undefined)?.state && (
                <StateChip
                  label="Attention"
                  value={String((analysis.attention as Record<string, unknown>).state)}
                  icon={Eye}
                />
              )}
              {!!(analysis.flow_state as Record<string, unknown> | undefined)?.in_flow && (
                <StateChip label="Flow" value="In Flow" icon={Zap} />
              )}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------
function MetricBar({
  label,
  value,
  colorClass,
  textClass,
  icon: Icon,
}: {
  label: string;
  value: number;
  colorClass: string;
  textClass: string;
  icon: React.ComponentType<{ className?: string }>;
}) {
  return (
    <div className="flex items-center gap-3">
      <Icon className={`h-3.5 w-3.5 ${textClass} shrink-0`} />
      <span className="text-xs w-10 text-muted-foreground">{label}</span>
      <div className="flex-1 h-1.5 bg-card rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${colorClass}`}
          style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
        />
      </div>
      <span className={`text-xs font-mono w-8 text-right ${textClass}`}>
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
