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
} from "lucide-react";
import { useDevice } from "@/hooks/use-device";
import { OpenAIService, type ChatResponse } from "@/lib/openai";

interface AICompanionProps {
  userId: string;
}

export function AICompanion({ userId }: AICompanionProps) {
  const [message, setMessage] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const queryClient = useQueryClient();

  const device = useDevice();
  const { latestFrame, state: deviceState } = device;
  const isStreaming = deviceState === "streaming";
  const analysis = latestFrame?.analysis;

  // Load chat history from DB
  const { data: chatHistory = [] } = useQuery<ChatResponse[]>({
    queryKey: ["ai-chat", userId],
    queryFn: () => OpenAIService.getChatHistory(userId),
    staleTime: 30_000,
  });

  // Auto-scroll on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, isTyping]);

  const sendMessage = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || isTyping) return;

    // Optimistically add user message to UI
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
      // Build EEG context to pass with each message
      const eegContext = buildEegContext();
      const fullMessage = eegContext ? `${trimmed}\n\n[EEG context: ${eegContext}]` : trimmed;

      // Get current history for conversation memory
      const currentHistory = queryClient.getQueryData<ChatResponse[]>(["ai-chat", userId]) ?? [];

      const aiResponse = await OpenAIService.sendChatMessage(fullMessage, userId, currentHistory);

      queryClient.setQueryData<ChatResponse[]>(["ai-chat", userId], (prev = []) => {
        // Replace optimistic user msg with real one from server, then add AI reply
        const withoutOptimistic = prev.filter((m) => m.id !== optimisticUser.id);
        return [
          ...withoutOptimistic,
          { ...optimisticUser, id: `user-${Date.now()}` },
          aiResponse,
        ];
      });
    } catch {
      // Remove optimistic message on error, restore input
      queryClient.setQueryData<ChatResponse[]>(["ai-chat", userId], (prev = []) =>
        prev.filter((m) => m.id !== optimisticUser.id)
      );
      setMessage(trimmed);
    } finally {
      setIsTyping(false);
    }
  };

  const buildEegContext = (): string => {
    if (!isStreaming || !analysis) return "";
    const parts: string[] = [];
    const emotions = analysis.emotions as Record<string, unknown> | undefined;
    if (emotions?.emotion) parts.push(`emotion: ${emotions.emotion}`);
    if (emotions?.stress_index != null) parts.push(`stress: ${Math.round((emotions.stress_index as number) * 100)}%`);
    if (emotions?.focus_index != null) parts.push(`focus: ${Math.round((emotions.focus_index as number) * 100)}%`);
    const sleep = (analysis.sleep_staging as Record<string, unknown> | undefined)?.stage;
    if (sleep) parts.push(`sleep stage: ${sleep}`);
    return parts.join(", ");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(message);
    }
  };

  const quickActions = [
    { icon: Wind, label: "Breathing Exercise", prompt: "Guide me through a calming breathing exercise right now." },
    { icon: Bot, label: "Guided Meditation", prompt: "Start a short guided meditation session for me." },
    { icon: Heart, label: "Mood Check-In", prompt: "How am I doing emotionally? Give me a mood check-in." },
    { icon: Leaf, label: "Stress Relief", prompt: "What can I do right now to relieve stress?" },
  ];

  // Live metrics
  const emotions = analysis?.emotions as Record<string, unknown> | undefined;
  const stressVal = ((emotions?.stress_index as number) ?? 0) * 100;
  const focusVal = ((emotions?.focus_index as number) ?? 0) * 100;
  const relaxVal = ((emotions?.relaxation_index as number) ?? 0) * 100;
  const valenceVal = ((emotions?.valence as number) ?? 0) * 100;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Chat Interface */}
      <div className="lg:col-span-2 glass-card rounded-xl hover-glow flex flex-col" style={{ height: "calc(100vh - 10rem)" }}>
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border/30">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-gradient-to-br from-primary to-secondary rounded-full flex items-center justify-center">
              <Bot className="h-5 w-5 text-white" />
            </div>
            <div>
              <h3 className="text-sm font-semibold">AI Brain Companion</h3>
              <p className="text-xs text-muted-foreground">Powered by GPT-5</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
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
                    ? "Hello! I'm reading your Muse 2 brain activity live. Ask me anything — how you feel, stress tips, meditation, or just have a conversation."
                    : "Hello! I'm your AI wellness companion. Connect your Muse 2 for real-time brain insights, or just ask me anything about wellness, focus, sleep, or stress."}
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
                <span className={`text-[10px] mt-1 block ${chat.isUser ? "text-primary-foreground/60 text-right" : "text-muted-foreground"}`}>
                  {new Date(chat.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </span>
              </div>
              {chat.isUser && (
                <div className="w-8 h-8 bg-gradient-to-br from-success to-primary rounded-full flex items-center justify-center shrink-0 mb-1">
                  <Brain className="h-4 w-4 text-white" />
                </div>
              )}
            </div>
          ))}

          {/* Typing indicator */}
          {isTyping && (
            <div className="flex items-end gap-2">
              <div className="w-8 h-8 bg-gradient-to-br from-primary to-secondary rounded-full flex items-center justify-center shrink-0">
                <Bot className="h-4 w-4 text-white" />
              </div>
              <div className="bg-card/60 rounded-2xl rounded-bl-sm px-4 py-3">
                <div className="flex gap-1 items-center h-4">
                  <span className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                  <span className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                  <span className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
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
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Message AI companion... (Enter to send, Shift+Enter for new line)"
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
            {isStreaming ? "EEG context is automatically included with each message." : "Connect your Muse 2 for brain-aware responses."}
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
                  className="w-full flex items-center gap-3 px-4 py-3 rounded-xl border border-border/30 bg-card/30 hover:bg-card/60 transition-colors text-sm text-left disabled:opacity-50"
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
            {isStreaming && <span className="text-[10px] font-mono text-primary animate-pulse">LIVE</span>}
          </div>
          <div className="space-y-4">
            <MetricBar label="Stress" value={stressVal} colorClass="bg-rose-500" textClass="text-rose-400" icon={Zap} />
            <MetricBar label="Focus" value={focusVal} colorClass="bg-primary" textClass="text-primary" icon={Eye} />
            <MetricBar label="Relaxation" value={relaxVal} colorClass="bg-emerald-500" textClass="text-emerald-400" icon={Leaf} />
            <MetricBar label="Mood" value={valenceVal} colorClass="bg-secondary" textClass="text-secondary" icon={Heart} />
          </div>

          {isStreaming && analysis && (
            <div className="pt-3 mt-3 border-t border-border/30 space-y-2">
              {(analysis.sleep_staging as Record<string, unknown> | undefined)?.stage && (
                <StateChip label="Sleep" value={String((analysis.sleep_staging as Record<string, unknown>).stage)} icon={Moon} />
              )}
              {(analysis.attention as Record<string, unknown> | undefined)?.state && (
                <StateChip label="Attention" value={String((analysis.attention as Record<string, unknown>).state)} icon={Eye} />
              )}
              {(analysis.flow_state as Record<string, unknown> | undefined)?.in_flow && (
                <StateChip label="Flow" value="In Flow" icon={Zap} />
              )}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

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
      <span className="text-xs w-16 text-muted-foreground">{label}</span>
      <div className="flex-1 h-1.5 bg-card rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${colorClass}`}
          style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
        />
      </div>
      <span className={`text-xs font-mono w-8 text-right ${textClass}`}>{value.toFixed(0)}%</span>
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
