/**
 * InterventionTriggerToast — non-blocking toast/card for EEG-triggered suggestions.
 *
 * Shows when the intervention trigger engine detects a condition requiring
 * user attention. Includes "Dismiss" and "Try it" buttons. Logs events.
 *
 * @see Issue #504
 */

import { useCallback } from "react";
import { useLocation } from "wouter";
import {
  Wind,
  Music,
  Coffee,
  X,
  ChevronRight,
} from "lucide-react";
import type { InterventionTrigger } from "@/lib/eeg-intervention-trigger";

// ── Icon + style mapping ─────────────────────────────────────────────────────

const TYPE_CONFIG: Record<
  InterventionTrigger["type"],
  { icon: React.ElementType; route: string; label: string; color: string; border: string }
> = {
  breathing: {
    icon: Wind,
    route: "/biofeedback",
    label: "Try breathing exercise",
    color: "text-cyan-400 bg-cyan-500/15",
    border: "border-cyan-500/40",
  },
  music_change: {
    icon: Music,
    route: "/biofeedback?tab=music&mood=calm",
    label: "Switch to calm music",
    color: "text-indigo-400 bg-indigo-500/15",
    border: "border-indigo-500/40",
  },
  notification: {
    icon: Coffee,
    route: "/biofeedback",
    label: "View suggestion",
    color: "text-amber-400 bg-amber-500/15",
    border: "border-amber-500/40",
  },
  break_suggestion: {
    icon: Coffee,
    route: "/biofeedback",
    label: "Take a break",
    color: "text-amber-400 bg-amber-500/15",
    border: "border-amber-500/40",
  },
};

// ── Props ────────────────────────────────────────────────────────────────────

export interface InterventionTriggerToastProps {
  trigger: InterventionTrigger;
  onDismiss: () => void;
  onAction?: () => void;
}

// ── Component ────────────────────────────────────────────────────────────────

export function InterventionTriggerToast({
  trigger,
  onDismiss,
  onAction,
}: InterventionTriggerToastProps) {
  const [, navigate] = useLocation();
  const cfg = TYPE_CONFIG[trigger.type] ?? TYPE_CONFIG.break_suggestion;
  const Icon = cfg.icon;

  const handleAction = useCallback(() => {
    trigger.action();
    onAction?.();
    navigate(cfg.route);
    onDismiss();
  }, [trigger, onAction, navigate, cfg.route, onDismiss]);

  return (
    <div
      className={
        `fixed bottom-20 right-4 z-50 w-72 rounded-xl border ${cfg.border} ` +
        "bg-background/90 shadow-xl backdrop-blur-sm " +
        "animate-in slide-in-from-bottom-4 duration-300"
      }
      role="alert"
      aria-live="polite"
      data-testid="intervention-trigger-toast"
    >
      {/* Header */}
      <div className="flex items-start gap-3 px-4 pt-3 pb-1.5">
        <div className={`mt-0.5 shrink-0 rounded-md p-1.5 ${cfg.color}`}>
          <Icon className="h-4 w-4" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold text-foreground leading-snug">
            {trigger.type === "breathing" && "Breathing suggested"}
            {trigger.type === "music_change" && "Try calming music"}
            {trigger.type === "break_suggestion" && "Break suggested"}
            {trigger.type === "notification" && "Suggestion"}
          </p>
          <p className="mt-0.5 text-[10px] text-muted-foreground leading-relaxed">
            {trigger.reason}
          </p>
        </div>
        <button
          onClick={onDismiss}
          className="shrink-0 rounded p-1 hover:bg-muted/50 transition-colors"
          aria-label="Dismiss"
        >
          <X className="h-3 w-3 text-muted-foreground" />
        </button>
      </div>

      {/* Actions */}
      <div className="px-4 pb-3 pt-1 flex gap-2">
        <button
          onClick={handleAction}
          className={
            `flex-1 flex items-center justify-center gap-1 rounded-lg px-3 py-1.5 ` +
            `text-[11px] font-medium transition-colors ${cfg.color} hover:opacity-80`
          }
        >
          {cfg.label}
          <ChevronRight className="h-3 w-3" />
        </button>
        <button
          onClick={onDismiss}
          className="rounded-lg px-3 py-1.5 text-[11px] text-muted-foreground hover:bg-muted/50 transition-colors"
        >
          Dismiss
        </button>
      </div>
    </div>
  );
}
