/**
 * InterventionSuggestion — "What can I do?" action card for mood/emotion data.
 *
 * Every page that shows mood or emotion data must pair it with an actionable
 * intervention. This component maps the current emotional state to a concrete
 * suggestion with a navigation link.
 *
 * Mapping:
 *   Stressed/Anxious -> "Try a 2-minute breathing exercise" -> /biofeedback
 *   Sad/Low          -> "Talk to AI companion" -> /ai-companion
 *   Angry            -> "Try cognitive reappraisal" (inline text)
 *   Neutral/Balanced -> "Try a focus session" -> /neurofeedback
 *   Happy/Excited    -> "Capture this moment in your dream journal" -> /dreams
 *
 * @see Issue #524
 */

import { useLocation } from "wouter";
import { Wind, MessageCircle, Brain, BookOpen, Lightbulb, type LucideIcon } from "lucide-react";

export interface InterventionSuggestionProps {
  /** Primary emotion label: "happy" | "sad" | "angry" | "fear" | "neutral" | etc. */
  emotion?: string;
  /** Stress index 0-1 */
  stressIndex?: number;
  /** Valence -1 to 1 */
  valence?: number;
  /** Compact mode — smaller card, less padding */
  compact?: boolean;
}

interface Suggestion {
  icon: LucideIcon;
  iconColor: string;
  title: string;
  body: string;
  route: string | null; // null = inline text exercise, no navigation
  actionLabel: string;
}

function getSuggestion(
  emotion?: string,
  stressIndex?: number,
  valence?: number,
): Suggestion {
  const stress = stressIndex ?? 0;
  const val = valence ?? 0;
  const emo = (emotion ?? "neutral").toLowerCase();

  // Stressed or anxious — breathing exercise
  if (stress > 0.55 || emo === "anxious" || emo === "fear") {
    return {
      icon: Wind,
      iconColor: "#0891b2",
      title: "Try a 2-minute breathing exercise",
      body: "Deep breathing activates your parasympathetic nervous system, reducing cortisol and calming your mind.",
      route: "/biofeedback",
      actionLabel: "Start breathing",
    };
  }

  // Sad or low mood
  if (emo === "sad" || val < -0.3) {
    return {
      icon: MessageCircle,
      iconColor: "#6366f1",
      title: "Talk to your AI companion",
      body: "Sometimes expressing what you feel helps process it. Your companion is here to listen without judgment.",
      route: "/ai-companion",
      actionLabel: "Open companion",
    };
  }

  // Angry — cognitive reappraisal (inline, no navigation)
  if (emo === "angry") {
    return {
      icon: Lightbulb,
      iconColor: "#d4a017",
      title: "Try cognitive reappraisal",
      body: "Pause and reframe: What triggered this? Is there another way to interpret the situation? Name the feeling without acting on it.",
      route: null,
      actionLabel: "Noted",
    };
  }

  // Happy or excited — capture the moment
  if (emo === "happy" || emo === "excited" || emo === "surprise" || val > 0.4) {
    return {
      icon: BookOpen,
      iconColor: "#a78bfa",
      title: "Capture this moment in your dream journal",
      body: "Positive emotional peaks are worth recording. Writing amplifies the experience and helps your brain encode it.",
      route: "/dreams",
      actionLabel: "Open journal",
    };
  }

  // Neutral / balanced / default — focus session
  return {
    icon: Brain,
    iconColor: "#06b6d4",
    title: "Great! Try a focus session",
    body: "Your mind is in a balanced state — an ideal starting point for neurofeedback training.",
    route: "/neurofeedback",
    actionLabel: "Start session",
  };
}

export function InterventionSuggestion({
  emotion,
  stressIndex,
  valence,
  compact = false,
}: InterventionSuggestionProps) {
  const [, navigate] = useLocation();
  const suggestion = getSuggestion(emotion, stressIndex, valence);
  const Icon = suggestion.icon;

  const handleAction = () => {
    if (suggestion.route) {
      navigate(suggestion.route);
    }
  };

  if (compact) {
    return (
      <div
        data-testid="intervention-suggestion"
        className="flex items-center gap-3 rounded-xl border border-border/20 bg-card/50 p-3"
      >
        <div
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg"
          style={{ background: `${suggestion.iconColor}15` }}
        >
          <Icon className="h-4 w-4" style={{ color: suggestion.iconColor }} />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium text-foreground leading-tight truncate">
            {suggestion.title}
          </p>
        </div>
        {suggestion.route && (
          <button
            onClick={handleAction}
            className="shrink-0 rounded-lg px-2.5 py-1 text-[10px] font-medium transition-colors hover:opacity-80"
            style={{
              background: `${suggestion.iconColor}15`,
              color: suggestion.iconColor,
            }}
            data-testid="intervention-action"
          >
            {suggestion.actionLabel}
          </button>
        )}
      </div>
    );
  }

  return (
    <div
      data-testid="intervention-suggestion"
      className="rounded-xl border border-border/20 bg-card/50 p-4 space-y-3"
    >
      <div className="flex items-start gap-3">
        <div
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg mt-0.5"
          style={{ background: `${suggestion.iconColor}15` }}
        >
          <Icon className="h-4.5 w-4.5" style={{ color: suggestion.iconColor }} />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-foreground leading-snug">
            {suggestion.title}
          </p>
          <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
            {suggestion.body}
          </p>
        </div>
      </div>
      {suggestion.route && (
        <button
          onClick={handleAction}
          className="w-full rounded-lg py-2 text-xs font-medium transition-colors hover:opacity-80"
          style={{
            background: `${suggestion.iconColor}15`,
            color: suggestion.iconColor,
          }}
          data-testid="intervention-action"
        >
          {suggestion.actionLabel}
        </button>
      )}
    </div>
  );
}

/** Exported for testing — returns the suggestion logic without rendering */
export { getSuggestion };
