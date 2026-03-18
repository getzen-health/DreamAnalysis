/**
 * InterventionBanner — real-time closed-loop intervention notifications.
 *
 * Polls /api/interventions/check every 30 seconds and shows a dismissable
 * slide-in card when the backend recommends an action (breathing, music,
 * food, walk).  The banner respects the 10-minute server-side cooldown so
 * it will not spam the user.
 *
 * Other components that have live EEG data (emotion-lab, brain-monitor) can
 * call updateInterventionBrainState(stress, focus) to feed the current brain
 * state into the next check cycle.
 */

import { useEffect, useState, useRef, useCallback } from "react";
import { useLocation } from "wouter";
import {
  Wind,
  Music,
  Headphones,
  Apple,
  Footprints,
  X,
  ChevronRight,
  Brain,
} from "lucide-react";
import { getParticipantId } from "@/lib/participant";
import { resolveUrl } from "@/lib/queryClient";
import { getMLApiUrl } from "@/lib/ml-api";
import { triggerSpotifyPlay } from "@/components/spotify-connect";

// ── Shared brain state (module-level) ────────────────────────────────────────
// Other components write here; the banner reads it on every check cycle.
let _sharedStress = 0;
let _sharedFocus = 0.5;

/**
 * Call this from emotion-lab / brain-monitor WebSocket handlers to keep the
 * intervention engine up-to-date with the current brain state.
 */
export function updateInterventionBrainState(
  stress: number,
  focus: number,
): void {
  _sharedStress = Math.max(0, Math.min(1, stress));
  _sharedFocus = Math.max(0, Math.min(1, focus));
}

// ── Types ─────────────────────────────────────────────────────────────────────

interface Intervention {
  type: string;
  title: string;
  body: string;
  action_label: string;
  action_url: string;
  icon: string;
  evidence: string;
  priority: number;
}

interface CheckResponse {
  intervention: Intervention | null;
  has_recommendation: boolean;
}

// ── Icon mapping ──────────────────────────────────────────────────────────────

const ICON_MAP: Record<string, React.ElementType> = {
  wind: Wind,
  music: Music,
  headphones: Headphones,
  apple: Apple,
  footprints: Footprints,
};

// Priority → colour scheme
const PRIORITY_STYLE: Record<number, { border: string; badge: string; icon: string }> = {
  1: { border: "border-rose-500/40",    badge: "bg-rose-500/15 text-rose-300",    icon: "text-rose-400"    },
  2: { border: "border-amber-500/40",  badge: "bg-amber-500/15 text-amber-300", icon: "text-amber-400"  },
  3: { border: "border-indigo-500/40",   badge: "bg-indigo-500/15 text-indigo-300",  icon: "text-indigo-400"   },
};


function extraHeaders(): Record<string, string> {
  const url = getMLApiUrl();
  return url.includes("ngrok") ? { "ngrok-skip-browser-warning": "true" } : {};
}

async function mlPost<T>(endpoint: string, body: unknown): Promise<T> {
  const res = await fetch(`${getMLApiUrl()}/api${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...extraHeaders() },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`ML API ${endpoint} → ${res.status}`);
  return res.json() as Promise<T>;
}

// ── Component ─────────────────────────────────────────────────────────────────

const POLL_INTERVAL_MS = 30_000;   // 30 seconds
const INITIAL_DELAY_MS = 8_000;    // wait for page to settle before first check

export function InterventionBanner() {
  const [intervention, setIntervention] = useState<Intervention | null>(null);
  const [visible, setVisible] = useState(false);
  const [, navigate] = useLocation();
  const userId = useRef(getParticipantId());

  const checkIntervention = useCallback(async () => {
    try {
      const data = await mlPost<CheckResponse>("/interventions/check", {
        user_id: userId.current,
        stress_index: _sharedStress,
        focus_index: _sharedFocus,
        minutes_since_last_meal: null,
      });

      if (data.has_recommendation && data.intervention) {
        setIntervention(data.intervention);
        setVisible(true);

        // Auto-play Spotify when a music intervention fires (silent if not connected)
        if (data.intervention.type === "music_calm") {
          triggerSpotifyPlay("calm").catch(() => {});
        } else if (data.intervention.type === "music_focus") {
          triggerSpotifyPlay("focus").catch(() => {});
        }

        // Record that the banner was shown (starts cooldown)
        mlPost("/interventions/trigger", {
          user_id: userId.current,
          intervention_type: data.intervention.type,
          stress_before: _sharedStress,
          focus_before: _sharedFocus,
        }).catch(() => {});
      }
    } catch {
      // ML backend offline — silent fail, never block the UI
    }

    // Just-in-time push notification trigger (fires even when app is backgrounded)
    // The server enforces a 15-minute per-user cooldown so this won't spam.
    if (_sharedStress >= 0.70 || _sharedFocus <= 0.25) {
      fetch(resolveUrl("/api/notifications/brain-state-trigger"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userId: userId.current,
          stress: _sharedStress,
          focus: _sharedFocus,
        }),
      }).catch(() => {}); // silent fail if VAPID not configured
    }
  }, []);

  useEffect(() => {
    const initial = setTimeout(checkIntervention, INITIAL_DELAY_MS);
    const interval = setInterval(checkIntervention, POLL_INTERVAL_MS);
    return () => {
      clearTimeout(initial);
      clearInterval(interval);
    };
  }, [checkIntervention]);

  const handleDismiss = async () => {
    setVisible(false);
    try {
      await mlPost("/interventions/snooze", {
        user_id: userId.current,
        minutes: 10,
      });
    } catch {
      // silent
    }
  };

  const handleAction = () => {
    if (intervention?.action_url) {
      // action_url is a relative path like /biofeedback?tab=music&mood=calm
      navigate(intervention.action_url);
    }
    setVisible(false);
  };

  if (!visible || !intervention) return null;

  const Icon = ICON_MAP[intervention.icon] ?? Brain;
  const style = PRIORITY_STYLE[intervention.priority] ?? PRIORITY_STYLE[3];

  return (
    <div
      className={
        `fixed bottom-6 right-6 z-50 w-80 rounded-xl border ${style.border} ` +
        "bg-background/90 shadow-xl backdrop-blur-sm " +
        "animate-in slide-in-from-bottom-4 duration-300"
      }
      role="alert"
      aria-live="polite"
    >
      {/* Header */}
      <div className="flex items-start gap-3 px-4 pt-4 pb-2">
        <div className={`mt-0.5 shrink-0 rounded-md p-1.5 ${style.badge}`}>
          <Icon className={`h-4 w-4 ${style.icon}`} />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-foreground leading-snug">
            {intervention.title}
          </p>
          <p className="mt-1 text-xs text-muted-foreground leading-relaxed">
            {intervention.body}
          </p>
        </div>
        <button
          onClick={handleDismiss}
          className="shrink-0 rounded p-1 hover:bg-muted/50 transition-colors"
          aria-label="Dismiss"
        >
          <X className="h-3.5 w-3.5 text-muted-foreground" />
        </button>
      </div>

      {/* Action */}
      <div className="px-4 pb-4 pt-1 flex gap-2">
        <button
          onClick={handleAction}
          className={
            `flex-1 flex items-center justify-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ` +
            `${style.badge} hover:opacity-80`
          }
        >
          {intervention.action_label}
          <ChevronRight className="h-3 w-3" />
        </button>
        <button
          onClick={handleDismiss}
          className="rounded-lg px-3 py-1.5 text-xs text-muted-foreground hover:bg-muted/50 transition-colors"
        >
          Not now
        </button>
      </div>
    </div>
  );
}
