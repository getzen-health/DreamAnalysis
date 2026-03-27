import type { DeviationMetric } from "./baseline-store";

export interface Intervention {
  id: string;
  trigger: { metric: DeviationMetric; direction: "high" | "low" };
  durationBucket: "2min" | "5min" | "20min";
  description: string;
  deeplink: string;
  isInline: boolean; // true = no navigation, show inline
}

export interface EffectivenessResult {
  interventionId: string;
  effective: boolean;
  recordedAt: string;
}

const LIBRARY: Intervention[] = [
  { id: "box_breathing",   trigger: { metric: "stress",  direction: "high" }, durationBucket: "2min", description: "4-4-4-4 box breathing",                        deeplink: "/biofeedback",    isInline: false },
  { id: "cold_water",      trigger: { metric: "arousal", direction: "high" }, durationBucket: "2min", description: "Cold water on wrists + 5 slow breaths",         deeplink: "",                isInline: true  },
  { id: "shake_body",      trigger: { metric: "focus",   direction: "low"  }, durationBucket: "2min", description: "Shake body for 60 seconds",                     deeplink: "",                isInline: true  },
  { id: "send_message",    trigger: { metric: "valence", direction: "low"  }, durationBucket: "2min", description: "Send one message to someone you like",           deeplink: "",                isInline: true  },
  { id: "coherent_breath", trigger: { metric: "stress",  direction: "high" }, durationBucket: "5min", description: "Guided coherent breathing (5s in / 5s out)",    deeplink: "/biofeedback",    isInline: false },
  { id: "open_focus",      trigger: { metric: "focus",   direction: "low"  }, durationBucket: "5min", description: "Open-focus meditation — defocus eyes",           deeplink: "/neurofeedback",  isInline: false },
  { id: "brain_dump",      trigger: { metric: "arousal", direction: "high" }, durationBucket: "5min", description: "Brain dump — voice note or text",                deeplink: "/ai-companion",   isInline: false },
  { id: "walk_outside",    trigger: { metric: "valence", direction: "low"  }, durationBucket: "5min", description: "Walk outside, no phone",                         deeplink: "",                isInline: true  },
  { id: "breathing_478",   trigger: { metric: "energy",  direction: "low"  }, durationBucket: "5min", description: "4-7-8 breathing + progressive muscle relaxation", deeplink: "/biofeedback",   isInline: false },
];

const PENDING_KEY = "ndw_intervention_pending";
const RESULTS_KEY = "ndw_intervention_results";
const EFFECTIVENESS_RECOVERY_Z = 0.5;
const EFFECTIVENESS_MIN_WAIT_MS = 25 * 60 * 1000;
const EFFECTIVENESS_TIMEOUT_MS  =  2 * 60 * 60 * 1000;

interface PendingEntry {
  tappedAt: string;
  metric: DeviationMetric;
  baselineZScore: number;
}

export class InterventionLibrary {
  getForDeviation(metric: DeviationMetric, direction: "high" | "low"): Intervention[] {
    return LIBRARY.filter(
      (i) => i.trigger.metric === metric && i.trigger.direction === direction,
    );
  }

  getById(id: string): Intervention | undefined {
    return LIBRARY.find((i) => i.id === id);
  }

  recordTap(interventionId: string, metric: DeviationMetric, baselineZScore: number): void {
    const pending = this.loadPending();
    // Intentional: re-tapping the same intervention resets the clock and baseline.
    // The prior tap's window is abandoned — the new tap is the measurement anchor.
    pending[interventionId] = {
      tappedAt: new Date().toISOString(),
      metric,
      baselineZScore,
    };
    this.savePending(pending);
  }

  checkEffectiveness(metric: DeviationMetric, currentZScore: number): EffectivenessResult[] {
    const pending = this.loadPending();
    const results: EffectivenessResult[] = [];
    const now = Date.now();
    const stored: EffectivenessResult[] = (() => {
      try {
        return JSON.parse(localStorage.getItem(RESULTS_KEY) ?? "[]") as EffectivenessResult[];
      } catch {
        return [];
      }
    })();

    for (const [id, entry] of Object.entries(pending)) {
      if (entry.metric !== metric) continue;
      const age = now - new Date(entry.tappedAt).getTime();
      if (age < EFFECTIVENESS_MIN_WAIT_MS) continue;

      // effective = recovered within the measurement window.
      // When age >= TIMEOUT: effective is false — the window has closed regardless
      // of current z-score, so recovery cannot be attributed to the intervention.
      const effective =
        age < EFFECTIVENESS_TIMEOUT_MS &&
        entry.baselineZScore - currentZScore > EFFECTIVENESS_RECOVERY_Z;

      const result: EffectivenessResult = {
        interventionId: id,
        effective,
        recordedAt: new Date().toISOString(),
      };
      stored.push(result);
      results.push(result);
      delete pending[id];
    }

    this.savePending(pending);
    try {
      localStorage.setItem(RESULTS_KEY, JSON.stringify(stored));
    } catch { /* storage unavailable */ }

    return results;
  }

  private loadPending(): Record<string, PendingEntry> {
    try {
      return JSON.parse(localStorage.getItem(PENDING_KEY) ?? "{}") as Record<string, PendingEntry>;
    } catch {
      return {};
    }
  }

  private savePending(pending: Record<string, PendingEntry>): void {
    try {
      localStorage.setItem(PENDING_KEY, JSON.stringify(pending));
    } catch { /* storage unavailable */ }
  }
}
