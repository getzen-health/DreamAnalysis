/**
 * Inter-brain synchrony computation for couples meditation.
 *
 * Uses power envelope correlation -- the simplest valid hyperscanning metric.
 * True hyperscanning uses phase-locking value (PLV) or coherence, which requires
 * time-synchronized raw data from both devices. Power envelope correlation is a
 * reasonable proxy that works with the data available from separate device streams.
 *
 * Research basis:
 * - Communications Biology 2026: brain synchrony is real and measurable
 * - Romantic couples show strongest inter-brain coupling (Nonlinear Dynamics 2025)
 * - Music therapy hyperscanning: therapist-patient neural sync measurable
 */

export interface BrainSyncResult {
  /** Overall synchronization score (0-1) */
  overallSync: number;
  /** Per-band synchrony scores */
  bandSync: {
    /** Alpha-band synchrony -- most relevant for meditation */
    alpha: number;
    /** Theta-band synchrony -- deep meditation */
    theta: number;
    /** Beta-band synchrony -- active engagement */
    beta: number;
  };
  /** Current synchrony phase */
  phase: "connecting" | "syncing" | "in_sync" | "deep_sync";
  /** Human-readable status message */
  message: string;
}

/**
 * Compute per-band synchrony using power envelope correlation.
 *
 * Formula: sync = 1 - |p1 - p2| / max(p1, p2)
 * Returns 0 (totally different) to 1 (identical power levels).
 * When both powers are zero, returns 1 (identical -- both silent).
 */
function bandSynchrony(p1: number, p2: number): number {
  const maxPower = Math.max(p1, p2);
  if (maxPower === 0) return 1; // Both zero = identical = fully synced
  return 1 - Math.abs(p1 - p2) / maxPower;
}

/** Phase thresholds */
const PHASE_THRESHOLDS = {
  deep_sync: 0.7,
  in_sync: 0.5,
  syncing: 0.3,
} as const;

/** Phase messages */
const PHASE_MESSAGES: Record<BrainSyncResult["phase"], string> = {
  connecting: "Finding each other's rhythm...",
  syncing: "Your brains are beginning to synchronize",
  in_sync: "Beautiful -- your brain waves are in sync",
  deep_sync: "Deep synchrony achieved -- breathing as one mind",
};

/**
 * Compute inter-brain synchrony between two people's EEG band powers.
 *
 * @param person1Alpha - Person 1's alpha band power (0-1)
 * @param person1Theta - Person 1's theta band power (0-1)
 * @param person1Beta  - Person 1's beta band power (0-1)
 * @param person2Alpha - Person 2's alpha band power (0-1)
 * @param person2Theta - Person 2's theta band power (0-1)
 * @param person2Beta  - Person 2's beta band power (0-1)
 */
export function computeBrainSync(
  person1Alpha: number,
  person1Theta: number,
  person1Beta: number,
  person2Alpha: number,
  person2Theta: number,
  person2Beta: number,
): BrainSyncResult {
  const alphaSync = bandSynchrony(person1Alpha, person2Alpha);
  const thetaSync = bandSynchrony(person1Theta, person2Theta);
  const betaSync = bandSynchrony(person1Beta, person2Beta);

  // Weighted average: alpha 50%, theta 30%, beta 20%
  // Alpha is most relevant for meditation synchrony
  const overallSync = 0.5 * alphaSync + 0.3 * thetaSync + 0.2 * betaSync;

  // Determine phase from overall sync score
  let phase: BrainSyncResult["phase"];
  if (overallSync >= PHASE_THRESHOLDS.deep_sync) {
    phase = "deep_sync";
  } else if (overallSync >= PHASE_THRESHOLDS.in_sync) {
    phase = "in_sync";
  } else if (overallSync >= PHASE_THRESHOLDS.syncing) {
    phase = "syncing";
  } else {
    phase = "connecting";
  }

  return {
    overallSync,
    bandSync: {
      alpha: alphaSync,
      theta: thetaSync,
      beta: betaSync,
    },
    phase,
    message: PHASE_MESSAGES[phase],
  };
}
