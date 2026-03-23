/**
 * cycle-phase-adjustment.ts — Menstrual cycle phase adjustment for EEG emotion baselines.
 *
 * If cycle data exists, adjusts emotion baselines based on hormonal phase:
 *   - Luteal phase (days 15-28): estrogen/progesterone drop -> lower mood, higher irritability
 *   - Follicular phase (days 6-13): rising estrogen -> improved mood baseline
 *   - Ovulatory phase (days 14-15): peak estrogen -> highest mood and energy
 *   - Menstrual (days 1-5): lower energy baseline
 *
 * Phase boundaries are adapted for different cycle lengths.
 * Adjustments are small (+-0.15 max) -- they shift baselines, not override signals.
 */

// ── Types ──────────────────────────────────────────────────────────────────

export type CyclePhase = "menstrual" | "follicular" | "ovulatory" | "luteal";

export interface CyclePhaseAdjustment {
  moodOffset: number;          // positive = mood naturally higher
  energyOffset: number;        // positive = energy naturally higher
  irritabilityOffset: number;  // positive = irritability naturally higher
}

export interface CyclePhaseContext {
  phase: CyclePhase;
  message: string;
  dayInCycle: number;
}

// ── Phase determination ────────────────────────────────────────────────────

/**
 * Determine cycle phase from day number and cycle length.
 *
 * Standard 28-day cycle:
 *   Days  1-5:   Menstrual
 *   Days  6-13:  Follicular
 *   Days 14-15:  Ovulatory (around ovulation)
 *   Days 16-28:  Luteal
 *
 * For other cycle lengths, phases are scaled proportionally:
 *   - Menstrual: always days 1-5 (relatively constant)
 *   - Luteal: always ~14 days before end (fairly constant)
 *   - Ovulatory: 2 days before luteal
 *   - Follicular: everything between menstrual and ovulatory
 */
export function getCyclePhase(dayInCycle: number, cycleLength: number): CyclePhase | null {
  if (dayInCycle <= 0 || dayInCycle > cycleLength) return null;

  // Menstrual: always days 1-5
  const menstrualEnd = 5;

  // Luteal phase starts ~12 days before cycle end (biologically stable at ~14 days)
  const lutealStart = Math.max(menstrualEnd + 4, cycleLength - 12);

  // Ovulatory: 2 days before luteal (days 14-15 in a 28-day cycle)
  const ovulatoryStart = lutealStart - 2;

  // Follicular: between menstrual end and ovulatory start
  const follicularEnd = ovulatoryStart - 1;

  if (dayInCycle <= menstrualEnd) return "menstrual";
  if (dayInCycle <= follicularEnd) return "follicular";
  if (dayInCycle < lutealStart) return "ovulatory";
  return "luteal";
}

// ── Phase adjustments ──────────────────────────────────────────────────────

const PHASE_ADJUSTMENTS: Record<CyclePhase, CyclePhaseAdjustment> = {
  menstrual: {
    moodOffset: -0.05,
    energyOffset: -0.12,
    irritabilityOffset: 0.05,
  },
  follicular: {
    moodOffset: 0.10,
    energyOffset: 0.08,
    irritabilityOffset: -0.05,
  },
  ovulatory: {
    moodOffset: 0.12,
    energyOffset: 0.10,
    irritabilityOffset: -0.08,
  },
  luteal: {
    moodOffset: -0.12,
    energyOffset: -0.05,
    irritabilityOffset: 0.12,
  },
};

export function getCyclePhaseAdjustment(phase: CyclePhase): CyclePhaseAdjustment {
  return PHASE_ADJUSTMENTS[phase];
}

// ── Phase context messages ─────────────────────────────────────────────────

const PHASE_MESSAGES: Record<CyclePhase, string> = {
  menstrual: "Menstrual phase — energy may be lower than usual. Rest is restorative.",
  follicular: "Follicular phase — rising estrogen supports improved mood and focus.",
  ovulatory: "Ovulatory phase — peak energy and mood. Great time for challenging tasks.",
  luteal: "Luteal phase — mood may be lower than usual. Be gentle with yourself.",
};

export function getCyclePhaseContext(phase: CyclePhase): CyclePhaseContext {
  return {
    phase,
    message: PHASE_MESSAGES[phase],
    dayInCycle: 0, // Caller should set this
  };
}

// ── Compute current phase from stored cycle data ──────────────────────────

/**
 * Given last period start date and cycle length, compute today's cycle day and phase.
 * Returns null if no cycle data or date is invalid.
 */
export function getCurrentCyclePhase(
  lastPeriodStart: string | undefined | null,
  cycleLength: number = 28,
): { phase: CyclePhase; dayInCycle: number } | null {
  if (!lastPeriodStart) return null;

  try {
    const startDate = new Date(lastPeriodStart);
    if (isNaN(startDate.getTime())) return null;

    const today = new Date();
    const diffMs = today.getTime() - startDate.getTime();
    const diffDays = Math.floor(diffMs / (24 * 60 * 60 * 1000));

    if (diffDays < 0) return null;

    // Compute day in current cycle (wrapping if past one cycle)
    const dayInCycle = (diffDays % cycleLength) + 1;

    const phase = getCyclePhase(dayInCycle, cycleLength);
    if (!phase) return null;

    return { phase, dayInCycle };
  } catch {
    return null;
  }
}
