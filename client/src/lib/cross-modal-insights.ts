/**
 * cross-modal-insights.ts — Pure function library that correlates historical
 * health data across modalities and produces personalized insight strings.
 *
 * No API calls, no side effects — just computation over arrays of data.
 */

// ── Types ──────────────────────────────────────────────────────────────────

export interface CorrelationInput {
  emotionHistory: {
    stress: number;
    focus: number;
    valence: number;
    energy: number;
    timestamp: string;
  }[];
  foodLogs: {
    mealType: string;
    totalCalories: number;
    loggedAt: string;
  }[];
  sleepData: {
    quality: number;   // 0-100
    duration: number;  // hours
    date: string;      // YYYY-MM-DD
  }[];
  steps: {
    count: number;
    date: string;      // YYYY-MM-DD
  }[];
}

export type InsightCategory =
  | "food_sleep"
  | "exercise_mood"
  | "sleep_focus"
  | "time_pattern"
  | "streak_mood"
  | "food_stress";

export interface PersonalInsight {
  id: string;
  text: string;
  category: InsightCategory;
  confidence: "strong" | "moderate" | "weak";
  dataPoints: number;
}

// ── Helpers ────────────────────────────────────────────────────────────────

const MIN_SAMPLES = 5;
const MIN_PERCENT_DIFF = 10;

/**
 * Confidence is based on the smaller of the two comparison groups.
 * This reflects how reliable the comparison is — a group of 5 vs 100
 * is only as reliable as the group of 5.
 */
function getConfidence(minGroupSize: number): "strong" | "moderate" | "weak" {
  if (minGroupSize >= 15) return "strong";
  if (minGroupSize >= 10) return "moderate";
  return "weak";
}

function pctDiff(a: number, b: number): number {
  if (b === 0) return 0;
  return Math.round(((a - b) / Math.abs(b)) * 100);
}

function dateStr(d: Date): string {
  return d.toISOString().slice(0, 10);
}

function getHour(timestamp: string): number {
  return new Date(timestamp).getUTCHours();
}

function getDayOfWeek(timestamp: string): number {
  return new Date(timestamp).getUTCDay(); // 0=Sun, 6=Sat
}

function isWeekend(dayOfWeek: number): boolean {
  return dayOfWeek === 0 || dayOfWeek === 6;
}

/** Get the next calendar date string from a YYYY-MM-DD date */
function nextDate(dateString: string): string {
  const d = new Date(dateString + "T12:00:00");
  d.setDate(d.getDate() + 1);
  return dateStr(d);
}

function avg(nums: number[]): number {
  if (nums.length === 0) return 0;
  return nums.reduce((s, n) => s + n, 0) / nums.length;
}

// ── Correlation Functions ──────────────────────────────────────────────────

/**
 * 1. Late eating -> sleep quality
 * Split nights into "ate after 9pm" vs "no late food". Compare avg sleep quality.
 */
function detectLateEatingSleep(input: CorrelationInput): PersonalInsight | null {
  const { foodLogs, sleepData } = input;
  if (sleepData.length < MIN_SAMPLES) return null;

  // Build a set of dates where the user ate after 9pm
  const lateEatingDates = new Set<string>();
  for (const log of foodLogs) {
    if (!log.loggedAt) continue;
    const hour = getHour(log.loggedAt);
    if (hour >= 21) {
      lateEatingDates.add(dateStr(new Date(log.loggedAt)));
    }
  }

  // Sleep quality grouped by whether the user ate late the night before
  const lateGroup: number[] = [];
  const noLateGroup: number[] = [];

  for (const sleep of sleepData) {
    if (sleep.quality == null) continue;
    // The relevant eating is from the night before: same date as sleep.date
    // (assuming sleep.date is the night of the sleep)
    if (lateEatingDates.has(sleep.date)) {
      lateGroup.push(sleep.quality);
    } else {
      noLateGroup.push(sleep.quality);
    }
  }

  if (lateGroup.length < MIN_SAMPLES || noLateGroup.length < MIN_SAMPLES) return null;

  const avgLate = avg(lateGroup);
  const avgNoLate = avg(noLateGroup);
  const diff = pctDiff(avgNoLate, avgLate);

  if (Math.abs(diff) < MIN_PERCENT_DIFF) return null;

  const totalSamples = lateGroup.length + noLateGroup.length;
  const minGroup = Math.min(lateGroup.length, noLateGroup.length);

  if (diff > 0) {
    return {
      id: "late-eating-sleep",
      text: `You sleep ${diff}% better on nights you don't eat after 9pm`,
      category: "food_sleep",
      confidence: getConfidence(minGroup),
      dataPoints: totalSamples,
    };
  } else {
    return {
      id: "late-eating-sleep",
      text: `You sleep ${Math.abs(diff)}% better on nights you eat after 9pm`,
      category: "food_sleep",
      confidence: getConfidence(minGroup),
      dataPoints: totalSamples,
    };
  }
}

/**
 * 2. Exercise -> next-day mood
 * Days with >7000 steps vs <3000 steps -> compare next-day avg valence.
 */
function detectExerciseMood(input: CorrelationInput): PersonalInsight | null {
  const { steps, emotionHistory } = input;
  if (steps.length < MIN_SAMPLES || emotionHistory.length < MIN_SAMPLES) return null;

  // Build map of date -> average valence
  const valenceByDate = new Map<string, number[]>();
  for (const e of emotionHistory) {
    const d = dateStr(new Date(e.timestamp));
    if (!valenceByDate.has(d)) valenceByDate.set(d, []);
    valenceByDate.get(d)!.push(e.valence);
  }

  const highActivityNextDayValence: number[] = [];
  const lowActivityNextDayValence: number[] = [];

  for (const step of steps) {
    const next = nextDate(step.date);
    const nextValences = valenceByDate.get(next);
    if (!nextValences || nextValences.length === 0) continue;

    const avgV = avg(nextValences);
    if (step.count > 7000) {
      highActivityNextDayValence.push(avgV);
    } else if (step.count < 3000) {
      lowActivityNextDayValence.push(avgV);
    }
  }

  if (highActivityNextDayValence.length < MIN_SAMPLES || lowActivityNextDayValence.length < MIN_SAMPLES) return null;

  const avgHigh = avg(highActivityNextDayValence);
  const avgLow = avg(lowActivityNextDayValence);

  // Valence is -1 to 1 — normalize to 0-100 for percentage
  const highNorm = (avgHigh + 1) * 50;
  const lowNorm = (avgLow + 1) * 50;
  const diff = pctDiff(highNorm, lowNorm);

  if (Math.abs(diff) < MIN_PERCENT_DIFF) return null;

  const totalSamples = highActivityNextDayValence.length + lowActivityNextDayValence.length;
  const minGroup = Math.min(highActivityNextDayValence.length, lowActivityNextDayValence.length);

  if (diff > 0) {
    return {
      id: "exercise-mood",
      text: `Active days boost your next-day mood by ${diff}%`,
      category: "exercise_mood",
      confidence: getConfidence(minGroup),
      dataPoints: totalSamples,
    };
  } else {
    return {
      id: "exercise-mood",
      text: `Sedentary days boost your next-day mood by ${Math.abs(diff)}%`,
      category: "exercise_mood",
      confidence: getConfidence(minGroup),
      dataPoints: totalSamples,
    };
  }
}

/**
 * 3. Sleep duration -> focus
 * Days after >=7h sleep vs <6h sleep -> compare avg focus.
 */
function detectSleepFocus(input: CorrelationInput): PersonalInsight | null {
  const { sleepData, emotionHistory } = input;
  if (sleepData.length < MIN_SAMPLES || emotionHistory.length < MIN_SAMPLES) return null;

  // Build map of date -> sleep duration
  const sleepByDate = new Map<string, number>();
  for (const s of sleepData) {
    sleepByDate.set(s.date, s.duration);
  }

  // Build map of date -> average focus
  const focusByDate = new Map<string, number[]>();
  for (const e of emotionHistory) {
    const d = dateStr(new Date(e.timestamp));
    if (!focusByDate.has(d)) focusByDate.set(d, []);
    focusByDate.get(d)!.push(e.focus);
  }

  const goodSleepFocus: number[] = [];
  const poorSleepFocus: number[] = [];

  for (const [sleepDate, duration] of sleepByDate) {
    const next = nextDate(sleepDate);
    const focusEntries = focusByDate.get(next);
    if (!focusEntries || focusEntries.length === 0) continue;

    const avgF = avg(focusEntries);
    if (duration >= 7) {
      goodSleepFocus.push(avgF);
    } else if (duration < 6) {
      poorSleepFocus.push(avgF);
    }
  }

  if (goodSleepFocus.length < MIN_SAMPLES || poorSleepFocus.length < MIN_SAMPLES) return null;

  const avgGood = avg(goodSleepFocus);
  const avgPoor = avg(poorSleepFocus);
  // Focus is 0-1 — normalize to 0-100
  const diff = pctDiff(avgGood * 100, avgPoor * 100);

  if (Math.abs(diff) < MIN_PERCENT_DIFF) return null;

  const totalSamples = goodSleepFocus.length + poorSleepFocus.length;
  const minGroup = Math.min(goodSleepFocus.length, poorSleepFocus.length);

  if (diff > 0) {
    return {
      id: "sleep-focus",
      text: `You're ${diff}% more focused after 7+ hours of sleep`,
      category: "sleep_focus",
      confidence: getConfidence(minGroup),
      dataPoints: totalSamples,
    };
  } else {
    return {
      id: "sleep-focus",
      text: `You're ${Math.abs(diff)}% more focused after shorter sleep`,
      category: "sleep_focus",
      confidence: getConfidence(minGroup),
      dataPoints: totalSamples,
    };
  }
}

/**
 * 4. Morning vs evening mood
 * Compare avg valence before noon vs after 6pm.
 */
function detectTimePatternMood(input: CorrelationInput): PersonalInsight | null {
  const { emotionHistory } = input;

  const morningValence: number[] = [];
  const eveningValence: number[] = [];

  for (const e of emotionHistory) {
    const hour = getHour(e.timestamp);
    if (hour < 12) {
      morningValence.push(e.valence);
    } else if (hour >= 18) {
      eveningValence.push(e.valence);
    }
  }

  if (morningValence.length < MIN_SAMPLES || eveningValence.length < MIN_SAMPLES) return null;

  const avgMorning = avg(morningValence);
  const avgEvening = avg(eveningValence);

  // Normalize valence to 0-100 for percent comparison
  const morningNorm = (avgMorning + 1) * 50;
  const eveningNorm = (avgEvening + 1) * 50;
  const diff = pctDiff(morningNorm, eveningNorm);

  if (Math.abs(diff) < MIN_PERCENT_DIFF) return null;

  const totalSamples = morningValence.length + eveningValence.length;
  const minGroup = Math.min(morningValence.length, eveningValence.length);

  if (diff > 0) {
    return {
      id: "time-pattern-mood",
      text: `Your mood is ${diff}% higher in the mornings`,
      category: "time_pattern",
      confidence: getConfidence(minGroup),
      dataPoints: totalSamples,
    };
  } else {
    return {
      id: "time-pattern-mood",
      text: `Your mood is ${Math.abs(diff)}% higher in the evenings`,
      category: "time_pattern",
      confidence: getConfidence(minGroup),
      dataPoints: totalSamples,
    };
  }
}

/**
 * 5. Stress after meals
 * Compare avg stress 2h after meals vs baseline (all other stress readings).
 */
function detectMealStress(input: CorrelationInput): PersonalInsight | null {
  const { foodLogs, emotionHistory } = input;
  if (foodLogs.length < MIN_SAMPLES || emotionHistory.length < MIN_SAMPLES) return null;

  // Sort emotion history by timestamp
  const sorted = [...emotionHistory].sort(
    (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
  );

  // For each meal, find emotion readings within 2h after
  const postMealStress: number[] = [];
  const postMealTimestamps = new Set<string>();

  for (const meal of foodLogs) {
    if (!meal.loggedAt) continue;
    const mealTime = new Date(meal.loggedAt).getTime();
    const twoHoursLater = mealTime + 2 * 60 * 60 * 1000;

    for (const e of sorted) {
      const eTime = new Date(e.timestamp).getTime();
      if (eTime >= mealTime && eTime <= twoHoursLater) {
        postMealStress.push(e.stress);
        postMealTimestamps.add(e.timestamp);
      }
    }
  }

  // Baseline = all readings NOT within 2h of any meal
  const baselineStress: number[] = [];
  for (const e of sorted) {
    if (!postMealTimestamps.has(e.timestamp)) {
      baselineStress.push(e.stress);
    }
  }

  if (postMealStress.length < MIN_SAMPLES || baselineStress.length < MIN_SAMPLES) return null;

  const avgPost = avg(postMealStress);
  const avgBaseline = avg(baselineStress);
  // Stress is 0-1 — normalize to 0-100
  const diff = pctDiff(avgPost * 100, avgBaseline * 100);

  if (Math.abs(diff) < MIN_PERCENT_DIFF) return null;

  const totalSamples = postMealStress.length + baselineStress.length;
  const minGroup = Math.min(postMealStress.length, baselineStress.length);

  if (diff > 0) {
    return {
      id: "meal-stress",
      text: `Your stress rises ${diff}% after eating`,
      category: "food_stress",
      confidence: getConfidence(minGroup),
      dataPoints: totalSamples,
    };
  } else {
    return {
      id: "meal-stress",
      text: `Your stress drops ${Math.abs(diff)}% after eating`,
      category: "food_stress",
      confidence: getConfidence(minGroup),
      dataPoints: totalSamples,
    };
  }
}

/**
 * 6. Weekend vs weekday focus
 * Compare avg focus on weekdays vs weekends.
 */
function detectWeekdayFocus(input: CorrelationInput): PersonalInsight | null {
  const { emotionHistory } = input;

  const weekdayFocus: number[] = [];
  const weekendFocus: number[] = [];

  for (const e of emotionHistory) {
    const day = getDayOfWeek(e.timestamp);
    if (isWeekend(day)) {
      weekendFocus.push(e.focus);
    } else {
      weekdayFocus.push(e.focus);
    }
  }

  if (weekdayFocus.length < MIN_SAMPLES || weekendFocus.length < MIN_SAMPLES) return null;

  const avgWeekday = avg(weekdayFocus);
  const avgWeekend = avg(weekendFocus);
  // Focus is 0-1 — normalize to 0-100
  const diff = pctDiff(avgWeekday * 100, avgWeekend * 100);

  if (Math.abs(diff) < MIN_PERCENT_DIFF) return null;

  const totalSamples = weekdayFocus.length + weekendFocus.length;
  const minGroup = Math.min(weekdayFocus.length, weekendFocus.length);

  if (diff > 0) {
    return {
      id: "weekday-focus",
      text: `Your focus is ${diff}% higher on weekdays`,
      category: "streak_mood",
      confidence: getConfidence(minGroup),
      dataPoints: totalSamples,
    };
  } else {
    return {
      id: "weekday-focus",
      text: `Your focus is ${Math.abs(diff)}% higher on weekends`,
      category: "streak_mood",
      confidence: getConfidence(minGroup),
      dataPoints: totalSamples,
    };
  }
}

// ── Main Export ─────────────────────────────────────────────────────────────

/**
 * Analyzes all available historical data and returns personalized insight strings.
 * Pure function — no side effects, no API calls.
 */
export function generateInsights(input: CorrelationInput): PersonalInsight[] {
  const detectors = [
    detectLateEatingSleep,
    detectExerciseMood,
    detectSleepFocus,
    detectTimePatternMood,
    detectMealStress,
    detectWeekdayFocus,
  ];

  const insights: PersonalInsight[] = [];

  for (const detect of detectors) {
    const result = detect(input);
    if (result) {
      insights.push(result);
    }
  }

  // Sort by confidence strength (strong first) then by data points
  const order = { strong: 0, moderate: 1, weak: 2 };
  insights.sort((a, b) => {
    const confDiff = order[a.confidence] - order[b.confidence];
    if (confDiff !== 0) return confDiff;
    return b.dataPoints - a.dataPoints;
  });

  return insights;
}
