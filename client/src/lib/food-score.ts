/**
 * food-score.ts — Yuka-style health score for individual food items / meals.
 *
 * Scoring inspired by Nutri-Score / Yuka:
 *   - Positive points: protein, fiber
 *   - Negative points: calories (excess), sugar estimate, sodium estimate, fat
 *   - Score = 100 - (negative * 5) + (positive * 5), clamped 0-100
 *
 * Also generates brain-impact and mood-prediction text based on macros.
 */

// ── Types ───────────────────────────────────────────────────────────────────

export type FoodRating = "excellent" | "good" | "mediocre" | "poor";

export interface NutrientFlag {
  nutrient: string;
  level: "low" | "moderate" | "high";
  isGood: boolean;
  /** 0-100 percentage of daily recommended intake */
  dailyPct: number;
}

export interface FoodScoreResult {
  /** 0-100 overall health score */
  score: number;
  rating: FoodRating;
  /** Hex color for the score ring */
  color: string;
  /** One-line brain impact summary */
  brainImpact: string;
  /** Mood prediction text */
  moodPrediction: string;
  /** Per-nutrient breakdown */
  nutrientFlags: NutrientFlag[];
  /** Buy/eat recommendation */
  verdict: "buy" | "okay" | "avoid";
  verdictText: string;
  verdictReason: string;
  /** GLP-1 impact (for users on semaglutide/tirzepatide) */
  glpImpact: string | null;
}

export interface FoodScoreInput {
  calories: number;
  protein_g: number;
  carbs_g: number;
  fat_g: number;
  fiber_g?: number;
  sugar_g?: number;
  sodium_mg?: number;
}

// ── Constants (daily reference values) ──────────────────────────────────────

const DAILY_CALORIES = 2000;
const DAILY_PROTEIN_G = 50;
const DAILY_CARBS_G = 275;
const DAILY_FAT_G = 78;
const DAILY_FIBER_G = 28;
const DAILY_SUGAR_G = 50; // WHO recommended limit
const DAILY_SODIUM_MG = 2300;

// ── Score colors ────────────────────────────────────────────────────────────

const SCORE_COLORS = {
  excellent: "#06b6d4", // cyan-500
  good: "#22c55e",      // green-500
  mediocre: "#d4a017",  // amber
  poor: "#e879a8",      // rose
} as const;

// ── Helpers ─────────────────────────────────────────────────────────────────

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function getRating(score: number): FoodRating {
  if (score >= 70) return "excellent";
  if (score >= 50) return "good";
  if (score >= 30) return "mediocre";
  return "poor";
}

/**
 * Estimate sugar from total carbs when not available.
 * Heuristic: ~40% of carbs are sugars in a typical mixed meal.
 * This is conservative -- packaged foods often have higher sugar ratios.
 */
function estimateSugar(carbs_g: number, sugar_g?: number): number {
  if (sugar_g != null && sugar_g >= 0) return sugar_g;
  return carbs_g * 0.4;
}

/**
 * Estimate sodium when not available.
 * Average meal: ~600mg sodium per 500 calories.
 */
function estimateSodium(calories: number, sodium_mg?: number): number {
  if (sodium_mg != null && sodium_mg >= 0) return sodium_mg;
  return (calories / 500) * 600;
}

// ── Brain impact analysis ───────────────────────────────────────────────────

function analyzeBrainImpact(food: FoodScoreInput): string {
  const sugar = estimateSugar(food.carbs_g, food.sugar_g);
  const proteinCalPct = (food.protein_g * 4) / Math.max(food.calories, 1);
  const carbCalPct = (food.carbs_g * 4) / Math.max(food.calories, 1);
  const sugarRatio = sugar / Math.max(food.carbs_g, 1);

  // High sugar dominance
  if (sugar > 25 || sugarRatio > 0.6) {
    return "May cause energy crash and reduce focus -- high sugar triggers rapid glucose spike then drop";
  }

  // High protein
  if (food.protein_g > 20 && proteinCalPct > 0.25) {
    return "Supports neurotransmitter production -- protein provides amino acids for serotonin and dopamine";
  }

  // High fiber
  if ((food.fiber_g ?? 0) > 8) {
    return "Stabilizes blood sugar for steady energy -- fiber slows glucose absorption into the bloodstream";
  }

  // Balanced macro
  if (proteinCalPct > 0.15 && carbCalPct < 0.55 && carbCalPct > 0.3) {
    return "Good balance of sustained energy and cognitive fuel -- supports steady focus and mood";
  }

  // Very high fat
  if (food.fat_g > 30 && (food.fat_g * 9) / Math.max(food.calories, 1) > 0.5) {
    return "High fat may slow digestion and cause drowsiness -- energy diverted to digestion";
  }

  // Very high carb, low protein
  if (carbCalPct > 0.65 && food.protein_g < 10) {
    return "Carb-heavy without protein -- expect a quick energy burst followed by a dip";
  }

  return "Moderate brain impact -- a mix of nutrients provides baseline cognitive support";
}

// ── Mood prediction ─────────────────────────────────────────────────────────

function predictMood(food: FoodScoreInput): string {
  const sugar = estimateSugar(food.carbs_g, food.sugar_g);
  const proteinCalPct = (food.protein_g * 4) / Math.max(food.calories, 1);
  const sugarRatio = sugar / Math.max(food.carbs_g, 1);

  // High protein + moderate carb = mood boost
  if (food.protein_g > 15 && proteinCalPct > 0.2 && food.carbs_g > 20 && food.carbs_g < 80) {
    return "This meal may boost your mood for 2-3 hours -- protein and moderate carbs support serotonin production";
  }

  // High sugar spike
  if (sugar > 30 || sugarRatio > 0.65) {
    return "Sugar spike likely -- expect an energy crash in 1-2 hours and possible irritability";
  }

  // Very low calorie
  if (food.calories < 150) {
    return "Light snack -- may not sustain mood for long. Consider pairing with protein if you feel hungry again soon";
  }

  // Heavy meal
  if (food.calories > 800) {
    return "Large meal may cause post-meal drowsiness -- blood flow shifts to digestion for 1-2 hours";
  }

  // High fiber + moderate protein
  if ((food.fiber_g ?? 0) > 5 && food.protein_g > 10) {
    return "Steady energy ahead -- fiber and protein together create a slow, sustained mood lift";
  }

  // High fat comfort food
  if (food.fat_g > 25 && food.calories > 500) {
    return "Comfort food effect -- fat triggers short-term satisfaction, but energy may dip in 2 hours";
  }

  return "Moderate mood impact -- a balanced meal supports stable energy and focus";
}

// ── Main scoring function ───────────────────────────────────────────────────

export function calculateFoodScore(food: FoodScoreInput): FoodScoreResult {
  const sugar = estimateSugar(food.carbs_g, food.sugar_g);
  const sodium = estimateSodium(food.calories, food.sodium_mg);

  // ── Negative points (0-10 scale each, higher = worse) ──

  // Calories: penalize above ~500 per meal (25% of daily)
  const calNeg = clamp((food.calories - 300) / 150, 0, 3);

  // Sugar: penalize above 10g
  const sugarNeg = clamp((sugar - 5) / 8, 0, 3);

  // Sodium: penalize above 500mg
  const sodiumNeg = clamp((sodium - 400) / 300, 0, 2);

  // Saturated fat proxy: assume ~40% of total fat is saturated
  const satFatEstimate = food.fat_g * 0.4;
  const fatNeg = clamp((satFatEstimate - 5) / 5, 0, 2);

  const totalNegative = calNeg + sugarNeg + sodiumNeg + fatNeg;

  // ── Positive points (0-10 scale each, higher = better) ──

  // Protein: reward above 10g
  const proteinPos = clamp((food.protein_g - 5) / 8, 0, 3);

  // Fiber: reward above 3g
  const fiberPos = clamp(((food.fiber_g ?? 0) - 1) / 4, 0, 3);

  // Variety bonus: if meal has decent protein AND fiber AND not too much sugar
  const varietyPos = (food.protein_g > 10 && (food.fiber_g ?? 0) > 3 && sugar < 15) ? 1 : 0;

  const totalPositive = proteinPos + fiberPos + varietyPos;

  // ── Final score ──

  const rawScore = 100 - (totalNegative * 7) + (totalPositive * 5);
  const score = Math.round(clamp(rawScore, 0, 100));
  const rating = getRating(score);

  // ── Nutrient flags ──

  const nutrientFlags: NutrientFlag[] = [
    {
      nutrient: "Calories",
      level: food.calories > 600 ? "high" : food.calories > 300 ? "moderate" : "low",
      isGood: food.calories <= 600,
      dailyPct: Math.round((food.calories / DAILY_CALORIES) * 100),
    },
    {
      nutrient: "Protein",
      level: food.protein_g > 20 ? "high" : food.protein_g > 10 ? "moderate" : "low",
      isGood: food.protein_g >= 10,
      dailyPct: Math.round((food.protein_g / DAILY_PROTEIN_G) * 100),
    },
    {
      nutrient: "Sugar",
      level: sugar > 20 ? "high" : sugar > 10 ? "moderate" : "low",
      isGood: sugar <= 10,
      dailyPct: Math.round((sugar / DAILY_SUGAR_G) * 100),
    },
    {
      nutrient: "Fiber",
      level: (food.fiber_g ?? 0) > 8 ? "high" : (food.fiber_g ?? 0) > 3 ? "moderate" : "low",
      isGood: (food.fiber_g ?? 0) >= 3,
      dailyPct: Math.round(((food.fiber_g ?? 0) / DAILY_FIBER_G) * 100),
    },
    {
      nutrient: "Fat",
      level: food.fat_g > 25 ? "high" : food.fat_g > 12 ? "moderate" : "low",
      isGood: food.fat_g <= 25,
      dailyPct: Math.round((food.fat_g / DAILY_FAT_G) * 100),
    },
    {
      nutrient: "Sodium",
      level: sodium > 800 ? "high" : sodium > 400 ? "moderate" : "low",
      isGood: sodium <= 600,
      dailyPct: Math.round((sodium / DAILY_SODIUM_MG) * 100),
    },
  ];

  // ── Verdict: Buy / Okay / Avoid ─────────────────────────────────────
  let verdict: "buy" | "okay" | "avoid";
  let verdictText: string;
  let verdictReason: string;

  if (score >= 70) {
    verdict = "buy";
    verdictText = "Recommended";
    verdictReason = "High nutritional value. Good balance of protein and fiber with low sugar. Great choice for sustained energy and brain health.";
  } else if (score >= 40) {
    verdict = "okay";
    verdictText = "Acceptable";
    verdictReason = "Moderate nutritional value. Fine in moderation but consider pairing with more protein or fiber to balance the meal.";
  } else {
    verdict = "avoid";
    verdictText = "Not recommended";
    verdictReason = sugar > 20
      ? "High sugar content will spike blood glucose and crash your energy. Consider a lower-sugar alternative."
      : food.fat_g > 30
      ? "Excessive fat relative to other nutrients. If consumed, balance with vegetables and lean protein."
      : "Poor nutritional profile — high in empty calories with little beneficial nutrients. Look for alternatives with more protein and fiber.";
  }

  // ── GLP-1 impact (for users on Ozempic/Wegovy/Mounjaro) ──────────
  let glpImpact: string | null = null;
  try {
    const onGlp = localStorage.getItem("ndw_glp1_active") === "true";
    if (onGlp) {
      if (sugar > 25) {
        glpImpact = "High sugar — may cause nausea with GLP-1. Choose low-glycemic foods to minimize GI side effects.";
      } else if (food.fat_g > 30) {
        glpImpact = "High fat — may slow gastric emptying further with GLP-1. Smaller portions recommended to avoid discomfort.";
      } else if (food.protein_g > 20 && sugar < 10) {
        glpImpact = "Great choice on GLP-1 — high protein keeps you full longer and supports muscle preservation during weight loss.";
      } else if (food.fiber_g && food.fiber_g > 5) {
        glpImpact = "Good fiber content — helps with digestion on GLP-1. Stay hydrated.";
      } else {
        glpImpact = "Moderate choice on GLP-1. Aim for 20g+ protein per meal to preserve lean mass.";
      }
    }
  } catch { /* no localStorage access */ }

  return {
    score,
    rating,
    color: SCORE_COLORS[rating],
    brainImpact: analyzeBrainImpact(food),
    moodPrediction: predictMood(food),
    nutrientFlags,
    verdict,
    verdictText,
    verdictReason,
    glpImpact,
  };
}
