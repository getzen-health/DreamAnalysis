import { useState, useMemo, useRef, useCallback, useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { resolveUrl, apiRequest } from "@/lib/queryClient";
import { getParticipantId } from "@/lib/participant";
import { hapticSuccess } from "@/lib/haptics";
import { useVoiceData } from "@/hooks/use-voice-data";
import { useScores } from "@/hooks/use-scores";
import { ScoreGauge } from "@/components/score-gauge";
import { lookupBarcode, type BarcodeProduct } from "@/lib/barcode-api";
import { cardVariants, listItemVariants } from "@/lib/animations";
import { syncFoodLogToML } from "@/lib/ml-api";
import { RecentReadings, formatTimeAgo } from "@/components/recent-readings";
import { UtensilsCrossed, Brain as BrainIcon } from "lucide-react";
import { FoodScoreCard } from "@/components/food-score-card";
import { SectionErrorBoundary } from "@/components/section-error-boundary";
import { getEmotionHistory, getFoodLogs as sbGetFoodLogs, sbGetGeneric, sbGetSetting, sbSaveGeneric } from "../lib/supabase-store";
import {
  computeMealCognitiveCorrelation,
  generateInsight,
  type TimestampedReading,
  type MealTimestamp,
  type MealCognitiveInsight,
} from "@/lib/meal-cognitive-correlation";
// Supabase writes are done via dynamic import("@/lib/supabase-browser") to avoid
// double-writing localStorage (this file manages its own localStorage entries).
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

// ── Types ─────────────────────────────────────────────────────────────────────

interface FoodItem {
  name: string;
  portion: string;
  calories: number;
  carbs_g: number;
  protein_g: number;
  fat_g: number;
}

interface VitaminData {
  vitamin_d_mcg: number;
  vitamin_b12_mcg: number;
  vitamin_c_mg: number;
  iron_mg: number;
  magnesium_mg: number;
  zinc_mg: number;
  omega3_g: number;
}

interface FoodLog {
  id: string;
  loggedAt: string;
  mealType: string | null;
  summary: string | null;
  totalCalories: number | null;
  dominantMacro: string | null;
  foodItems: FoodItem[] | null;
  vitamins?: VitaminData | null;
}

interface FavoriteMeal {
  id: string;
  summary: string;
  mealType: string;
  foodItems: FoodItem[];
  totalCalories: number;
  savedAt: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const MEAL_ICONS: Record<string, string> = {
  breakfast: "🌅",
  lunch: "☀️",
  dinner: "🌙",
  snack: "🍎",
};

function isToday(iso: string): boolean {
  return new Date(iso).toDateString() === new Date().toDateString();
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function getMealLabel(mealType: string | null): string {
  if (!mealType) return "Meal";
  return mealType.charAt(0).toUpperCase() + mealType.slice(1);
}

function autoMealType(): string {
  const h = new Date().getHours();
  if (h >= 5 && h < 10) return "breakfast";
  if (h >= 11 && h < 15) return "lunch";
  if (h >= 17 && h < 22) return "dinner";
  return "snack";
}

function getCravingFromVoice(voice: { stress_index?: number; valence?: number } | null): { text: string; label: string } {
  if (!voice) return { text: "balanced -- you're eating from hunger, not emotion", label: "Balanced" };
  const stress = voice.stress_index ?? 0;
  const valence = voice.valence ?? 0;
  if (stress > 0.6) return { text: "stress eating -- your body seeks comfort food", label: "Stress" };
  if (valence > 0.3) return { text: "mindful eating -- you're calm and present", label: "Mindful" };
  if (valence < -0.2) return { text: "comfort seeking -- emotional eating tendency", label: "Comfort" };
  return { text: "balanced -- you're eating from hunger, not emotion", label: "Balanced" };
}

// ── Favorites localStorage helpers ────────────────────────────────────────────

const FAVORITES_KEY = "ndw_favorite_meals";

function loadFavorites(): FavoriteMeal[] {
  try {
    const raw = sbGetSetting(FAVORITES_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveFavorites(favs: FavoriteMeal[]) {
  try {
    sbSaveGeneric(FAVORITES_KEY, favs);
  } catch {}
}

function toggleFavorite(log: FoodLog): FavoriteMeal[] {
  const favs = loadFavorites();
  const idx = favs.findIndex((f) => f.id === log.id);
  if (idx >= 0) {
    favs.splice(idx, 1);
  } else {
    favs.unshift({
      id: log.id,
      summary: log.summary ?? "Meal",
      mealType: log.mealType ?? "snack",
      foodItems: log.foodItems ?? [],
      totalCalories: log.totalCalories ?? 0,
      savedAt: new Date().toISOString(),
    });
    // Keep max 20 favorites
    if (favs.length > 20) favs.length = 20;
  }
  saveFavorites(favs);
  return favs;
}

function isFavorite(logId: string, favs: FavoriteMeal[]): boolean {
  return favs.some((f) => f.id === logId);
}

// ── Food log localStorage persistence helper ──────────────────────────────────
// Ensures every food log is persisted locally as the primary store,
// regardless of whether the API call succeeds or fails.

function persistFoodLogLocally(userId: string, entry: FoodLog): void {
  try {
    const key = `ndw_food_logs_${userId}`;
    const existing: FoodLog[] = sbGetGeneric(key) ?? [];
    // Deduplicate: don't add if same id already exists
    if (existing.some((l) => l.id === entry.id)) return;
    existing.unshift(entry);
    // Keep max 200 entries to avoid localStorage quota issues
    if (existing.length > 200) existing.length = 200;
    sbSaveGeneric(key, existing);
  } catch { /* localStorage full or unavailable */ }
  // Also persist via Express API (primary server storage — no auth required)
  import("@/lib/queryClient").then(({ resolveUrl }) => {
    fetch(resolveUrl("/api/food/log"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        userId,
        mealType: entry.mealType ?? "meal",
        summary: entry.summary ?? (entry.foodItems ?? []).map((fi: any) => fi.name).join(", "),
        totalCalories: entry.totalCalories ?? 0,
        dominantMacro: entry.dominantMacro ?? null,
        foodItems: entry.foodItems ?? [],
      }),
    }).catch((e) => console.warn("[nutrition] Express API food/log failed:", e));
  }).catch(() => {});

  // Also try Supabase (may fail due to RLS if not authenticated)
  import("@/lib/supabase-browser").then(({ getSupabase }) =>
    getSupabase().then((sb) => {
      if (!sb) return;
      sb.from("food_logs").insert({
        user_id: userId,
        meal_type: entry.mealType ?? "meal",
        summary: entry.summary ?? (entry.foodItems ?? []).map((fi: any) => fi.name).join(", "),
        calories: entry.totalCalories ?? null,
        protein: (entry.foodItems ?? []).reduce((s: number, fi: any) => s + (fi.protein_g ?? 0), 0) || null,
        carbs: (entry.foodItems ?? []).reduce((s: number, fi: any) => s + (fi.carbs_g ?? 0), 0) || null,
        fat: (entry.foodItems ?? []).reduce((s: number, fi: any) => s + (fi.fat_g ?? 0), 0) || null,
        fiber: (entry.foodItems ?? []).reduce((s: number, fi: any) => s + (fi.fiber_g ?? 0), 0) || null,
        food_items: entry.foodItems ?? null,
        vitamins: entry.vitamins ?? null,
        food_quality_score: null,
        created_at: entry.loggedAt ?? new Date().toISOString(),
      }).then(({ error }: { error: { message: string } | null }) => {
        if (error) console.error("[nutrition] Supabase food_logs insert error:", error.message);
        else console.log("[nutrition] Supabase food_logs insert succeeded");
      });
    })
  ).catch((err) => console.error("[nutrition] Supabase food_logs insert failed:", err));
}

// ── Nutrition insights generator ──────────────────────────────────────────────

function generateInsights(
  todayLogs: FoodLog[],
  allLogs: FoodLog[] | undefined,
  totalProtein: number,
  totalCarbs: number,
  totalCalories: number
): string[] {
  const insights: string[] = [];

  if (todayLogs.length === 0 && (!allLogs || allLogs.length === 0)) {
    return ["Start logging meals to unlock personalized nutrition insights."];
  }

  // Check protein distribution across meals
  const mealsByType = new Map<string, number>();
  for (const log of todayLogs) {
    const mt = log.mealType ?? "snack";
    const protein = (log.foodItems ?? []).reduce((s, f) => s + (f.protein_g ?? 0), 0);
    mealsByType.set(mt, (mealsByType.get(mt) ?? 0) + protein);
  }
  const lunchProtein = mealsByType.get("lunch") ?? 0;
  const dinnerProtein = mealsByType.get("dinner") ?? 0;
  if (lunchProtein > 20 && lunchProtein > dinnerProtein) {
    insights.push("You tend to eat more protein at lunch -- great for afternoon focus.");
  }

  // Check vegetable/variety in recent meals
  if (allLogs && allLogs.length >= 3) {
    const recent = allLogs.slice(0, 5);
    const hasVeg = recent.filter((l) =>
      (l.foodItems ?? []).some((fi) =>
        /vegetable|salad|broccoli|spinach|carrot|kale|lettuce|greens/i.test(fi.name)
      )
    );
    if (hasVeg.length <= 1) {
      insights.push("Consider adding more vegetables -- only " + hasVeg.length + " of your last " + recent.length + " meals included them.");
    }
  }

  // Late night eating
  const lateNightMeals = todayLogs.filter((l) => {
    const h = new Date(l.loggedAt).getHours();
    return h >= 21 || h < 5;
  });
  if (lateNightMeals.length > 0) {
    insights.push("Late-night eating can affect sleep quality and morning recovery.");
  }

  // Protein too low
  if (totalProtein < 30 && todayLogs.length >= 2) {
    insights.push("Your protein intake is low today. Consider adding eggs, chicken, or beans to your next meal.");
  }

  // Carb-heavy day
  if (totalCarbs > 200 && totalProtein < totalCarbs * 0.3) {
    insights.push("Today is carb-heavy relative to protein. Adding protein can help sustain energy.");
  }

  // Return at most 2
  return insights.slice(0, 2);
}

// ── Local fallback food analysis (no API key needed) ──────────────────────────

interface LocalFoodAnalysis {
  food_items: FoodItem[];
  total_calories: number;
  dominant_macro: string;
  summary: string;
}

function estimateNutritionLocally(description: string): LocalFoodAnalysis {
  const lower = description.toLowerCase();
  const items = description.split(/[,\n]+/).map(s => s.trim()).filter(Boolean);

  const foodItems: FoodItem[] = items.map(name => {
    const n = name.toLowerCase();
    let calories = 200;
    let protein = 8;
    let carbs = 25;
    let fat = 8;

    // Protein-rich foods
    if (/chicken|turkey|poultry/.test(n)) { calories = 230; protein = 30; carbs = 0; fat = 10; }
    else if (/fish|salmon|tuna|shrimp|prawns/.test(n)) { calories = 200; protein = 28; carbs = 0; fat = 8; }
    else if (/beef|steak|lamb|pork/.test(n)) { calories = 280; protein = 26; carbs = 0; fat = 18; }
    else if (/egg/.test(n)) { calories = 155; protein = 13; carbs = 1; fat = 11; }
    else if (/tofu|tempeh/.test(n)) { calories = 145; protein = 15; carbs = 4; fat = 8; }
    else if (/paneer/.test(n)) { calories = 260; protein = 18; carbs = 4; fat = 20; }

    // Carb-rich foods
    else if (/rice/.test(n)) { calories = 210; protein = 4; carbs = 45; fat = 1; }
    else if (/bread|toast|naan|roti|chapati/.test(n)) { calories = 180; protein = 6; carbs = 34; fat = 3; }
    else if (/pasta|noodle|spaghetti/.test(n)) { calories = 220; protein = 8; carbs = 43; fat = 1; }
    else if (/potato|sweet potato/.test(n)) { calories = 160; protein = 4; carbs = 37; fat = 0; }
    else if (/oat|oatmeal|cereal|granola/.test(n)) { calories = 190; protein = 7; carbs = 32; fat = 5; }

    // Vegetables & salads
    else if (/salad|lettuce|greens/.test(n)) { calories = 50; protein = 2; carbs = 8; fat = 1; }
    else if (/vegetable|broccoli|spinach|carrot|cauliflower|zucchini|pepper|asparagus/.test(n)) { calories = 60; protein = 3; carbs = 10; fat = 1; }
    else if (/avocado/.test(n)) { calories = 160; protein = 2; carbs = 9; fat = 15; }

    // Fruits
    else if (/banana/.test(n)) { calories = 105; protein = 1; carbs = 27; fat = 0; }
    else if (/apple|orange|pear|peach|mango|berry|berries|strawberr/.test(n)) { calories = 80; protein = 1; carbs = 20; fat = 0; }

    // Dairy
    else if (/milk/.test(n)) { calories = 120; protein = 8; carbs = 12; fat = 5; }
    else if (/yogurt|curd|dahi/.test(n)) { calories = 100; protein = 10; carbs = 6; fat = 4; }
    else if (/cheese/.test(n)) { calories = 110; protein = 7; carbs = 1; fat = 9; }

    // Fats & oils
    else if (/butter|oil|ghee/.test(n)) { calories = 100; protein = 0; carbs = 0; fat = 11; }
    else if (/nut|almond|cashew|peanut|walnut/.test(n)) { calories = 170; protein = 6; carbs = 6; fat = 15; }

    // Snacks & drinks
    else if (/chips|fries/.test(n)) { calories = 250; protein = 3; carbs = 30; fat = 14; }
    else if (/pizza/.test(n)) { calories = 300; protein = 12; carbs = 36; fat = 12; }
    else if (/burger/.test(n)) { calories = 400; protein = 20; carbs = 35; fat = 20; }
    else if (/sandwich|wrap/.test(n)) { calories = 350; protein = 15; carbs = 40; fat = 14; }
    else if (/soup/.test(n)) { calories = 150; protein = 8; carbs = 18; fat = 5; }
    else if (/coffee|tea/.test(n)) { calories = 5; protein = 0; carbs = 1; fat = 0; }
    else if (/juice|smoothie/.test(n)) { calories = 120; protein = 1; carbs = 28; fat = 0; }
    else if (/soda|cola/.test(n)) { calories = 140; protein = 0; carbs = 39; fat = 0; }

    // Indian foods
    else if (/dal|daal|lentil/.test(n)) { calories = 180; protein = 12; carbs = 30; fat = 3; }
    else if (/biryani/.test(n)) { calories = 350; protein = 15; carbs = 45; fat = 12; }
    else if (/curry/.test(n)) { calories = 250; protein = 12; carbs = 15; fat = 16; }
    else if (/dosa|idli|sambar/.test(n)) { calories = 160; protein = 5; carbs = 28; fat = 4; }
    else if (/samosa|pakora/.test(n)) { calories = 250; protein = 5; carbs = 25; fat = 15; }

    return {
      name: name.charAt(0).toUpperCase() + name.slice(1),
      portion: "1 serving",
      calories,
      protein_g: protein,
      carbs_g: carbs,
      fat_g: fat,
    };
  });

  const totalCalories = foodItems.reduce((s, fi) => s + fi.calories, 0);
  const totalProtein = foodItems.reduce((s, fi) => s + fi.protein_g, 0);
  const totalCarbs = foodItems.reduce((s, fi) => s + fi.carbs_g, 0);
  const totalFat = foodItems.reduce((s, fi) => s + fi.fat_g, 0);

  const dominant =
    totalProtein >= totalCarbs && totalProtein >= totalFat ? "protein"
    : totalCarbs >= totalFat ? "carbs"
    : "fat";

  return {
    food_items: foodItems,
    total_calories: totalCalories,
    dominant_macro: dominant,
    summary: items.map(i => i.charAt(0).toUpperCase() + i.slice(1)).join(", "),
  };
}

// ── Supplement Tracking ─────────────────────────────────────────────────────

interface Supplement {
  id: string;
  name: string;
  dosage: string;
  timeOfDay: string;
  type: "vitamin" | "mineral" | "glp1" | "other";
}

interface SupplementDailyLog {
  [supplementId: string]: boolean;
}

// ── GLP-1 Injection Tracking ──────────────────────────────────────────────────

interface Glp1Injection {
  id: string;
  medication: string;
  dose: string;
  date: string; // ISO date string
  site?: string;
}

const GLP1_MEDICATIONS = [
  { name: "Ozempic", defaultDose: "0.5 mg", halfLifeDays: 7, cadenceDays: 7 },
  { name: "Wegovy", defaultDose: "1.7 mg", halfLifeDays: 7, cadenceDays: 7 },
  { name: "Mounjaro", defaultDose: "5 mg", halfLifeDays: 5, cadenceDays: 7 },
  { name: "Zepbound", defaultDose: "5 mg", halfLifeDays: 5, cadenceDays: 7 },
  { name: "Saxenda", defaultDose: "1.8 mg", halfLifeDays: 0.54, cadenceDays: 1 },
];

const GLP1_INJECTIONS_KEY = "ndw_glp1_injections";

function loadGlp1Injections(): Glp1Injection[] {
  try {
    const raw = sbGetSetting(GLP1_INJECTIONS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

function saveGlp1Injections(injections: Glp1Injection[]) {
  try { sbSaveGeneric(GLP1_INJECTIONS_KEY, injections); } catch {}
}

function getNextInjectionDate(injections: Glp1Injection[], medication: string): Date | null {
  const med = GLP1_MEDICATIONS.find(m => m.name === medication);
  if (!med) return null;
  const medInjections = injections
    .filter(i => i.medication === medication)
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
  if (medInjections.length === 0) return null;
  const lastDate = new Date(medInjections[0].date);
  const next = new Date(lastDate);
  next.setDate(next.getDate() + med.cadenceDays);
  return next;
}

function computeDecayLevel(injections: Glp1Injection[], medication: string): number {
  const med = GLP1_MEDICATIONS.find(m => m.name === medication);
  if (!med || med.halfLifeDays <= 0) return 0;
  const medInjections = injections.filter(i => i.medication === medication);
  if (medInjections.length === 0) return 0;
  const last = medInjections.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())[0];
  const daysSince = (Date.now() - new Date(last.date).getTime()) / (1000 * 60 * 60 * 24);
  return Math.max(0, Math.pow(0.5, daysSince / med.halfLifeDays));
}

// ── Supplement Micronutrient Mapping ──────────────────────────────────────────

function getSupplementVitamins(supplement: Supplement): VitaminData {
  const v: VitaminData = {
    vitamin_d_mcg: 0, vitamin_b12_mcg: 0, vitamin_c_mg: 0,
    iron_mg: 0, magnesium_mg: 0, zinc_mg: 0, omega3_g: 0,
  };
  const n = supplement.name.toLowerCase();
  const dosageNum = parseFloat(supplement.dosage) || 0;

  if (/vitamin d/i.test(n)) v.vitamin_d_mcg = dosageNum > 0 ? (dosageNum <= 100 ? dosageNum : dosageNum * 0.025) : 50; // 2000 IU = 50 mcg
  if (/b12|vitamin b12/i.test(n)) v.vitamin_b12_mcg = dosageNum > 0 ? dosageNum : 1000;
  if (/vitamin c/i.test(n)) v.vitamin_c_mg = dosageNum > 0 ? dosageNum : 500;
  if (/^iron/i.test(n)) v.iron_mg = dosageNum > 0 ? dosageNum : 18;
  if (/magnesium/i.test(n)) v.magnesium_mg = dosageNum > 0 ? dosageNum : 400;
  if (/^zinc/i.test(n)) v.zinc_mg = dosageNum > 0 ? dosageNum : 15;
  if (/omega.?3|fish oil/i.test(n)) v.omega3_g = dosageNum > 0 ? dosageNum / 1000 : 1;
  if (/calcium/i.test(n)) { /* calcium not in our VitaminData yet */ }
  if (/multivitamin/i.test(n)) {
    v.vitamin_d_mcg += 10; v.vitamin_b12_mcg += 2.4; v.vitamin_c_mg += 60;
    v.iron_mg += 8; v.magnesium_mg += 50; v.zinc_mg += 5; v.omega3_g += 0;
  }

  return v;
}

// ── Food Quality Score Calculation ─────────────────────────────────────────────

const FOOD_QUALITY_KEY = "ndw_food_quality_score";

function calculateFoodQualityScore(
  todayLogs: FoodLog[],
  totalProtein: number,
  totalCarbs: number,
  totalFat: number,
  totalCalories: number,
): number {
  if (todayLogs.length === 0) return 0;

  let score = 50; // base score

  const allItems = todayLogs.flatMap(l => l.foodItems ?? []);

  // Protein adequacy (+15 max)
  if (totalProtein >= 50) score += 15;
  else if (totalProtein >= 30) score += 10;
  else if (totalProtein >= 15) score += 5;

  // Vegetable presence (+10)
  const hasVeg = allItems.some(fi =>
    /vegetable|salad|broccoli|spinach|carrot|kale|lettuce|greens|zucchini|pepper|asparagus|cauliflower/i.test(fi.name)
  );
  if (hasVeg) score += 10;

  // Fruit presence (+5)
  const hasFruit = allItems.some(fi =>
    /fruit|apple|banana|orange|berry|berries|mango|grape|kiwi|pear/i.test(fi.name)
  );
  if (hasFruit) score += 5;

  // Whole grains (+5)
  const hasWholeGrain = allItems.some(fi =>
    /whole grain|oat|quinoa|brown rice|whole wheat/i.test(fi.name)
  );
  if (hasWholeGrain) score += 5;

  // Meal regularity (+5 for 3+ meals)
  if (todayLogs.length >= 3) score += 5;

  // Balanced macros (+10 if no single macro dominates >60%)
  if (totalCalories > 0) {
    const proteinPct = (totalProtein * 4) / totalCalories;
    const carbPct = (totalCarbs * 4) / totalCalories;
    const fatPct = (totalFat * 9) / totalCalories;
    if (proteinPct <= 0.6 && carbPct <= 0.6 && fatPct <= 0.6) score += 10;
  }

  // Penalties
  const hasProcessed = allItems.some(fi =>
    /chips|fries|candy|soda|pizza|burger|hot dog|donut|cookie/i.test(fi.name)
  );
  if (hasProcessed) score -= 10;

  // Calorie overshoot penalty
  if (totalCalories > CAL_GOAL * 1.3) score -= 10;

  // Clamp 0-100
  score = Math.max(0, Math.min(100, score));

  // Persist to localStorage
  try { sbSaveGeneric(FOOD_QUALITY_KEY, { score, date: new Date().toISOString().slice(0, 10) }); } catch {}

  return score;
}

const SUPPLEMENT_PRESETS = [
  { name: "Vitamin D", dosage: "2000 IU", type: "vitamin" as const },
  { name: "Vitamin B12", dosage: "1000 mcg", type: "vitamin" as const },
  { name: "Fish Oil / Omega-3", dosage: "1000 mg", type: "other" as const },
  { name: "Magnesium", dosage: "400 mg", type: "mineral" as const },
  { name: "Iron", dosage: "18 mg", type: "mineral" as const },
  { name: "Zinc", dosage: "15 mg", type: "mineral" as const },
  { name: "Vitamin C", dosage: "500 mg", type: "vitamin" as const },
  { name: "Calcium", dosage: "500 mg", type: "mineral" as const },
  { name: "Probiotics", dosage: "1 capsule", type: "other" as const },
  { name: "Ozempic", dosage: "0.5 mg", type: "glp1" as const },
  { name: "Wegovy", dosage: "1.7 mg", type: "glp1" as const },
  { name: "Mounjaro", dosage: "5 mg", type: "glp1" as const },
];

const SUPPLEMENTS_KEY = "ndw_supplements";
const SUPPLEMENT_LOG_KEY = "ndw_supplement_log";

function loadSupplements(): Supplement[] {
  try {
    const raw = sbGetSetting(SUPPLEMENTS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

function saveSupplements(supps: Supplement[]) {
  try { sbSaveGeneric(SUPPLEMENTS_KEY, supps); } catch {}
}

function getTodayLogKey(): string {
  return `${SUPPLEMENT_LOG_KEY}_${new Date().toISOString().slice(0, 10)}`;
}

function loadTodayLog(): SupplementDailyLog {
  try {
    const raw = sbGetSetting(getTodayLogKey());
    return raw ? JSON.parse(raw) : {};
  } catch { return {}; }
}

function saveTodayLog(log: SupplementDailyLog) {
  try { sbSaveGeneric(getTodayLogKey(), log); } catch {}
}

function SupplementTracker({ onSupplementsChange }: { onSupplementsChange?: (supplements: Supplement[], dailyLog: SupplementDailyLog) => void }) {
  const [supplements, setSupplements] = useState<Supplement[]>(loadSupplements);
  const [dailyLog, setDailyLog] = useState<SupplementDailyLog>(loadTodayLog);
  const [showAdd, setShowAdd] = useState(false);
  const [customName, setCustomName] = useState("");
  const [customDosage, setCustomDosage] = useState("");

  const toggleTaken = (id: string) => {
    const updated = { ...dailyLog, [id]: !dailyLog[id] };
    setDailyLog(updated);
    saveTodayLog(updated);
    onSupplementsChange?.(supplements, updated);
  };

  const addSupplement = (name: string, dosage: string, type: Supplement["type"]) => {
    const newSupp: Supplement = {
      id: `supp_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      name, dosage, timeOfDay: "morning", type,
    };
    const updated = [...supplements, newSupp];
    setSupplements(updated);
    saveSupplements(updated);
    setCustomName("");
    setCustomDosage("");
    setShowAdd(false);
    onSupplementsChange?.(updated, dailyLog);
  };

  const removeSupplement = (id: string) => {
    const updated = supplements.filter(s => s.id !== id);
    setSupplements(updated);
    saveSupplements(updated);
    onSupplementsChange?.(updated, dailyLog);
  };

  const takenCount = supplements.filter(s => dailyLog[s.id]).length;
  const totalCount = supplements.length;

  return (
    <div>
      <div style={{
        fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
        textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 10,
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <span>Supplements</span>
        {totalCount > 0 && (
          <span style={{ fontSize: 12, fontWeight: 700, color: takenCount === totalCount ? "#06b6d4" : "#d4a017" }}>
            {takenCount}/{totalCount}
          </span>
        )}
      </div>

      {/* Supplement list */}
      {supplements.length > 0 ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 10 }}>
          {supplements.map(supp => {
            const taken = !!dailyLog[supp.id];
            const isGlp1 = supp.type === "glp1";
            return (
              <div key={supp.id} style={{
                display: "flex", alignItems: "center", gap: 8,
                padding: "6px 8px", borderRadius: 8,
                background: taken ? "rgba(34, 197, 94, 0.08)" : "transparent",
                transition: "background 0.2s",
              }}>
                <button
                  onClick={() => toggleTaken(supp.id)}
                  style={{
                    width: 22, height: 22, borderRadius: 6, flexShrink: 0,
                    border: taken ? "2px solid #06b6d4" : "2px solid var(--border)",
                    background: taken ? "#06b6d4" : "transparent",
                    cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
                    color: "white", fontSize: 12, fontWeight: 700,
                  }}
                >
                  {taken ? "\u2713" : ""}
                </button>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    fontSize: 12, fontWeight: 500, color: taken ? "var(--muted-foreground)" : "var(--foreground)",
                    textDecoration: taken ? "line-through" : "none",
                  }}>
                    {supp.name}
                    {isGlp1 && (
                      <span style={{ fontSize: 9, color: "#7c3aed", marginLeft: 6, fontWeight: 600, textDecoration: "none" }}>GLP-1</span>
                    )}
                  </div>
                  <div style={{ fontSize: 10, color: "var(--muted-foreground)" }}>{supp.dosage}</div>
                </div>
                <button
                  onClick={() => removeSupplement(supp.id)}
                  style={{
                    background: "none", border: "none", cursor: "pointer",
                    color: "var(--muted-foreground)", fontSize: 14, padding: 4, flexShrink: 0,
                  }}
                >
                  x
                </button>
              </div>
            );
          })}
        </div>
      ) : (
        <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginBottom: 10 }}>
          Add supplements you take daily to track adherence
        </div>
      )}

      {/* Add supplement UI */}
      {showAdd ? (
        <div style={{ marginBottom: 8 }}>
          {/* Presets */}
          <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginBottom: 6, fontWeight: 600 }}>
            Quick add:
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 8 }}>
            {SUPPLEMENT_PRESETS.filter(p => !supplements.some(s => s.name === p.name)).slice(0, 8).map(preset => (
              <button
                key={preset.name}
                onClick={() => addSupplement(preset.name, preset.dosage, preset.type)}
                style={{
                  fontSize: 10, padding: "4px 10px", borderRadius: 8,
                  background: preset.type === "glp1" ? "rgba(124, 58, 237, 0.12)" : "var(--muted)",
                  color: preset.type === "glp1" ? "#7c3aed" : "var(--foreground)",
                  border: "1px solid var(--border)", cursor: "pointer", fontWeight: 500,
                }}
              >
                + {preset.name}
              </button>
            ))}
          </div>
          {/* Custom */}
          <div style={{ display: "flex", gap: 6 }}>
            <input
              value={customName}
              onChange={e => setCustomName(e.target.value)}
              placeholder="Name"
              style={{
                flex: 2, background: "var(--background)", color: "var(--foreground)",
                border: "1px solid var(--border)", borderRadius: 8, padding: "6px 10px",
                fontSize: 12, outline: "none",
              }}
            />
            <input
              value={customDosage}
              onChange={e => setCustomDosage(e.target.value)}
              placeholder="Dosage"
              style={{
                flex: 1, background: "var(--background)", color: "var(--foreground)",
                border: "1px solid var(--border)", borderRadius: 8, padding: "6px 10px",
                fontSize: 12, outline: "none",
              }}
            />
            <button
              disabled={!customName.trim()}
              onClick={() => addSupplement(customName.trim(), customDosage.trim() || "1 dose", "other")}
              style={{
                background: customName.trim() ? "var(--primary)" : "var(--muted)",
                color: customName.trim() ? "white" : "var(--muted-foreground)",
                borderRadius: 8, padding: "6px 12px", fontSize: 11, fontWeight: 600,
                border: "none", cursor: "pointer",
              }}
            >
              Add
            </button>
          </div>
          <button
            onClick={() => setShowAdd(false)}
            style={{
              width: "100%", marginTop: 6, background: "none", border: "none",
              color: "var(--muted-foreground)", fontSize: 11, cursor: "pointer",
            }}
          >
            Cancel
          </button>
        </div>
      ) : (
        <button
          onClick={() => setShowAdd(true)}
          style={{
            width: "100%", background: "var(--muted)", color: "var(--foreground)",
            borderRadius: 10, padding: 8, fontSize: 12, fontWeight: 500,
            border: "1px solid var(--border)", cursor: "pointer",
          }}
        >
          + Add Supplement
        </button>
      )}
    </div>
  );
}

// ── Calorie Ring (SVG) — compact version for sticky header ──────────────────

const CAL_GOAL = 2000;
const PROTEIN_GOAL = 50;
const CARBS_GOAL = 275;
const FAT_GOAL = 78;

function CalorieRingCompact({ calories }: { calories: number }) {
  const size = 100;
  const r = 40;
  const stroke = 7;
  const circ = 2 * Math.PI * r;
  const pct = Math.min(calories / CAL_GOAL, 1);
  const dash = circ * pct;
  const cx = size / 2;

  const remaining = CAL_GOAL - calories;
  const isOver = remaining < 0;

  return (
    <div style={{ position: "relative", width: size, height: size, flexShrink: 0 }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={cx} cy={cx} r={r} fill="none" stroke="var(--border)" strokeWidth={stroke} />
        <circle
          cx={cx} cy={cx} r={r} fill="none"
          stroke="url(#calGradCompact)" strokeWidth={stroke}
          strokeDasharray={`${dash} ${circ}`} strokeLinecap="round"
        />
        <defs>
          <linearGradient id="calGradCompact" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#d4a017" />
            <stop offset="100%" stopColor="#ea580c" />
          </linearGradient>
        </defs>
      </svg>
      <div style={{
        position: "absolute", inset: 0,
        display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
      }}>
        <span style={{ fontSize: 28, fontWeight: 700, color: "var(--foreground)", lineHeight: 1 }}>
          {calories}
        </span>
        <span style={{ fontSize: 9, color: "var(--muted-foreground)", marginTop: 1 }}>
          {isOver ? "over" : "kcal"}
        </span>
      </div>
    </div>
  );
}

// ── Macro Progress Bar ────────────────────────────────────────────────────────

function MacroBar({ value, goal, color }: { value: number; goal: number; color: string }) {
  const pct = Math.min((value / goal) * 100, 100);
  return (
    <div style={{ height: 8, background: "var(--border)", borderRadius: 4, overflow: "hidden", flex: 1 }}>
      <div style={{
        width: `${pct}%`, height: "100%", background: color,
        borderRadius: 2, transition: "width 0.4s ease",
      }} />
    </div>
  );
}

// ── Vitamin Tracker ──────────────────────────────────────────────────────────

interface VitaminTarget {
  key: keyof VitaminData;
  label: string;
  unit: string;
  daily: number;
}

const VITAMIN_TARGETS: VitaminTarget[] = [
  { key: "vitamin_d_mcg", label: "Vitamin D", unit: "mcg", daily: 15 },
  { key: "vitamin_b12_mcg", label: "B12", unit: "mcg", daily: 2.4 },
  { key: "vitamin_c_mg", label: "Vitamin C", unit: "mg", daily: 90 },
  { key: "iron_mg", label: "Iron", unit: "mg", daily: 18 },
  { key: "magnesium_mg", label: "Magnesium", unit: "mg", daily: 400 },
  { key: "zinc_mg", label: "Zinc", unit: "mg", daily: 11 },
  { key: "omega3_g", label: "Omega-3", unit: "g", daily: 1.6 },
];

/** Estimate micronutrients from food item names when GPT vitamins data is unavailable. */
function estimateVitaminsFromFood(items: FoodItem[]): VitaminData {
  const v: VitaminData = {
    vitamin_d_mcg: 0, vitamin_b12_mcg: 0, vitamin_c_mg: 0,
    iron_mg: 0, magnesium_mg: 0, zinc_mg: 0, omega3_g: 0,
  };
  for (const item of items) {
    const n = item.name.toLowerCase();
    // Vitamin D sources
    if (/salmon|tuna|sardine|mackerel|trout|herring/i.test(n)) v.vitamin_d_mcg += 10;
    if (/egg/i.test(n)) v.vitamin_d_mcg += 1;
    if (/milk|yogurt|cheese/i.test(n)) v.vitamin_d_mcg += 2.5;
    // B12 sources
    if (/beef|steak|lamb/i.test(n)) v.vitamin_b12_mcg += 2.5;
    if (/chicken|turkey|poultry/i.test(n)) v.vitamin_b12_mcg += 0.3;
    if (/salmon|tuna|fish|sardine/i.test(n)) v.vitamin_b12_mcg += 4;
    if (/egg/i.test(n)) v.vitamin_b12_mcg += 0.6;
    if (/milk|yogurt/i.test(n)) v.vitamin_b12_mcg += 1;
    // Vitamin C sources
    if (/orange|lemon|lime|grapefruit|citrus/i.test(n)) v.vitamin_c_mg += 50;
    if (/strawberry|kiwi|mango|papaya|pineapple/i.test(n)) v.vitamin_c_mg += 40;
    if (/broccoli|pepper|tomato|cauliflower|brussels/i.test(n)) v.vitamin_c_mg += 40;
    if (/spinach|kale|cabbage/i.test(n)) v.vitamin_c_mg += 20;
    // Iron sources
    if (/beef|steak|lamb/i.test(n)) v.iron_mg += 3;
    if (/spinach|kale|lentil|bean|chickpea/i.test(n)) v.iron_mg += 3;
    if (/tofu|tempeh/i.test(n)) v.iron_mg += 3;
    // Magnesium sources
    if (/almond|cashew|peanut|nut/i.test(n)) v.magnesium_mg += 50;
    if (/spinach|avocado|banana/i.test(n)) v.magnesium_mg += 30;
    if (/dark choc|cacao/i.test(n)) v.magnesium_mg += 60;
    if (/quinoa|oat|brown rice|whole grain/i.test(n)) v.magnesium_mg += 40;
    // Zinc sources
    if (/beef|steak|lamb/i.test(n)) v.zinc_mg += 4;
    if (/oyster|crab|lobster|shrimp/i.test(n)) v.zinc_mg += 6;
    if (/chicken|turkey/i.test(n)) v.zinc_mg += 2;
    if (/chickpea|lentil|bean/i.test(n)) v.zinc_mg += 1.5;
    // Omega-3 sources
    if (/salmon|sardine|mackerel|herring|anchov/i.test(n)) v.omega3_g += 1.5;
    if (/tuna|trout/i.test(n)) v.omega3_g += 0.8;
    if (/walnut|flax|chia/i.test(n)) v.omega3_g += 0.5;
    if (/avocado/i.test(n)) v.omega3_g += 0.1;
  }
  return v;
}

function VitaminBar({ value, goal, color }: { value: number; goal: number; color: string }) {
  const pct = Math.min((value / goal) * 100, 100);
  return (
    <div style={{ height: 8, background: "var(--border)", borderRadius: 4, overflow: "hidden", flex: 1 }}>
      <div style={{
        height: "100%", width: `${pct}%`, background: color,
        borderRadius: 2, transition: "width 0.5s ease",
      }} />
    </div>
  );
}

function VitaminTracker({ todayLogs, supplementVitamins }: { todayLogs: FoodLog[]; supplementVitamins?: VitaminData | null }) {
  const vitaminTotals = useMemo(() => {
    const totals: VitaminData = {
      vitamin_d_mcg: 0, vitamin_b12_mcg: 0, vitamin_c_mg: 0,
      iron_mg: 0, magnesium_mg: 0, zinc_mg: 0, omega3_g: 0,
    };
    // Add food-sourced vitamins
    for (const log of todayLogs) {
      if (log.vitamins) {
        for (const k of Object.keys(totals) as (keyof VitaminData)[]) {
          totals[k] += log.vitamins[k] ?? 0;
        }
      } else if (log.foodItems) {
        const est = estimateVitaminsFromFood(log.foodItems);
        for (const k of Object.keys(totals) as (keyof VitaminData)[]) {
          totals[k] += est[k];
        }
      }
    }
    // Add supplement-sourced vitamins
    if (supplementVitamins) {
      for (const k of Object.keys(totals) as (keyof VitaminData)[]) {
        totals[k] += supplementVitamins[k] ?? 0;
      }
    }
    return totals;
  }, [todayLogs, supplementVitamins]);

  const hasFoodSources = todayLogs.length > 0;
  const hasSupplementSources = supplementVitamins != null && Object.values(supplementVitamins).some(v => v > 0);

  return (
    <div>
      <div style={{
        fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
        textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 10,
      }}>
        Micronutrients
      </div>
      {/* Source indicators */}
      <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
        <span style={{
          fontSize: 9, padding: "2px 8px", borderRadius: 8,
          background: hasFoodSources ? "rgba(6, 182, 212, 0.12)" : "rgba(148, 163, 184, 0.1)",
          color: hasFoodSources ? "#06b6d4" : "var(--muted-foreground)",
          fontWeight: 600,
        }}>
          Food {hasFoodSources ? `(${todayLogs.length} meals)` : "(none)"}
        </span>
        <span style={{
          fontSize: 9, padding: "2px 8px", borderRadius: 8,
          background: hasSupplementSources ? "rgba(124, 58, 237, 0.12)" : "rgba(148, 163, 184, 0.1)",
          color: hasSupplementSources ? "#7c3aed" : "var(--muted-foreground)",
          fontWeight: 600,
        }}>
          Supplements {hasSupplementSources ? "(active)" : "(none taken)"}
        </span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {VITAMIN_TARGETS.map((vt) => {
          const val = vitaminTotals[vt.key];
          const pct = vt.daily > 0 ? (val / vt.daily) * 100 : 0;
          const color = pct >= 80 ? "#06b6d4" : pct >= 50 ? "#d4a017" : "#e879a8";
          return (
            <div key={vt.key} style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ width: 60, fontSize: 11, fontWeight: 500, color: "var(--foreground)" }}>
                {vt.label}
              </div>
              <VitaminBar value={val} goal={vt.daily} color={color} />
              <div style={{ width: 56, fontSize: 10, textAlign: "right", color: "var(--muted-foreground)" }}>
                {val > 0 ? `${Math.round(val * 10) / 10}${vt.unit}` : "--"}
              </div>
            </div>
          );
        })}
      </div>
      {!hasFoodSources && !hasSupplementSources && (
        <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 8 }}>
          Log meals or take supplements to track micronutrients
        </div>
      )}
      {!hasSupplementSources && hasFoodSources && (
        <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 8 }}>
          No supplements taken today -- check the Supplements tab to log yours
        </div>
      )}
    </div>
  );
}

// ── GLP-1 Support Tracker ────────────────────────────────────────────────────

const GLP1_FOODS: Record<string, string[]> = {
  protein: ["chicken", "turkey", "fish", "salmon", "tuna", "egg", "beef", "steak", "tofu", "tempeh", "shrimp", "pork", "lamb", "whey", "protein", "greek yogurt"],
  fiber: ["oat", "quinoa", "lentil", "bean", "chickpea", "broccoli", "brussels", "artichoke", "pea", "apple", "berry", "chia", "flax", "whole grain", "brown rice", "avocado", "sweet potato"],
  healthyFat: ["avocado", "almond", "walnut", "cashew", "olive", "nut butter", "peanut butter", "coconut oil", "flax", "chia", "salmon", "sardine", "mackerel"],
  fermented: ["yogurt", "kefir", "kimchi", "sauerkraut", "miso", "tempeh", "kombucha", "pickle"],
};

function computeGlp1Score(
  todayLogs: FoodLog[],
  totalProtein: number,
  totalCalories: number,
): { score: number; foods: string[]; breakdown: { protein: number; fiber: number; healthyFat: number; fermented: number } } {
  const foods: string[] = [];
  let fiberCount = 0;
  let healthyFatCount = 0;
  let fermentedCount = 0;

  for (const log of todayLogs) {
    for (const item of log.foodItems ?? []) {
      const n = item.name.toLowerCase();
      for (const [cat, keywords] of Object.entries(GLP1_FOODS)) {
        if (keywords.some((kw) => n.includes(kw))) {
          if (!foods.includes(item.name)) foods.push(item.name);
          if (cat === "fiber") fiberCount++;
          if (cat === "healthyFat") healthyFatCount++;
          if (cat === "fermented") fermentedCount++;
        }
      }
    }
  }

  const proteinPct = totalCalories > 0 ? (totalProtein * 4 / totalCalories) * 100 : 0;
  const proteinScore = Math.min(25, (proteinPct / 25) * 25);
  const fiberScore = Math.min(25, (fiberCount / 3) * 25);
  const fatScore = Math.min(25, (healthyFatCount / 2) * 25);
  const fermentedScore = Math.min(25, fermentedCount > 0 ? 25 : 0);

  const score = Math.round(proteinScore + fiberScore + fatScore + fermentedScore);

  return {
    score,
    foods,
    breakdown: {
      protein: Math.round(proteinScore),
      fiber: Math.round(fiberScore),
      healthyFat: Math.round(fatScore),
      fermented: Math.round(fermentedScore),
    },
  };
}

function Glp1Tracker({
  todayLogs,
  totalProtein,
  totalCalories,
}: {
  todayLogs: FoodLog[];
  totalProtein: number;
  totalCalories: number;
}) {
  const glp1 = useMemo(
    () => computeGlp1Score(todayLogs, totalProtein, totalCalories),
    [todayLogs, totalProtein, totalCalories],
  );

  const scoreColor = glp1.score >= 70 ? "#06b6d4" : glp1.score >= 40 ? "#0891b2" : "#94a3b8";

  return (
    <div style={{ marginTop: 16 }}>
      <div style={{
        fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
        textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 10,
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <span>GLP-1 Support</span>
        <span style={{
          fontSize: 28, fontWeight: 700, color: scoreColor,
          background: `linear-gradient(135deg, ${scoreColor}, #0891b2)`,
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
        }}>
          {glp1.score}
        </span>
      </div>

      {/* Score breakdown */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginBottom: 10 }}>
        {([
          { label: "Protein", value: glp1.breakdown.protein, max: 25 },
          { label: "Fiber", value: glp1.breakdown.fiber, max: 25 },
          { label: "Healthy Fats", value: glp1.breakdown.healthyFat, max: 25 },
          { label: "Fermented", value: glp1.breakdown.fermented, max: 25 },
        ] as const).map((item) => (
          <div key={item.label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ fontSize: 10, color: "var(--muted-foreground)", width: 68, flexShrink: 0 }}>
              {item.label}
            </div>
            <div style={{ flex: 1, height: 8, background: "var(--border)", borderRadius: 4, overflow: "hidden" }}>
              <div style={{
                height: "100%",
                width: `${(item.value / item.max) * 100}%`,
                background: "linear-gradient(90deg, #0891b2, #22d3ee)",
                borderRadius: 2,
                transition: "width 0.5s ease",
              }} />
            </div>
          </div>
        ))}
      </div>

      {/* Contributing foods */}
      {glp1.foods.length > 0 ? (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
          {glp1.foods.slice(0, 6).map((f, i) => (
            <span
              key={i}
              style={{
                fontSize: 10, padding: "2px 8px", borderRadius: 10,
                background: "hsl(180 65% 50% / 0.12)", color: "#0891b2", fontWeight: 500,
              }}
            >
              {f}
            </span>
          ))}
        </div>
      ) : (
        <div style={{ fontSize: 11, color: "var(--muted-foreground)" }}>
          Log meals rich in protein, fiber, and healthy fats to boost GLP-1 support
        </div>
      )}
    </div>
  );
}

// ── GLP-1 Injection Tracker Component ───────────────────────────────────────

function Glp1InjectionTracker() {
  const [injections, setInjections] = useState<Glp1Injection[]>(loadGlp1Injections);
  const [showAdd, setShowAdd] = useState(false);
  const [selectedMed, setSelectedMed] = useState(GLP1_MEDICATIONS[0].name);
  const [customDose, setCustomDose] = useState("");

  const addInjection = () => {
    const med = GLP1_MEDICATIONS.find(m => m.name === selectedMed);
    const newInj: Glp1Injection = {
      id: `glp1_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      medication: selectedMed,
      dose: customDose.trim() || med?.defaultDose || "0.5 mg",
      date: new Date().toISOString(),
    };
    const updated = [newInj, ...injections];
    setInjections(updated);
    saveGlp1Injections(updated);
    // Also persist to Supabase (fire-and-forget, skip localStorage since saveGlp1Injections handles it)
    import("@/lib/supabase-browser").then(({ getSupabase }) =>
      getSupabase().then((sb) => {
        if (!sb) return;
        sb.from("glp1_injections").insert({
          user_id: "local",
          medication: newInj.medication,
          dose: parseFloat(newInj.dose) || null,
          injected_at: newInj.date,
        });
      })
    ).catch(() => {});
    setShowAdd(false);
    setCustomDose("");
  };

  const removeInjection = (id: string) => {
    const updated = injections.filter(i => i.id !== id);
    setInjections(updated);
    saveGlp1Injections(updated);
  };

  // Get unique medications used
  const activeMeds = [...new Set(injections.map(i => i.medication))];

  return (
    <div style={{ marginTop: 16 }}>
      <div style={{
        fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
        textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 10,
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <span>GLP-1 Injection Tracker</span>
        <span style={{ fontSize: 10, color: "#7c3aed", fontWeight: 700 }}>
          {injections.length} logged
        </span>
      </div>

      {/* Active medication decay + next injection */}
      {activeMeds.map(medName => {
        const med = GLP1_MEDICATIONS.find(m => m.name === medName);
        if (!med) return null;
        const decay = computeDecayLevel(injections, medName);
        const nextDate = getNextInjectionDate(injections, medName);
        const isOverdue = nextDate ? nextDate.getTime() < Date.now() : false;
        const daysUntil = nextDate ? Math.ceil((nextDate.getTime() - Date.now()) / (1000 * 60 * 60 * 24)) : null;
        const decayPct = Math.round(decay * 100);
        const decayColor = decayPct >= 50 ? "#06b6d4" : decayPct >= 25 ? "#d4a017" : "#e879a8";

        return (
          <div key={medName} style={{
            background: "var(--muted)", borderRadius: 12, padding: 12, marginBottom: 8,
          }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
              <div>
                <span style={{ fontSize: 13, fontWeight: 600, color: "var(--foreground)" }}>{medName}</span>
                <span style={{ fontSize: 9, color: "#7c3aed", marginLeft: 6, fontWeight: 600 }}>GLP-1</span>
              </div>
              <span style={{ fontSize: 18, fontWeight: 700, color: decayColor }}>{decayPct}%</span>
            </div>

            {/* Decay visualization */}
            <div style={{ marginBottom: 8 }}>
              <div style={{ fontSize: 9, color: "var(--muted-foreground)", marginBottom: 4 }}>
                Estimated active level (half-life ~{med.halfLifeDays}d)
              </div>
              <div style={{ height: 8, background: "var(--border)", borderRadius: 4, overflow: "hidden" }}>
                <div style={{
                  width: `${decayPct}%`, height: "100%",
                  background: `linear-gradient(90deg, ${decayColor}, ${decayColor}88)`,
                  borderRadius: 4, transition: "width 0.5s ease",
                }} />
              </div>
            </div>

            {/* Next injection reminder */}
            {nextDate && (
              <div style={{
                fontSize: 11, fontWeight: 500,
                color: isOverdue ? "#e879a8" : daysUntil != null && daysUntil <= 1 ? "#d4a017" : "var(--muted-foreground)",
                padding: "6px 8px", borderRadius: 8,
                background: isOverdue ? "rgba(232, 121, 168, 0.08)" : "transparent",
              }}>
                {isOverdue
                  ? `Overdue -- was due ${nextDate.toLocaleDateString()}`
                  : daysUntil === 0
                    ? "Due today"
                    : daysUntil === 1
                      ? "Due tomorrow"
                      : `Next injection: ${nextDate.toLocaleDateString()} (${daysUntil} days)`}
              </div>
            )}
          </div>
        );
      })}

      {/* Recent injections */}
      {injections.length > 0 && (
        <div style={{ marginBottom: 10 }}>
          <div style={{ fontSize: 10, color: "var(--muted-foreground)", fontWeight: 600, marginBottom: 6 }}>
            Recent injections
          </div>
          {injections.slice(0, 5).map(inj => (
            <div key={inj.id} style={{
              display: "flex", alignItems: "center", gap: 8, padding: "4px 0",
              borderBottom: "1px solid var(--border)",
            }}>
              <span style={{ fontSize: 11, fontWeight: 500, color: "var(--foreground)", flex: 1 }}>
                {inj.medication} -- {inj.dose}
              </span>
              <span style={{ fontSize: 10, color: "var(--muted-foreground)" }}>
                {new Date(inj.date).toLocaleDateString()}
              </span>
              <button
                onClick={() => removeInjection(inj.id)}
                style={{
                  background: "none", border: "none", cursor: "pointer",
                  color: "var(--muted-foreground)", fontSize: 12, padding: 2,
                }}
              >
                x
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Add injection */}
      {showAdd ? (
        <div style={{ marginBottom: 8 }}>
          <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
            <select
              value={selectedMed}
              onChange={e => setSelectedMed(e.target.value)}
              style={{
                flex: 2, background: "var(--background)", color: "var(--foreground)",
                border: "1px solid var(--border)", borderRadius: 8, padding: "6px 10px",
                fontSize: 12, outline: "none",
              }}
            >
              {GLP1_MEDICATIONS.map(m => (
                <option key={m.name} value={m.name}>{m.name}</option>
              ))}
            </select>
            <input
              value={customDose}
              onChange={e => setCustomDose(e.target.value)}
              placeholder={GLP1_MEDICATIONS.find(m => m.name === selectedMed)?.defaultDose || "Dose"}
              style={{
                flex: 1, background: "var(--background)", color: "var(--foreground)",
                border: "1px solid var(--border)", borderRadius: 8, padding: "6px 10px",
                fontSize: 12, outline: "none",
              }}
            />
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            <button
              onClick={() => setShowAdd(false)}
              style={{
                flex: 1, background: "transparent", color: "var(--muted-foreground)",
                borderRadius: 8, padding: "8px 12px", fontSize: 11, border: "1px solid var(--border)", cursor: "pointer",
              }}
            >
              Cancel
            </button>
            <button
              onClick={addInjection}
              style={{
                flex: 1, background: "#7c3aed", color: "white",
                borderRadius: 8, padding: "8px 12px", fontSize: 11, fontWeight: 600,
                border: "none", cursor: "pointer",
              }}
            >
              Log Injection
            </button>
          </div>
        </div>
      ) : (
        <button
          onClick={() => setShowAdd(true)}
          style={{
            width: "100%", background: "rgba(124, 58, 237, 0.08)", color: "#7c3aed",
            borderRadius: 10, padding: 8, fontSize: 12, fontWeight: 500,
            border: "1px solid rgba(124, 58, 237, 0.2)", cursor: "pointer",
          }}
        >
          + Log GLP-1 Injection
        </button>
      )}

      {injections.length === 0 && (
        <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 8 }}>
          Track GLP-1 injections (Ozempic, Wegovy, Mounjaro) with dose and schedule reminders
        </div>
      )}
    </div>
  );
}

// ── Expandable Meal Card ──────────────────────────────────────────────────────

function MealCard({
  log,
  isLast,
  onToggleFavorite,
  isFav,
}: {
  log: FoodLog;
  isLast: boolean;
  onToggleFavorite: (log: FoodLog) => void;
  isFav: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const items = log.foodItems ?? [];
  const totalP = items.reduce((s, f) => s + (f.protein_g ?? 0), 0);
  const totalC = items.reduce((s, f) => s + (f.carbs_g ?? 0), 0);
  const totalF = items.reduce((s, f) => s + (f.fat_g ?? 0), 0);

  // AI insight for this meal
  const insight = (() => {
    const cal = log.totalCalories ?? 0;
    if (cal > 800) return "High-calorie meal -- balance with lighter options later";
    if (cal < 200) return "Light meal -- you may need a snack soon";
    if (totalP > 25) return "Great protein intake -- supports muscle and mood";
    if (totalC > 60) return "Carb-heavy -- expect an energy boost, then a dip";
    return "Balanced meal -- good nutrient distribution";
  })();

  return (
    <div style={{ borderBottom: isLast ? "none" : "1px solid var(--border)" }}>
      {/* Main row -- tappable to expand */}
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          display: "flex", alignItems: "center", padding: "12px 14px", cursor: "pointer",
        }}
      >
        <span style={{ fontSize: 18, marginRight: 10 }}>
          {MEAL_ICONS[log.mealType ?? "snack"] ?? "🍽️"}
        </span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontSize: 13, fontWeight: 500, color: "var(--foreground)",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {log.summary ?? "Meal"}
          </div>
          <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>
            {formatTime(log.loggedAt)} · {getMealLabel(log.mealType)}
          </div>
        </div>
        {log.totalCalories != null && (
          <div style={{ fontSize: 13, fontWeight: 600, color: "#e8b94a", flexShrink: 0, marginRight: 6 }}>
            {log.totalCalories} kcal
          </div>
        )}
        <span style={{ color: "var(--muted-foreground)", fontSize: 14, transition: "transform 0.2s", transform: expanded ? "rotate(90deg)" : "none" }}>&rsaquo;</span>
      </div>

      {/* Expanded: per-item breakdown + AI insight + favorite */}
      {expanded && (
        <div style={{ padding: "0 14px 12px 42px" }}>
          {/* Per-item list */}
          {items.length > 0 && (
            <div style={{ marginBottom: 8 }}>
              {items.map((item, i) => (
                <div key={i} style={{
                  display: "flex", justifyContent: "space-between", alignItems: "baseline",
                  fontSize: 11, color: "var(--muted-foreground)", padding: "3px 0",
                }}>
                  <span style={{ color: "var(--foreground)", fontWeight: 500 }}>{item.name}</span>
                  <span style={{ fontSize: 10, flexShrink: 0, marginLeft: 8 }}>
                    {Math.round(item.calories)} kcal · {item.portion}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Macro breakdown */}
          <div style={{
            display: "flex", gap: 12, fontSize: 10, color: "var(--muted-foreground)", marginBottom: 8,
          }}>
            <span><span style={{ color: "#e879a8" }}>P</span> {Math.round(totalP)}g</span>
            <span><span style={{ color: "#0891b2" }}>C</span> {Math.round(totalC)}g</span>
            <span><span style={{ color: "#7c3aed" }}>F</span> {Math.round(totalF)}g</span>
          </div>

          {/* Food Health Score (Yuka-style) */}
          <div style={{
            padding: "4px 10px", background: "var(--muted)", borderRadius: 10,
            marginBottom: 8, border: "1px solid var(--border)",
          }}>
            <FoodScoreCard
              food={{
                calories: log.totalCalories ?? 0,
                protein_g: totalP,
                carbs_g: totalC,
                fat_g: totalF,
              }}
              compact
            />
          </div>

          {/* AI insight */}
          <div style={{
            fontSize: 11, color: "var(--muted-foreground)", fontStyle: "italic",
            padding: "6px 10px", background: "var(--muted)", borderRadius: 8,
            marginBottom: 8,
          }}>
            {insight}
          </div>

          {/* Favorite button */}
          <button
            onClick={(e) => { e.stopPropagation(); onToggleFavorite(log); }}
            style={{
              display: "flex", alignItems: "center", gap: 6,
              background: "none", border: "1px solid var(--border)", borderRadius: 8,
              padding: "6px 12px", fontSize: 11, cursor: "pointer",
              color: isFav ? "#e8b94a" : "var(--muted-foreground)",
            }}
          >
            <span>{isFav ? "\u2605" : "\u2606"}</span>
            <span>{isFav ? "Favorited" : "Save as Favorite"}</span>
          </button>
        </div>
      )}
    </div>
  );
}

// ── Barcode Entry Panel ───────────────────────────────────────────────────────

function BarcodePanel({
  onLog,
  onCancel,
}: {
  onLog: (items: FoodItem[], summary: string) => void;
  onCancel: () => void;
}) {
  const [barcodeInput, setBarcodeInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [product, setProduct] = useState<BarcodeProduct | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [servings, setServings] = useState("1");
  const [scanning, setScanning] = useState(false);
  const scannerRef = useRef<HTMLDivElement>(null);
  const scannerInstanceRef = useRef<any>(null);

  // Camera barcode scanner
  const startScanner = useCallback(async () => {
    setScanning(true);
    try {
      const { Html5Qrcode } = await import("html5-qrcode");
      const scannerId = "barcode-scanner-" + Date.now();
      if (scannerRef.current) {
        scannerRef.current.id = scannerId;
        const scanner = new Html5Qrcode(scannerId);
        scannerInstanceRef.current = scanner;
        await scanner.start(
          { facingMode: "environment" },
          { fps: 10, qrbox: { width: 250, height: 100 }, aspectRatio: 2.0 },
          (decodedText: string) => {
            // Barcode found! Stop scanner and look up
            scanner.stop().catch(() => {});
            setScanning(false);
            setBarcodeInput(decodedText);
            // Auto-lookup
            setLoading(true);
            setNotFound(false);
            setProduct(null);
            lookupBarcode(decodedText).then((result) => {
              if (!result) setNotFound(true);
              else { setProduct(result); setServings("1"); }
            }).catch(() => setNotFound(true)).finally(() => setLoading(false));
          },
          () => {} // ignore scan errors
        );
      }
    } catch (e) {
      console.error("Barcode scanner failed:", e);
      setScanning(false);
    }
  }, []);

  const stopScanner = useCallback(() => {
    scannerInstanceRef.current?.stop?.().catch(() => {});
    scannerInstanceRef.current = null;
    setScanning(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => { scannerInstanceRef.current?.stop?.().catch(() => {}); };
  }, []);

  const handleLookup = useCallback(async () => {
    if (!barcodeInput.trim()) return;
    setLoading(true);
    setNotFound(false);
    setProduct(null);
    try {
      const result = await lookupBarcode(barcodeInput.trim());
      if (!result) {
        setNotFound(true);
      } else {
        setProduct(result);
        setServings("1");
      }
    } catch {
      setNotFound(true);
    } finally {
      setLoading(false);
    }
  }, [barcodeInput]);

  const handleLog = useCallback(() => {
    if (!product) return;
    const s = parseFloat(servings) || 1;
    const item: FoodItem = {
      name: `${product.name}${product.brand ? ` (${product.brand})` : ""}`,
      portion: product.servingSize
        ? `${s === 1 ? "" : `${s}x `}${product.servingSize}`
        : `${s} serving${s !== 1 ? "s" : ""}`,
      calories: Math.round(product.calories * s),
      protein_g: Math.round(product.protein_g * s * 10) / 10,
      carbs_g: Math.round(product.carbs_g * s * 10) / 10,
      fat_g: Math.round(product.fat_g * s * 10) / 10,
    };
    onLog(
      [item],
      `${product.name}${product.servingSize ? ` (${product.servingSize})` : ""}`
    );
  }, [product, servings, onLog]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
      style={{
        background: "var(--card)", borderRadius: 16, border: "1px solid var(--border)",
        padding: 14, marginBottom: 14, boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
      }}
    >
      <div style={{ fontSize: 12, fontWeight: 600, color: "var(--primary)", marginBottom: 8 }}>
        Barcode Scanner
      </div>

      {/* Camera scanner */}
      {scanning ? (
        <div style={{ marginBottom: 10 }}>
          <div ref={scannerRef} style={{ width: "100%", borderRadius: 10, overflow: "hidden" }} />
          <button onClick={stopScanner} style={{
            width: "100%", marginTop: 8, padding: 8, fontSize: 12, fontWeight: 600,
            background: "#e879a8", color: "white", border: "none", borderRadius: 8, cursor: "pointer",
          }}>
            Stop Scanner
          </button>
        </div>
      ) : (
        <button onClick={startScanner} style={{
          width: "100%", marginBottom: 10, padding: 12, fontSize: 13, fontWeight: 600,
          background: "var(--primary)", color: "white", border: "none", borderRadius: 10, cursor: "pointer",
          display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
        }}>
          📷 Scan Barcode with Camera
        </button>
      )}

      <p style={{ fontSize: 11, color: "var(--muted-foreground)", margin: "0 0 8px 0" }}>
        Or enter the barcode number manually:
      </p>
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <input
          value={barcodeInput}
          onChange={(e) => { setBarcodeInput(e.target.value); setNotFound(false); setProduct(null); }}
          placeholder="e.g. 012345678901"
          onKeyDown={(e) => { if (e.key === "Enter") handleLookup(); }}
          disabled={loading}
          style={{
            flex: 1, background: "var(--background)", color: "var(--foreground)",
            border: "1px solid var(--border)", borderRadius: 8, padding: "8px 12px",
            fontSize: 13, outline: "none", fontFamily: "inherit",
          }}
        />
        <button
          onClick={handleLookup}
          disabled={loading || !barcodeInput.trim()}
          style={{
            background: barcodeInput.trim() ? "var(--primary)" : "var(--muted)",
            color: barcodeInput.trim() ? "white" : "var(--muted-foreground)",
            borderRadius: 8, padding: "8px 14px", fontSize: 12, fontWeight: 600,
            border: "none", cursor: "pointer",
          }}
        >
          {loading ? "Searching…" : "Look Up"}
        </button>
      </div>

      {/* Product found */}
      {product && (
        <div style={{ background: "var(--muted)", borderRadius: 10, padding: 12, marginBottom: 8 }}>
          <div style={{ display: "flex", gap: 10, marginBottom: 8 }}>
            {product.imageUrl && (
              <img src={product.imageUrl} alt={product.name}
                style={{ width: 48, height: 48, borderRadius: 6, objectFit: "cover", border: "1px solid var(--border)" }}
              />
            )}
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: "var(--foreground)" }}>{product.name}</div>
              {product.brand && <div style={{ fontSize: 10, color: "var(--muted-foreground)" }}>{product.brand}</div>}
              {product.servingSize && <div style={{ fontSize: 10, color: "var(--muted-foreground)" }}>Per serving: {product.servingSize}</div>}
            </div>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 4, textAlign: "center", marginBottom: 8 }}>
            <div style={{ background: "var(--card)", borderRadius: 6, padding: "4px 0" }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "var(--foreground)" }}>{product.calories}</div>
              <div style={{ fontSize: 9, color: "var(--muted-foreground)" }}>kcal</div>
            </div>
            <div style={{ background: "var(--card)", borderRadius: 6, padding: "4px 0" }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#e879a8" }}>{product.protein_g}g</div>
              <div style={{ fontSize: 9, color: "var(--muted-foreground)" }}>protein</div>
            </div>
            <div style={{ background: "var(--card)", borderRadius: 6, padding: "4px 0" }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#0891b2" }}>{product.carbs_g}g</div>
              <div style={{ fontSize: 9, color: "var(--muted-foreground)" }}>carbs</div>
            </div>
            <div style={{ background: "var(--card)", borderRadius: 6, padding: "4px 0" }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#7c3aed" }}>{product.fat_g}g</div>
              <div style={{ fontSize: 9, color: "var(--muted-foreground)" }}>fat</div>
            </div>
          </div>
          {product.allergens && (
            <div style={{ fontSize: 10, color: "#d4a017", marginBottom: 6 }}>
              Contains: {product.allergens}
            </div>
          )}
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
            <span style={{ fontSize: 11, color: "var(--muted-foreground)" }}>Servings:</span>
            <input
              type="number" min="0.25" step="0.25"
              value={servings} onChange={(e) => setServings(e.target.value)}
              style={{
                width: 60, background: "var(--card)", color: "var(--foreground)",
                border: "1px solid var(--border)", borderRadius: 6, padding: "4px 8px",
                fontSize: 12, textAlign: "center", outline: "none",
              }}
            />
          </div>
          <button onClick={handleLog} style={{
            width: "100%", background: "var(--primary)", color: "white",
            borderRadius: 10, padding: 10, fontSize: 13, fontWeight: 600,
            border: "none", cursor: "pointer",
          }}>
            Log This Meal
          </button>
        </div>
      )}

      {/* Not found */}
      {notFound && (
        <div style={{
          background: "rgba(212, 160, 23, 0.08)", border: "1px solid rgba(212, 160, 23, 0.3)",
          borderRadius: 8, padding: 10, fontSize: 11, color: "#d4a017", marginBottom: 8,
        }}>
          Barcode not found in OpenFoodFacts. Try describing your meal instead.
        </div>
      )}

      <button onClick={onCancel} style={{
        width: "100%", background: "transparent", color: "var(--muted-foreground)",
        borderRadius: 10, padding: 8, fontSize: 12, border: "1px solid var(--border)",
        cursor: "pointer",
      }}>
        Cancel
      </button>
    </motion.div>
  );
}

// ── Glucose Section (conditional on CGM device) ───────────────────────────────

function GlucoseSection({ userId }: { userId: string }) {
  const { data: connections } = useQuery<Array<{ provider: string; status: string }>>({
    queryKey: ["/api/device-connections", userId],
    queryFn: async () => {
      try {
        const res = await fetch(resolveUrl(`/api/device-connections/${userId}`));
        if (!res.ok) return [];
        return res.json();
      } catch {
        return [];
      }
    },
    staleTime: 60_000,
  });

  const hasCGM = useMemo(() => {
    if (!connections) return false;
    return connections.some(
      (c) =>
        (c.provider === "dexcom" || c.provider === "libre" || c.provider === "cgm") &&
        c.status === "active"
    );
  }, [connections]);

  const { data: glucoseData } = useQuery<{ current: number | null; trend: string | null }>({
    queryKey: ["/api/glucose/current", userId],
    queryFn: async () => {
      try {
        const res = await fetch(resolveUrl(`/api/glucose/current/${userId}`));
        if (!res.ok) return { current: null, trend: null };
        return res.json();
      } catch {
        return { current: null, trend: null };
      }
    },
    enabled: hasCGM,
    refetchInterval: 30_000,
  });

  if (!hasCGM) return null;

  const glucose = glucoseData?.current;
  if (glucose == null) return null;

  const glucoseColor =
    glucose >= 70 && glucose <= 120
      ? "#06b6d4"
      : glucose > 120 && glucose <= 140
        ? "#d4a017"
        : "#e879a8";

  const trendArrow = glucoseData?.trend === "rising"
    ? "\u2191"
    : glucoseData?.trend === "falling"
      ? "\u2193"
      : "\u2192";

  return (
    <div style={{ marginTop: 16 }}>
      <div style={{
        fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
        textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 8,
      }}>
        Real-Time Glucose
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
        <span style={{ fontSize: 28, fontWeight: 700, color: glucoseColor }}>{glucose}</span>
        <span style={{ fontSize: 14, color: glucoseColor }}>{trendArrow}</span>
        <span style={{ fontSize: 12, color: "var(--muted-foreground)" }}>mg/dL</span>
      </div>
      <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 4 }}>
        {glucose >= 70 && glucose <= 120
          ? "In range -- good glucose control"
          : glucose > 140
            ? "Elevated -- consider a walk or lighter carbs"
            : glucose > 120
              ? "Slightly elevated -- monitor over the next hour"
              : "Below range -- consider a small snack"}
      </div>
    </div>
  );
}

// ── Tab definitions ──────────────────────────────────────────────────────────

type TabId = "log" | "vitamins" | "supplements" | "insights" | "history";

const TABS: { id: TabId; label: string }[] = [
  { id: "log", label: "Log" },
  { id: "vitamins", label: "Vitamins" },
  { id: "supplements", label: "Supplements" },
  { id: "insights", label: "Insights" },
  { id: "history", label: "History" },
];

// ── Tab slide animation variants ─────────────────────────────────────────────

const tabSlideVariants = {
  enter: (direction: number) => ({
    x: direction > 0 ? 60 : -60,
    opacity: 0,
  }),
  center: {
    x: 0,
    opacity: 1,
  },
  exit: (direction: number) => ({
    x: direction > 0 ? -60 : 60,
    opacity: 0,
  }),
};

// ── Main Component ────────────────────────────────────────────────────────────

export default function Nutrition() {
  const userId = getParticipantId();
  const qc = useQueryClient();
  const [captureMode, setCaptureMode] = useState<"none" | "camera" | "text" | "barcode">("none");
  const [mealText, setMealText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [selectedMealType, setSelectedMealType] = useState<string>(autoMealType());
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const [favorites, setFavorites] = useState<FavoriteMeal[]>(loadFavorites);

  const [activeTab, setActiveTab] = useState<TabId>("log");
  const [tabDirection, setTabDirection] = useState(0);

  // Supplement vitamin tracking (Tasks 2/5: combine food + supplement sources)
  const [supplementVitamins, setSupplementVitamins] = useState<VitaminData | null>(() => {
    // Initialize from stored supplements
    const supps = loadSupplements();
    const log = loadTodayLog();
    const totals: VitaminData = {
      vitamin_d_mcg: 0, vitamin_b12_mcg: 0, vitamin_c_mg: 0,
      iron_mg: 0, magnesium_mg: 0, zinc_mg: 0, omega3_g: 0,
    };
    let hasAny = false;
    for (const supp of supps) {
      if (log[supp.id]) {
        hasAny = true;
        const sv = getSupplementVitamins(supp);
        for (const k of Object.keys(totals) as (keyof VitaminData)[]) {
          totals[k] += sv[k];
        }
      }
    }
    return hasAny ? totals : null;
  });

  const handleSupplementsChange = useCallback((supps: Supplement[], log: SupplementDailyLog) => {
    const totals: VitaminData = {
      vitamin_d_mcg: 0, vitamin_b12_mcg: 0, vitamin_c_mg: 0,
      iron_mg: 0, magnesium_mg: 0, zinc_mg: 0, omega3_g: 0,
    };
    let hasAny = false;
    for (const supp of supps) {
      if (log[supp.id]) {
        hasAny = true;
        const sv = getSupplementVitamins(supp);
        for (const k of Object.keys(totals) as (keyof VitaminData)[]) {
          totals[k] += sv[k];
        }
      }
    }
    setSupplementVitamins(hasAny ? totals : null);
  }, []);

  // Helper: sync food log to Railway ML backend (fire-and-forget)
  const syncFoodToRailway = useCallback((data: {
    totalCalories?: number; summary?: string; foodItems?: FoodItem[];
    dominantMacro?: string; mealType?: string;
  }) => {
    syncFoodLogToML({
      user_id: userId,
      total_calories: data.totalCalories ?? 0,
      total_protein_g: data.foodItems?.reduce((s, i) => s + (i.protein_g ?? 0), 0) ?? 0,
      total_carbs_g: data.foodItems?.reduce((s, i) => s + (i.carbs_g ?? 0), 0) ?? 0,
      total_fat_g: data.foodItems?.reduce((s, i) => s + (i.fat_g ?? 0), 0) ?? 0,
      total_fiber_g: 0,
      dominant_macro: data.dominantMacro,
      meal_type: data.mealType,
      summary: data.summary,
      food_items: data.foodItems as Array<Record<string, unknown>> | undefined,
    });
  }, [userId]);

  // ── Meal-cognitive correlation (Issue #507) ──
  const [mealCogInsight, setMealCogInsight] = useState<string | null>(null);
  useEffect(() => {
    (async () => {
      try {
        const [foodLogs, emotionHistory] = await Promise.all([
          sbGetFoodLogs(userId),
          getEmotionHistory(userId, 7),
        ]);
        if (!foodLogs?.length || !emotionHistory?.length) return;

        const meals: MealTimestamp[] = foodLogs
          .filter((l: any) => l.loggedAt || l.created_at)
          .map((l: any) => ({
            timestamp: l.loggedAt ?? l.created_at,
            label: l.mealType ?? "meal",
          }));

        const readings: TimestampedReading[] = emotionHistory
          .filter((e: any) => e.timestamp || e.created_at)
          .map((e: any) => ({
            timestamp: e.timestamp ?? e.created_at,
            focus: e.focus ?? 0.5,
            stress: e.stress ?? 0.5,
          }));

        const insight = computeMealCognitiveCorrelation(meals, readings);
        if (insight && insight.type !== "no_change") {
          setMealCogInsight(generateInsight(insight));
        }
      } catch {
        // best-effort
      }
    })();
  }, [userId]);

  const { scores } = useScores(userId);

  const { data: logs } = useQuery<FoodLog[]>({
    queryKey: ["/api/food/logs", userId],
    queryFn: async () => {
      let apiLogs: FoodLog[] = [];
      // 1. Try Express API
      try {
        const res = await fetch(resolveUrl(`/api/food/logs/${userId}`));
        if (res.ok) {
          const data = await res.json();
          if (Array.isArray(data)) apiLogs = data;
        }
      } catch { /* API unavailable */ }
      // 2. Try Supabase (survives APK reinstall)
      let sbLogs: FoodLog[] = [];
      try {
        const { getSupabase } = await import("@/lib/supabase-browser");
        const sb = await getSupabase();
        if (sb) {
          const { data: rows } = await sb.from("food_logs").select("*")
            .eq("user_id", userId).order("created_at", { ascending: false }).limit(200);
          if (rows) {
            sbLogs = rows.map((r: any) => ({
              id: r.id ?? `sb_${r.created_at}`,
              loggedAt: r.created_at,
              mealType: r.meal_type ?? "meal",
              summary: r.summary,
              totalCalories: r.calories,
              dominantMacro: r.dominant_macro ?? null,
              foodItems: r.food_items ?? null,
              vitamins: r.vitamins ?? null,
            }));
          }
        }
      } catch { /* Supabase unavailable */ }
      // 3. localStorage (offline cache)
      let localLogs: FoodLog[] = [];
      try {
        localLogs = sbGetGeneric(`ndw_food_logs_${userId}`) ?? [];
      } catch { /* ignore */ }
      // Merge all sources, deduplicate by id
      const allLogs = [...apiLogs, ...sbLogs, ...localLogs];
      const seen = new Set<string>();
      const unique = allLogs.filter((l) => {
        const key = l.id ?? l.loggedAt;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
      unique.sort((a, b) => new Date(b.loggedAt).getTime() - new Date(a.loggedAt).getTime());
      return unique;
    },
  });

  const todayLogs = useMemo(() => {
    if (!logs) return [];
    return logs.filter((l) => isToday(l.loggedAt)).sort(
      (a, b) => new Date(b.loggedAt).getTime() - new Date(a.loggedAt).getTime()
    );
  }, [logs]);

  const totalCalories = useMemo(
    () => todayLogs.reduce((s, l) => s + (l.totalCalories ?? 0), 0),
    [todayLogs]
  );

  const { totalProtein, totalCarbs, totalFat } = useMemo(() => {
    let p = 0, c = 0, f = 0;
    for (const l of todayLogs) {
      for (const fi of l.foodItems ?? []) {
        p += fi.protein_g ?? 0;
        c += fi.carbs_g ?? 0;
        f += fi.fat_g ?? 0;
      }
    }
    return { totalProtein: p, totalCarbs: c, totalFat: f };
  }, [todayLogs]);

  const voiceData = useVoiceData();
  const craving = useMemo(() => getCravingFromVoice(voiceData), [voiceData]);

  // Compute recent unique meals for quick re-log
  const recentMeals = useMemo(() => {
    if (!logs) return [];
    const seen = new Set<string>();
    const result: FoodLog[] = [];
    for (const l of logs) {
      const key = l.summary ?? l.foodItems?.map((f) => f.name).join(", ") ?? "";
      if (!key || seen.has(key)) continue;
      seen.add(key);
      result.push(l);
      if (result.length >= 5) break;
    }
    return result;
  }, [logs]);

  // Compute nutrition insights
  const insights = useMemo(
    () => generateInsights(todayLogs, logs, totalProtein, totalCarbs, totalCalories),
    [todayLogs, logs, totalProtein, totalCarbs, totalCalories]
  );

  // ── History tab data ────────────────────────────────────────────────────────

  // Fetch food-mood correlation data for history view
  const { data: moodCorrelation } = useQuery<{
    entries: Array<{
      food: {
        id: number;
        summary: string | null;
        mealType: string | null;
        totalCalories: number | null;
        dominantMacro: string | null;
        foodItems: FoodItem[] | null;
        loggedAt: string;
      };
      moodLog: {
        moodScore: number | null;
        energyLevel: number | null;
        notes: string | null;
        loggedAt: string;
      } | null;
      emotionReading: {
        dominantEmotion: string | null;
        stress: number | null;
        happiness: number | null;
        valence: number | null;
        arousal: number | null;
        timestamp: string;
      } | null;
    }>;
    totalFoodLogs: number;
    matchedWithMood: number;
    matchedWithEmotion: number;
  }>({
    queryKey: ["/api/food/mood-correlation", userId],
    queryFn: async () => {
      try {
        const res = await fetch(resolveUrl(`/api/food/mood-correlation/${userId}?days=7`));
        if (res.ok) return res.json();
      } catch { /* API unavailable */ }
      return { entries: [], totalFoodLogs: 0, matchedWithMood: 0, matchedWithEmotion: 0 };
    },
    staleTime: 60_000,
    enabled: activeTab === "history",
  });

  // 7-day calorie trend data for AreaChart
  const calorieTrendData = useMemo(() => {
    const days: { date: string; label: string; calories: number }[] = [];
    const now = new Date();
    for (let i = 6; i >= 0; i--) {
      const d = new Date(now);
      d.setDate(d.getDate() - i);
      const dateStr = d.toDateString();
      const dayLabel = i === 0 ? "Today" : i === 1 ? "Yesterday" : d.toLocaleDateString([], { weekday: "short" });
      days.push({ date: dateStr, label: dayLabel, calories: 0 });
    }
    // Sum calories from all logs (API + local)
    const allLogs = logs ?? [];
    for (const log of allLogs) {
      const logDate = new Date(log.loggedAt).toDateString();
      const entry = days.find((d) => d.date === logDate);
      if (entry) {
        entry.calories += log.totalCalories ?? 0;
      }
    }
    return days;
  }, [logs]);

  // 7-day protein trend (combined with calorie trend for dual-line chart)
  const nutritionTrendData = useMemo(() => {
    const allLogs = logs ?? [];
    const now = new Date();
    return Array.from({ length: 7 }, (_, i) => {
      const d = new Date(now);
      d.setDate(d.getDate() - (6 - i));
      const dateStr = d.toDateString();
      const label = i === 6 ? "Today" : i === 5 ? "Yest" : d.toLocaleDateString([], { weekday: "short" });
      const dayLogs = allLogs.filter(l => new Date(l.loggedAt).toDateString() === dateStr);
      const cal = dayLogs.reduce((s, l) => s + (l.totalCalories ?? 0), 0);
      const prot = dayLogs.reduce((s, l) =>
        s + (l.foodItems ?? []).reduce((sp, fi) => sp + (fi.protein_g ?? 0), 0), 0);
      return { label, cal: cal || null, prot: prot ? Math.round(prot) : null };
    });
  }, [logs]);

  // Group logs by date for the meal timeline (last 7 days)
  const timelineByDate = useMemo(() => {
    const allLogs = logs ?? [];
    const now = new Date();
    const sevenDaysAgo = new Date(now);
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);

    const recentLogs = allLogs
      .filter((l) => new Date(l.loggedAt) >= sevenDaysAgo)
      .sort((a, b) => new Date(b.loggedAt).getTime() - new Date(a.loggedAt).getTime());

    const groups: Map<string, FoodLog[]> = new Map();
    for (const log of recentLogs) {
      const dateKey = new Date(log.loggedAt).toDateString();
      if (!groups.has(dateKey)) groups.set(dateKey, []);
      groups.get(dateKey)!.push(log);
    }
    return groups;
  }, [logs]);

  // Build mood map from correlation data for quick lookup by food log id
  const moodByFoodId = useMemo(() => {
    const map = new Map<number, { emotion: string | null; stress: number | null; moodScore: number | null }>();
    if (!moodCorrelation?.entries) return map;
    for (const entry of moodCorrelation.entries) {
      const emotion = entry.emotionReading?.dominantEmotion ?? null;
      const stress = entry.emotionReading?.stress ?? null;
      const moodScore = entry.moodLog?.moodScore ?? null;
      map.set(entry.food.id, { emotion, stress, moodScore });
    }
    return map;
  }, [moodCorrelation]);

  // Food-mood insight
  const foodMoodInsight = useMemo(() => {
    if (!moodCorrelation?.entries || moodCorrelation.entries.length < 2) {
      return null;
    }

    const entriesWithMood = moodCorrelation.entries.filter(
      (e) => e.emotionReading !== null || e.moodLog !== null
    );
    if (entriesWithMood.length < 2) return null;

    // Split into high-cal days (>2000) and lighter days
    const dailyCals = new Map<string, { calories: number; stressValues: number[]; moodValues: number[] }>();
    for (const entry of moodCorrelation.entries) {
      const dateKey = new Date(entry.food.loggedAt).toDateString();
      if (!dailyCals.has(dateKey)) dailyCals.set(dateKey, { calories: 0, stressValues: [], moodValues: [] });
      const day = dailyCals.get(dateKey)!;
      day.calories += entry.food.totalCalories ?? 0;
      if (entry.emotionReading?.stress != null) day.stressValues.push(entry.emotionReading.stress);
      if (entry.moodLog?.moodScore != null) day.moodValues.push(entry.moodLog.moodScore);
    }

    const daysArr = Array.from(dailyCals.values());
    const highCalDays = daysArr.filter((d) => d.calories > 2000 && d.stressValues.length > 0);
    const lightDays = daysArr.filter((d) => d.calories <= 2000 && d.stressValues.length > 0);

    if (highCalDays.length > 0 && lightDays.length > 0) {
      const avgStressHigh = highCalDays.reduce((s, d) => s + d.stressValues.reduce((a, b) => a + b, 0) / d.stressValues.length, 0) / highCalDays.length;
      const avgStressLight = lightDays.reduce((s, d) => s + d.stressValues.reduce((a, b) => a + b, 0) / d.stressValues.length, 0) / lightDays.length;
      return `On days you ate >2000 cal, your avg stress was ${Math.round(avgStressHigh * 100)}% vs ${Math.round(avgStressLight * 100)}% on lighter days.`;
    }

    // Fallback: check protein vs mood
    const withProteinAndMood = entriesWithMood.filter((e) => {
      const items = e.food.foodItems ?? [];
      const totalP = items.reduce((s, f) => s + (f.protein_g ?? 0), 0);
      return totalP > 0 && (e.moodLog?.moodScore != null || e.emotionReading?.valence != null);
    });

    if (withProteinAndMood.length >= 2) {
      const highProtein = withProteinAndMood.filter((e) => {
        const items = e.food.foodItems ?? [];
        return items.reduce((s, f) => s + (f.protein_g ?? 0), 0) > 20;
      });
      const lowProtein = withProteinAndMood.filter((e) => {
        const items = e.food.foodItems ?? [];
        return items.reduce((s, f) => s + (f.protein_g ?? 0), 0) <= 20;
      });

      if (highProtein.length > 0 && lowProtein.length > 0) {
        const avgValHigh = highProtein.reduce((s, e) => s + (e.emotionReading?.valence ?? e.moodLog?.moodScore ?? 0), 0) / highProtein.length;
        const avgValLow = lowProtein.reduce((s, e) => s + (e.emotionReading?.valence ?? e.moodLog?.moodScore ?? 0), 0) / lowProtein.length;
        if (avgValHigh > avgValLow) {
          return "Your mood tends to be more positive after meals with higher protein content.";
        }
      }
    }

    return `You logged ${moodCorrelation.totalFoodLogs} meals in the last 7 days, ${moodCorrelation.matchedWithEmotion} paired with emotion data.`;
  }, [moodCorrelation]);

  // Compute simple nutrition quality indicators from food items
  const qualityIndicators = useMemo(() => {
    const allItems = todayLogs.flatMap((l) => l.foodItems ?? []);
    const positives: string[] = [];
    const negatives: string[] = [];

    const hasVeg = allItems.some((fi) =>
      /vegetable|salad|broccoli|spinach|carrot|kale|lettuce|greens|zucchini|pepper/i.test(fi.name)
    );
    const hasFruit = allItems.some((fi) =>
      /fruit|apple|banana|orange|berry|berries|mango|grape/i.test(fi.name)
    );
    const hasWholeGrain = allItems.some((fi) =>
      /whole grain|oat|quinoa|brown rice|whole wheat/i.test(fi.name)
    );
    const hasProcessed = allItems.some((fi) =>
      /chips|fries|candy|soda|pizza|burger|hot dog|donut|cookie/i.test(fi.name)
    );
    const highSugar = totalCarbs > 150 && totalProtein < 40;

    if (hasVeg) positives.push("Vegetables logged");
    if (hasFruit) positives.push("Fruits detected");
    if (hasWholeGrain) positives.push("Whole grains included");
    if (totalProtein >= PROTEIN_GOAL * 0.5) positives.push("Good protein intake");

    if (hasProcessed) negatives.push("Processed food detected");
    if (highSugar) negatives.push("High sugar/carb ratio");
    if (totalProtein < 20 && todayLogs.length >= 2) negatives.push("Low protein so far");

    return { positives, negatives };
  }, [todayLogs, totalProtein, totalCarbs]);

  // Auto-calculated food quality score (Task 3)
  const foodQualityScore = useMemo(
    () => calculateFoodQualityScore(todayLogs, totalProtein, totalCarbs, totalFat, totalCalories),
    [todayLogs, totalProtein, totalCarbs, totalFat, totalCalories]
  );

  // Handle favorite toggle
  const handleToggleFavorite = useCallback((log: FoodLog) => {
    const updated = toggleFavorite(log);
    setFavorites(updated);
  }, []);

  // Handle re-logging a meal from recent/favorites
  const handleRelog = useCallback(async (summary: string, items: FoodItem[]) => {
    if (items.length === 0) return;
    setIsAnalyzing(true);
    setAnalysisError(null);
    try {
      const res = await apiRequest("POST", "/api/food/analyze", {
        userId,
        mealType: selectedMealType,
        textDescription: summary,
      });
      const data = await res.json();
      // Persist API result to localStorage
      if (data?.id) {
        persistFoodLogLocally(userId, {
          id: String(data.id),
          loggedAt: data.loggedAt ?? new Date().toISOString(),
          mealType: data.mealType ?? autoMealType(),
          summary: data.summary ?? null,
          totalCalories: data.totalCalories ?? null,
          dominantMacro: data.dominantMacro ?? null,
          foodItems: data.foodItems ?? null,
          vitamins: data.vitamins ?? null,
        });
      }
      hapticSuccess();
      qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
      syncFoodToRailway({ summary: data.summary, totalCalories: data.totalCalories, dominantMacro: data.dominantMacro, foodItems: data.foodItems, mealType: selectedMealType });
    } catch {
      // Fallback: use local estimation when API fails — always persist to localStorage
      const local = estimateNutritionLocally(summary);
      const entry: FoodLog = {
        id: `local_${Date.now()}`,
        loggedAt: new Date().toISOString(),
        mealType: selectedMealType,
        summary: local.summary,
        totalCalories: local.total_calories,
        dominantMacro: local.dominant_macro,
        foodItems: local.food_items,
        vitamins: null,
      };
      persistFoodLogLocally(userId, entry);
      try { await apiRequest("POST", "/api/food/log", { userId, ...entry }); } catch { /* ok */ }
      hapticSuccess();
      qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
      syncFoodToRailway({ summary: local.summary, totalCalories: local.total_calories, dominantMacro: local.dominant_macro, foodItems: local.food_items, mealType: selectedMealType });
    } finally {
      setIsAnalyzing(false);
    }
  }, [userId, qc, syncFoodToRailway]);

  // Handle barcode log
  const handleBarcodeLog = useCallback(async (items: FoodItem[], summary: string) => {
    setIsAnalyzing(true);
    setAnalysisError(null);
    setCaptureMode("none");
    try {
      const res = await apiRequest("POST", "/api/food/analyze", {
        userId,
        mealType: selectedMealType,
        textDescription: `${summary} - ${items.map((i) => `${i.name} (${i.calories} kcal, ${i.protein_g}g protein, ${i.carbs_g}g carbs, ${i.fat_g}g fat)`).join(", ")}`,
      });
      const data = await res.json();
      // Persist API result to localStorage
      if (data?.id) {
        persistFoodLogLocally(userId, {
          id: String(data.id),
          loggedAt: data.loggedAt ?? new Date().toISOString(),
          mealType: data.mealType ?? autoMealType(),
          summary: data.summary ?? null,
          totalCalories: data.totalCalories ?? null,
          dominantMacro: data.dominantMacro ?? null,
          foodItems: data.foodItems ?? null,
          vitamins: data.vitamins ?? null,
        });
      }
      hapticSuccess();
      qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
      syncFoodToRailway({ summary: data.summary, totalCalories: data.totalCalories, dominantMacro: data.dominantMacro, foodItems: data.foodItems, mealType: selectedMealType });
    } catch {
      // Fallback: log barcode items directly — always persist to localStorage
      const totalCal = items.reduce((s, i) => s + i.calories, 0);
      const totalP = items.reduce((s, i) => s + i.protein_g, 0);
      const totalC = items.reduce((s, i) => s + i.carbs_g, 0);
      const totalF = items.reduce((s, i) => s + i.fat_g, 0);
      const dominant = totalP >= totalC && totalP >= totalF ? "protein" : totalC >= totalF ? "carbs" : "fat";
      const entry: FoodLog = {
        id: `local_${Date.now()}`,
        loggedAt: new Date().toISOString(),
        mealType: selectedMealType,
        summary,
        totalCalories: totalCal,
        dominantMacro: dominant,
        foodItems: items,
        vitamins: null,
      };
      persistFoodLogLocally(userId, entry);
      try { await apiRequest("POST", "/api/food/log", { userId, ...entry }); } catch { /* ok */ }
      hapticSuccess();
      qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
      syncFoodToRailway({ summary, totalCalories: totalCal, dominantMacro: dominant, foodItems: items, mealType: selectedMealType });
    } finally {
      setIsAnalyzing(false);
    }
  }, [userId, qc, syncFoodToRailway]);

  // Tab switching with direction tracking
  const handleTabChange = useCallback((newTab: TabId) => {
    const oldIndex = TABS.findIndex(t => t.id === activeTab);
    const newIndex = TABS.findIndex(t => t.id === newTab);
    setTabDirection(newIndex > oldIndex ? 1 : -1);
    setActiveTab(newTab);
  }, [activeTab]);

  // Calorie remaining text
  const remaining = CAL_GOAL - totalCalories;
  const isOver = remaining < 0;
  const goalColor = isOver
    ? Math.abs(remaining) > 300 ? "#e879a8" : "#d4a017"
    : "#06b6d4";

  return (
    <main
      style={{
        background: "var(--background)",
        color: "var(--foreground)",
        fontFamily: "Inter, system-ui, sans-serif",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* ── Sticky Header: Title + Ring + Macros ─────────────────────────── */}
      <div style={{
        position: "sticky", top: 0, zIndex: 20,
        background: "var(--background)",
        borderBottom: "1px solid var(--border)",
        padding: "12px 16px 10px",
      }}>
        {/* Row: title + calorie ring + remaining */}
        <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 10 }}>
          <CalorieRingCompact calories={totalCalories} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <h1 style={{ fontSize: 18, fontWeight: 700, margin: 0, color: "var(--foreground)" }}>
              Nutrition
            </h1>
            <div style={{ fontSize: 12, fontWeight: 600, color: goalColor, marginTop: 4 }}>
              {isOver
                ? `${Math.abs(remaining)} kcal over`
                : remaining === 0
                  ? "Goal reached"
                  : `${remaining} kcal remaining`}
            </div>
            <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>
              of {CAL_GOAL} kcal goal
            </div>
          </div>
        </div>

        {/* Macro bars — always visible */}
        <div style={{ display: "flex", gap: 12 }}>
          {/* Protein */}
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 3 }}>
              <span style={{ fontSize: 10, fontWeight: 600, color: "#e879a8" }}>Protein</span>
              <span style={{ fontSize: 13, fontWeight: 700, color: "#e879a8" }}>{Math.round(totalProtein)}g</span>
            </div>
            <MacroBar value={totalProtein} goal={PROTEIN_GOAL} color="#e879a8" />
          </div>
          {/* Carbs */}
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 3 }}>
              <span style={{ fontSize: 10, fontWeight: 600, color: "#0891b2" }}>Carbs</span>
              <span style={{ fontSize: 13, fontWeight: 700, color: "#0891b2" }}>{Math.round(totalCarbs)}g</span>
            </div>
            <MacroBar value={totalCarbs} goal={CARBS_GOAL} color="#0891b2" />
          </div>
          {/* Fat */}
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 3 }}>
              <span style={{ fontSize: 10, fontWeight: 600, color: "#7c3aed" }}>Fat</span>
              <span style={{ fontSize: 13, fontWeight: 700, color: "#7c3aed" }}>{Math.round(totalFat)}g</span>
            </div>
            <MacroBar value={totalFat} goal={FAT_GOAL} color="#7c3aed" />
          </div>
        </div>

        {/* ── Tab Bar ──────────────────────────────────────────────────── */}
        <div style={{
          display: "flex", gap: 6, marginTop: 12,
          overflowX: "auto", WebkitOverflowScrolling: "touch",
        }}>
          {TABS.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => handleTabChange(tab.id)}
                style={{
                  flex: "0 0 auto",
                  padding: "6px 16px",
                  borderRadius: 20,
                  fontSize: 12,
                  fontWeight: isActive ? 600 : 500,
                  border: "none",
                  cursor: "pointer",
                  background: isActive ? "hsl(var(--primary) / 0.15)" : "transparent",
                  color: isActive ? "var(--primary)" : "var(--muted-foreground)",
                  transition: "all 0.2s ease",
                  whiteSpace: "nowrap",
                }}
              >
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* ── Tab Content ──────────────────────────────────────────────────── */}
      <div style={{ flex: 1, padding: "16px 16px 16px", overflowX: "hidden" }}>
        {/* Hidden file inputs — always rendered */}
        <input
          ref={cameraInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          style={{ display: "none" }}
          onChange={async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            setIsAnalyzing(true);
            setAnalysisError(null);
            setCaptureMode("camera");
            try {
              const reader = new FileReader();
              const base64 = await new Promise<string>((resolve, reject) => {
                reader.onload = () => {
                  const result = reader.result as string;
                  resolve(result.split(",")[1]);
                };
                reader.onerror = reject;
                reader.readAsDataURL(file);
              });
              const apiPromise = apiRequest("POST", "/api/food/analyze", {
                userId,
                mealType: selectedMealType,
                imageBase64: base64,
              });
              const timeoutPromise = new Promise<never>((_, reject) =>
                setTimeout(() => reject(new Error("Analysis timed out")), 30000)
              );
              const res = await Promise.race([apiPromise, timeoutPromise]);
              const data = await res.json();
              // Persist API result to localStorage
              if (data?.id) {
                persistFoodLogLocally(userId, {
                  id: String(data.id),
                  loggedAt: data.loggedAt ?? new Date().toISOString(),
                  mealType: data.mealType ?? autoMealType(),
                  summary: data.summary ?? null,
                  totalCalories: data.totalCalories ?? null,
                  dominantMacro: data.dominantMacro ?? null,
                  foodItems: data.foodItems ?? null,
                  vitamins: data.vitamins ?? null,
                });
              }
              hapticSuccess();
              syncFoodToRailway({ summary: data.summary, totalCalories: data.totalCalories, dominantMacro: data.dominantMacro, foodItems: data.foodItems, mealType: selectedMealType });
              qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
              setCaptureMode("none");
            } catch {
              // Fallback: local estimation + localStorage
              try {
                const fallbackDesc = file.name.replace(/\.[^.]+$/, "").replace(/[-_]/g, " ") || "mixed meal";
                const local = estimateNutritionLocally(fallbackDesc);
                const entry: FoodLog = {
                  id: `local_${Date.now()}`,
                  loggedAt: new Date().toISOString(),
                  mealType: selectedMealType,
                  summary: local.summary,
                  totalCalories: local.total_calories,
                  dominantMacro: local.dominant_macro,
                  foodItems: local.food_items,
                  vitamins: null,
                };
                persistFoodLogLocally(userId, entry);
                try { await apiRequest("POST", "/api/food/log", { userId, ...entry }); } catch { /* ok */ }
                hapticSuccess();
                syncFoodToRailway({ summary: local.summary, totalCalories: local.total_calories, dominantMacro: local.dominant_macro, foodItems: local.food_items, mealType: selectedMealType });
                qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
                setCaptureMode("none");
              } catch (fallbackErr) {
                setAnalysisError("Could not analyze meal. Try describing it instead.");
                setCaptureMode("none");
              }
            } finally {
              setIsAnalyzing(false);
              if (cameraInputRef.current) cameraInputRef.current.value = "";
            }
          }}
        />
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          style={{ display: "none" }}
          onChange={async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            setIsAnalyzing(true);
            setAnalysisError(null);
            setCaptureMode("camera");
            try {
              const reader = new FileReader();
              const base64 = await new Promise<string>((resolve, reject) => {
                reader.onload = () => {
                  const result = reader.result as string;
                  resolve(result.split(",")[1]);
                };
                reader.onerror = reject;
                reader.readAsDataURL(file);
              });
              const apiPromise = apiRequest("POST", "/api/food/analyze", {
                userId,
                mealType: selectedMealType,
                imageBase64: base64,
              });
              const timeoutPromise = new Promise<never>((_, reject) =>
                setTimeout(() => reject(new Error("Analysis timed out")), 30000)
              );
              const res = await Promise.race([apiPromise, timeoutPromise]);
              const data = await res.json();
              // Persist API result to localStorage
              if (data?.id) {
                persistFoodLogLocally(userId, {
                  id: String(data.id),
                  loggedAt: data.loggedAt ?? new Date().toISOString(),
                  mealType: data.mealType ?? autoMealType(),
                  summary: data.summary ?? null,
                  totalCalories: data.totalCalories ?? null,
                  dominantMacro: data.dominantMacro ?? null,
                  foodItems: data.foodItems ?? null,
                  vitamins: data.vitamins ?? null,
                });
              }
              hapticSuccess();
              syncFoodToRailway({ summary: data.summary, totalCalories: data.totalCalories, dominantMacro: data.dominantMacro, foodItems: data.foodItems, mealType: selectedMealType });
              qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
              setCaptureMode("none");
            } catch {
              // Fallback: local estimation + localStorage
              try {
                const fallbackDesc = file.name.replace(/\.[^.]+$/, "").replace(/[-_]/g, " ") || "mixed meal";
                const local = estimateNutritionLocally(fallbackDesc);
                const entry: FoodLog = {
                  id: `local_${Date.now()}`,
                  loggedAt: new Date().toISOString(),
                  mealType: selectedMealType,
                  summary: local.summary,
                  totalCalories: local.total_calories,
                  dominantMacro: local.dominant_macro,
                  foodItems: local.food_items,
                  vitamins: null,
                };
                persistFoodLogLocally(userId, entry);
                try { await apiRequest("POST", "/api/food/log", { userId, ...entry }); } catch { /* ok */ }
                hapticSuccess();
                syncFoodToRailway({ summary: local.summary, totalCalories: local.total_calories, dominantMacro: local.dominant_macro, foodItems: local.food_items, mealType: selectedMealType });
                qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
                setCaptureMode("none");
              } catch (fallbackErr) {
                setAnalysisError("Could not analyze meal. Try describing it instead.");
                setCaptureMode("none");
              }
            } finally {
              setIsAnalyzing(false);
              if (fileInputRef.current) fileInputRef.current.value = "";
            }
          }}
        />

        <AnimatePresence mode="wait" custom={tabDirection}>
          {/* ═══════════ LOG TAB ═══════════ */}
          {activeTab === "log" && (
            <motion.div
              key="log"
              custom={tabDirection}
              variants={tabSlideVariants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
            >
              {/* Meal-Cognitive Insight (Issue #507) */}
              {mealCogInsight && captureMode === "none" && (
                <div style={{
                  background: "linear-gradient(135deg, rgba(124,58,237,0.05) 0%, rgba(6,182,212,0.05) 100%)",
                  border: "1px solid rgba(124,58,237,0.15)",
                  borderRadius: 16, padding: 14, marginBottom: 12,
                  boxShadow: "0 2px 16px rgba(0,0,0,0.04)",
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                    <BrainIcon style={{ width: 16, height: 16, color: "#7c3aed" }} />
                    <span style={{ fontSize: 12, fontWeight: 600, color: "var(--foreground)" }}>Food-Brain Insight</span>
                  </div>
                  <p style={{ fontSize: 12, color: "var(--muted-foreground)", margin: 0, lineHeight: 1.5 }}>
                    {mealCogInsight}
                  </p>
                </div>
              )}

              {/* Mindful Eating Prompt — appears when emotional eating detected */}
              {captureMode === "none" && voiceData && (voiceData.stress_index ?? 0) > 0.4 && (
                <div style={{
                  background: "var(--card)", border: "1px solid var(--border)",
                  borderRadius: 16, padding: 14, marginBottom: 12,
                  boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                    <span style={{ fontSize: 12, fontWeight: 600, color: "var(--foreground)" }}>Before you eat...</span>
                  </div>
                  <p style={{ fontSize: 12, color: "var(--muted-foreground)", margin: 0, lineHeight: 1.5 }}>
                    Your voice analysis shows elevated stress. Take a breath and ask yourself:
                    <strong style={{ color: "var(--foreground)" }}> Am I eating because I'm hungry, or because I'm feeling {voiceData.emotion ?? "stressed"}?</strong>
                  </p>
                  <p style={{ fontSize: 10, color: "var(--muted-foreground)", margin: "6px 0 0 0", fontStyle: "italic" }}>
                    No judgment -- just awareness. Log your meal either way.
                  </p>
                </div>
              )}

              {/* Meal type selector */}
              {captureMode === "none" && (
                <div style={{ display: "flex", gap: 6, marginBottom: 10 }}>
                  {[
                    { value: "breakfast", label: "🌅 Breakfast" },
                    { value: "lunch", label: "☀️ Lunch" },
                    { value: "dinner", label: "🌙 Dinner" },
                    { value: "snack", label: "🍿 Snack" },
                  ].map((opt) => (
                    <button
                      key={opt.value}
                      onClick={() => setSelectedMealType(opt.value)}
                      style={{
                        flex: 1, padding: "8px 4px", fontSize: 11, fontWeight: 600, borderRadius: 8, border: "none", cursor: "pointer",
                        background: selectedMealType === opt.value ? "var(--primary)" : "var(--muted)",
                        color: selectedMealType === opt.value ? "white" : "var(--muted-foreground)",
                        transition: "all 0.2s",
                      }}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              )}

              {/* Action Buttons — Scan (primary) + Describe + Barcode */}
              {captureMode === "none" && (
                <div style={{ marginBottom: 14 }}>
                  <button
                    onClick={async () => {
                      // On native: use Capacitor Camera plugin
                      try {
                        const { Capacitor } = await import("@capacitor/core");
                        if (Capacitor.isNativePlatform()) {
                          const { Camera, CameraResultType, CameraSource } = await import("@capacitor/camera");
                          const photo = await Camera.getPhoto({
                            quality: 80,
                            allowEditing: false,
                            resultType: CameraResultType.Base64,
                            source: CameraSource.Camera,
                          });
                          if (photo.base64String) {
                            setIsAnalyzing(true);
                            setAnalysisError(null);
                            setCaptureMode("camera");
                            try {
                              const res = await apiRequest("POST", "/api/food/analyze", {
                                userId,
                                mealType: selectedMealType,
                                imageBase64: photo.base64String,
                              });
                              const data = await res.json();
                              // Persist API result to localStorage for offline access
                              if (data?.id) {
                                persistFoodLogLocally(userId, {
                                  id: String(data.id),
                                  loggedAt: data.loggedAt ?? new Date().toISOString(),
                                  mealType: data.mealType ?? autoMealType(),
                                  summary: data.summary ?? null,
                                  totalCalories: data.totalCalories ?? null,
                                  dominantMacro: data.dominantMacro ?? null,
                                  foodItems: data.foodItems ?? null,
                                  vitamins: data.vitamins ?? null,
                                });
                              }
                              hapticSuccess();
                              syncFoodToRailway({ summary: data.summary, totalCalories: data.totalCalories, dominantMacro: data.dominantMacro, foodItems: data.foodItems, mealType: selectedMealType });
                              qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
                              setCaptureMode("none");
                            } catch {
                              // Fallback: save generic meal
                              const local = estimateNutritionLocally("photo meal");
                              const entry: FoodLog = {
                                id: `local_${Date.now()}`,
                                loggedAt: new Date().toISOString(),
                                mealType: selectedMealType,
                                summary: local.summary,
                                totalCalories: local.total_calories,
                                dominantMacro: local.dominant_macro,
                                foodItems: local.food_items,
                                vitamins: null,
                              };
                              persistFoodLogLocally(userId, entry);
                              try { await apiRequest("POST", "/api/food/log", { userId, ...entry }); } catch { /* ok */ }
                              hapticSuccess();
                              syncFoodToRailway({ summary: local.summary, totalCalories: local.total_calories, dominantMacro: local.dominant_macro, foodItems: local.food_items, mealType: selectedMealType });
                              qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
                              setCaptureMode("none");
                            } finally {
                              setIsAnalyzing(false);
                            }
                          }
                          return;
                        }
                      } catch { /* not native or Camera plugin unavailable */ }
                      // On web: use file input
                      cameraInputRef.current?.click();
                    }}
                    style={{
                      width: "100%", background: "var(--primary)",
                      color: "white", borderRadius: 20, padding: 14, fontSize: 14, fontWeight: 700,
                      border: "none", cursor: "pointer", marginBottom: 8,
                      display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                    }}
                  >
                    Scan Your Meal
                  </button>
                  <div style={{ display: "flex", gap: 8 }}>
                    <button
                      onClick={() => setCaptureMode("text")}
                      style={{
                        flex: 1, background: "var(--card)", color: "var(--foreground)", borderRadius: 12,
                        padding: 10, fontSize: 12, fontWeight: 500, border: "1px solid var(--border)", cursor: "pointer",
                      }}
                    >
                      Describe Meal
                    </button>
                    <button
                      onClick={() => setCaptureMode("barcode")}
                      style={{
                        flex: 1, background: "var(--card)", color: "var(--foreground)", borderRadius: 12,
                        padding: 10, fontSize: 12, fontWeight: 500, border: "1px solid var(--border)", cursor: "pointer",
                      }}
                    >
                      Scan Barcode
                    </button>
                  </div>
                </div>
              )}

              {/* Analyzing state */}
              {isAnalyzing && (
                <div style={{
                  background: "var(--card)", borderRadius: 16, border: "1px solid var(--border)",
                  padding: 20, marginBottom: 14, textAlign: "center",
                  boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
                }}>
                  <div style={{
                    width: 28, height: 28, border: "3px solid var(--primary)", borderTopColor: "transparent",
                    borderRadius: "50%", margin: "0 auto 8px", animation: "spin 0.8s linear infinite",
                  }} />
                  <p style={{ fontSize: 13, color: "var(--foreground)", margin: 0 }}>Analyzing your meal...</p>
                  <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
                </div>
              )}

              {/* Error */}
              {analysisError && (
                <div style={{
                  background: "var(--card)", borderRadius: 16, border: "1px solid #2d1f18",
                  padding: 14, marginBottom: 14, boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
                }}>
                  <p style={{ fontSize: 12, color: "#e879a8", margin: 0 }}>
                    {analysisError.includes("authentication") || analysisError.includes("API key") || analysisError.includes("not configured") || analysisError.includes("503")
                      ? "AI analysis unavailable. Try describing your meal instead -- basic analysis works without an API key."
                      : analysisError}
                  </p>
                  <button onClick={() => { setAnalysisError(null); setCaptureMode("none"); }}
                    style={{ marginTop: 8, fontSize: 12, color: "var(--muted-foreground)", background: "none", border: "none", cursor: "pointer" }}>
                    Try again
                  </button>
                </div>
              )}

              {/* Barcode mode */}
              {captureMode === "barcode" && (
                <BarcodePanel
                  onLog={handleBarcodeLog}
                  onCancel={() => setCaptureMode("none")}
                />
              )}

              {/* Describe mode — text input */}
              {captureMode === "text" && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
                  style={{
                    background: "var(--card)", borderRadius: 16, border: "1px solid var(--border)",
                    padding: 14, marginBottom: 14, boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
                  }}
                >
                  <p style={{ fontSize: 12, color: "var(--muted-foreground)", margin: "0 0 8px 0" }}>
                    What did you eat? Be specific for better accuracy.
                  </p>
                  <textarea
                    value={mealText}
                    onChange={(e) => setMealText(e.target.value)}
                    placeholder="e.g. rice bowl with grilled chicken, steamed vegetables, and soy sauce"
                    style={{
                      width: "100%", minHeight: 80, background: "var(--background)", color: "var(--foreground)",
                      border: "1px solid var(--border)", borderRadius: 10, padding: 12, fontSize: 13,
                      resize: "none", outline: "none", fontFamily: "inherit",
                    }}
                  />
                  <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
                    <button
                      onClick={() => { setCaptureMode("none"); setMealText(""); }}
                      style={{
                        flex: 1, background: "transparent", color: "var(--muted-foreground)", borderRadius: 10,
                        padding: 10, fontSize: 13, border: "1px solid var(--border)", cursor: "pointer",
                      }}
                    >
                      Cancel
                    </button>
                    <button
                      disabled={!mealText.trim() || isAnalyzing}
                      onClick={async () => {
                        if (!mealText.trim()) return;
                        setIsAnalyzing(true);
                        setAnalysisError(null);
                        try {
                          const apiPromise = apiRequest("POST", "/api/food/analyze", {
                            userId,
                            mealType: selectedMealType,
                            textDescription: mealText.trim(),
                          });
                          const timeoutPromise = new Promise<never>((_, reject) =>
                            setTimeout(() => reject(new Error("Analysis timed out")), 30000)
                          );
                          const res = await Promise.race([apiPromise, timeoutPromise]);
                          const data = await res.json();
                          // Persist API result to localStorage
                          if (data?.id) {
                            persistFoodLogLocally(userId, {
                              id: String(data.id),
                              loggedAt: data.loggedAt ?? new Date().toISOString(),
                              mealType: data.mealType ?? autoMealType(),
                              summary: data.summary ?? null,
                              totalCalories: data.totalCalories ?? null,
                              dominantMacro: data.dominantMacro ?? null,
                              foodItems: data.foodItems ?? null,
                              vitamins: data.vitamins ?? null,
                            });
                          }
                          hapticSuccess();
                          syncFoodToRailway({ summary: data.summary, totalCalories: data.totalCalories, dominantMacro: data.dominantMacro, foodItems: data.foodItems, mealType: selectedMealType });
                          setMealText("");
                          qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
                          setCaptureMode("none");
                        } catch {
                          // Fallback: local estimation + localStorage
                          try {
                            const local = estimateNutritionLocally(mealText.trim());
                            const entry: FoodLog = {
                              id: `local_${Date.now()}`,
                              loggedAt: new Date().toISOString(),
                              mealType: selectedMealType,
                              summary: local.summary,
                              totalCalories: local.total_calories,
                              dominantMacro: local.dominant_macro,
                              foodItems: local.food_items,
                              vitamins: null,
                            };
                            persistFoodLogLocally(userId, entry);
                            try { await apiRequest("POST", "/api/food/log", { userId, ...entry }); } catch { /* ok */ }
                            hapticSuccess();
                            syncFoodToRailway({ summary: local.summary, totalCalories: local.total_calories, dominantMacro: local.dominant_macro, foodItems: local.food_items, mealType: selectedMealType });
                            setMealText("");
                            qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
                            setCaptureMode("none");
                          } catch {
                            setAnalysisError("Could not log meal. Please try again.");
                            setCaptureMode("none");
                          }
                        } finally {
                          setIsAnalyzing(false);
                        }
                      }}
                      style={{
                        flex: 1, background: mealText.trim() ? "var(--primary)" : "var(--muted)",
                        color: mealText.trim() ? "white" : "var(--muted-foreground)", borderRadius: 10,
                        padding: 10, fontSize: 13, fontWeight: 600, border: "none", cursor: "pointer",
                      }}
                    >
                      {isAnalyzing ? "Analyzing..." : "Log Meal"}
                    </button>
                  </div>
                </motion.div>
              )}

              {/* Today's Meals */}
              <div style={{ marginBottom: 16 }}>
                <div style={{
                  fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
                  textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 8,
                }}>
                  Today's Meals
                </div>
                <div style={{
                  background: "var(--card)", borderRadius: 16, border: "1px solid var(--border)",
                  overflow: "hidden", boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
                }}>
                  {todayLogs.length === 0 ? (
                    <div style={{ padding: "24px 16px", textAlign: "center" }}>
                      <div style={{ fontSize: 20, fontWeight: 700, color: "var(--muted-foreground)", marginBottom: 4 }}>0 calories</div>
                      <div style={{ fontSize: 12, color: "var(--muted-foreground)", marginBottom: 8 }}>No meals logged today</div>
                      <div style={{ fontSize: 11, color: "var(--muted-foreground)", opacity: 0.7 }}>Tap + to log your first meal</div>
                    </div>
                  ) : (
                    todayLogs.map((log, idx) => (
                      <MealCard
                        key={log.id}
                        log={log}
                        isLast={idx === todayLogs.length - 1}
                        onToggleFavorite={handleToggleFavorite}
                        isFav={isFavorite(log.id, favorites)}
                      />
                    ))
                  )}
                </div>
              </div>

              {/* Food score removed from always-visible area — shows when you tap a meal to expand */}

              {/* Daily Running Totals + Food Quality Score (Tasks 1 & 3) */}
              {todayLogs.length > 0 && (
                <div style={{
                  background: "var(--card)", border: "1px solid var(--border)",
                  borderRadius: 16, padding: 14, marginBottom: 16,
                  boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
                }}>
                  <div style={{
                    display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10,
                  }}>
                    <div style={{
                      fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
                      textTransform: "uppercase", letterSpacing: "0.5px",
                    }}>
                      Daily Totals
                    </div>
                    {/* Food Quality Score badge */}
                    <div style={{
                      display: "flex", alignItems: "center", gap: 6,
                      padding: "3px 10px", borderRadius: 10,
                      background: foodQualityScore >= 70 ? "rgba(6, 182, 212, 0.12)"
                        : foodQualityScore >= 40 ? "rgba(212, 160, 23, 0.12)"
                        : "rgba(232, 121, 168, 0.12)",
                    }}>
                      <span style={{
                        fontSize: 16, fontWeight: 700,
                        color: foodQualityScore >= 70 ? "#06b6d4" : foodQualityScore >= 40 ? "#d4a017" : "#e879a8",
                      }}>
                        {foodQualityScore}
                      </span>
                      <span style={{
                        fontSize: 9, fontWeight: 600,
                        color: foodQualityScore >= 70 ? "#06b6d4" : foodQualityScore >= 40 ? "#d4a017" : "#e879a8",
                      }}>
                        Quality
                      </span>
                    </div>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8, textAlign: "center" }}>
                    <div>
                      <div style={{ fontSize: 18, fontWeight: 700, color: "#e8b94a" }}>{totalCalories}</div>
                      <div style={{ fontSize: 9, color: "var(--muted-foreground)" }}>kcal</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 18, fontWeight: 700, color: "#e879a8" }}>{Math.round(totalProtein)}g</div>
                      <div style={{ fontSize: 9, color: "var(--muted-foreground)" }}>protein</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 18, fontWeight: 700, color: "#0891b2" }}>{Math.round(totalCarbs)}g</div>
                      <div style={{ fontSize: 9, color: "var(--muted-foreground)" }}>carbs</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 18, fontWeight: 700, color: "#7c3aed" }}>{Math.round(totalFat)}g</div>
                      <div style={{ fontSize: 9, color: "var(--muted-foreground)" }}>fat</div>
                    </div>
                  </div>
                  <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 8, textAlign: "center" }}>
                    {todayLogs.length} meal{todayLogs.length !== 1 ? "s" : ""} logged today
                  </div>
                </div>
              )}

              {/* Favorites + Recent Meals */}
              {captureMode === "none" && (recentMeals.length > 0 || favorites.length > 0) && (
                <div style={{ marginBottom: 16 }}>
                  {/* Favorites section */}
                  {favorites.length > 0 && (
                    <>
                      <div style={{
                        fontSize: 11, fontWeight: 600, color: "#e8b94a",
                        textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 6,
                      }}>
                        Favorites
                      </div>
                      <div style={{ display: "flex", gap: 8, overflowX: "auto", paddingBottom: 8, marginBottom: 8 }}>
                        {favorites.slice(0, 5).map((fav) => (
                          <button
                            key={fav.id}
                            onClick={() => handleRelog(fav.summary, fav.foodItems)}
                            disabled={isAnalyzing}
                            style={{
                              flexShrink: 0, background: "var(--card)", border: "1px solid var(--border)",
                              borderRadius: 12, padding: "8px 12px", cursor: "pointer", minWidth: 120, maxWidth: 160,
                              textAlign: "left", boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
                            }}
                          >
                            <div style={{ fontSize: 11, fontWeight: 500, color: "var(--foreground)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                              {fav.summary}
                            </div>
                            <div style={{ fontSize: 9, color: "var(--muted-foreground)", marginTop: 2 }}>
                              {fav.totalCalories} kcal -- Tap to re-log
                            </div>
                          </button>
                        ))}
                      </div>
                    </>
                  )}

                  {/* Recent meals */}
                  {recentMeals.length > 0 && (
                    <>
                      <div style={{
                        fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
                        textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 6,
                      }}>
                        Recent Meals
                      </div>
                      <div style={{ display: "flex", gap: 8, overflowX: "auto", paddingBottom: 4 }}>
                        {recentMeals.map((meal) => (
                          <button
                            key={meal.id}
                            onClick={() => handleRelog(meal.summary ?? "Meal", meal.foodItems ?? [])}
                            disabled={isAnalyzing}
                            style={{
                              flexShrink: 0, background: "var(--card)", border: "1px solid var(--border)",
                              borderRadius: 12, padding: "8px 12px", cursor: "pointer", minWidth: 120, maxWidth: 160,
                              textAlign: "left", boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
                            }}
                          >
                            <div style={{ fontSize: 11, fontWeight: 500, color: "var(--foreground)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                              {meal.summary ?? "Meal"}
                            </div>
                            <div style={{ fontSize: 9, color: "var(--muted-foreground)", marginTop: 2 }}>
                              {meal.totalCalories ?? 0} kcal -- Tap to re-log
                            </div>
                          </button>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )}
            </motion.div>
          )}

          {/* ═══════════ VITAMINS TAB ═══════════ */}
          {activeTab === "vitamins" && (
            <motion.div
              key="vitamins"
              custom={tabDirection}
              variants={tabSlideVariants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
            >
              <div style={{
                background: "var(--card)", border: "1px solid var(--border)",
                borderRadius: 16, padding: 16, marginBottom: 16,
                boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
              }}>
                <SectionErrorBoundary label="Vitamin Tracker">
                  <VitaminTracker todayLogs={todayLogs} supplementVitamins={supplementVitamins} />
                </SectionErrorBoundary>
              </div>

              <div style={{
                background: "var(--card)", border: "1px solid var(--border)",
                borderRadius: 16, padding: 16,
                boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
              }}>
                <SectionErrorBoundary label="GLP-1 Tracker">
                  <Glp1Tracker todayLogs={todayLogs} totalProtein={totalProtein} totalCalories={totalCalories} />
                </SectionErrorBoundary>
              </div>

              {/* Glucose Section (conditional) */}
              <SectionErrorBoundary label="Glucose">
                <GlucoseSection userId={userId} />
              </SectionErrorBoundary>
            </motion.div>
          )}

          {/* ═══════════ SUPPLEMENTS TAB ═══════════ */}
          {activeTab === "supplements" && (
            <motion.div
              key="supplements"
              custom={tabDirection}
              variants={tabSlideVariants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
            >
              <div style={{
                background: "var(--card)", border: "1px solid var(--border)",
                borderRadius: 16, padding: 16,
                boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
              }}>
                <SectionErrorBoundary label="Supplement Tracker">
                  <SupplementTracker onSupplementsChange={handleSupplementsChange} />
                </SectionErrorBoundary>
              </div>

              {/* GLP-1 Injection Tracker (Task 4) */}
              <div style={{
                background: "var(--card)", border: "1px solid var(--border)",
                borderRadius: 16, padding: 16, marginTop: 16,
                boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
              }}>
                <SectionErrorBoundary label="GLP-1 Injection Tracker">
                  <Glp1InjectionTracker />
                </SectionErrorBoundary>
              </div>
            </motion.div>
          )}

          {/* ═══════════ INSIGHTS TAB ═══════════ */}
          {activeTab === "insights" && (
            <motion.div
              key="insights"
              custom={tabDirection}
              variants={tabSlideVariants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
            >
              {/* ── Brain-Food Story card ─────────────────────────────────── */}
              {(() => {
                // Compute brain-food headline from today's data
                const hasLogs = todayLogs.length > 0;
                const proteinPct = hasLogs ? Math.round((totalProtein / 130) * 100) : 0;
                const stressHigh = (voiceData?.stress_index ?? 0) > 0.5;
                const focusFromFood = mealCogInsight;

                let headline = "Log a meal to see your brain-food story";
                let body = "Every meal affects your focus, stress, and mood for the next 2-4 hours. We track the connection so you can optimize.";
                let action = "Tap 'Log' and scan or describe your next meal.";

                if (hasLogs) {
                  if (focusFromFood && focusFromFood.includes("dip")) {
                    headline = "Post-meal focus dip detected in your pattern";
                    body = `${focusFromFood} High-carb meals spike blood sugar then crash it — this is likely the culprit.`;
                    action = "Add 10g protein to your next meal to flatten the focus curve.";
                  } else if (focusFromFood && focusFromFood.includes("boost")) {
                    headline = "Your meals are fueling sharp focus";
                    body = `${focusFromFood} This pattern suggests your meal timing and composition are working well.`;
                    action = "Keep logging to confirm this pattern over the next 3 days.";
                  } else if (stressHigh) {
                    headline = "Stress is high — your food can help or hurt";
                    body = `At ${Math.round((voiceData?.stress_index ?? 0) * 100)}% stress, your body craves quick energy (sugar). Magnesium-rich foods calm the nervous system instead.`;
                    action = "Try dark chocolate, nuts, or leafy greens with your next meal.";
                  } else if (proteinPct >= 80) {
                    headline = "Strong protein day — focus window is open";
                    body = `${Math.round(totalProtein)}g protein gives your brain steady dopamine and norepinephrine precursors. Your cognitive peak window is now.`;
                    action = "Use the next 2 hours for your hardest mental work.";
                  } else if (proteinPct < 40) {
                    headline = "Protein below target — focus may feel flat";
                    body = `${Math.round(totalProtein)}g so far (${proteinPct}% of 130g goal). Low protein means fewer neurotransmitter precursors — this shows up as brain fog.`;
                    action = "Add eggs, Greek yogurt, or fish to your next meal.";
                  } else {
                    headline = `${todayLogs.length} meal${todayLogs.length > 1 ? "s" : ""} tracked — brain impact computed`;
                    body = `${Math.round(totalCalories)} kcal, ${Math.round(totalProtein)}g protein today. Your food quality score is ${foodQualityScore}/100 — ${foodQualityScore >= 70 ? "excellent brain fuel" : foodQualityScore >= 40 ? "decent, with room to optimize" : "below optimal — your focus and mood will reflect this"}.`;
                    action = foodQualityScore >= 70 ? "Maintain this pattern through dinner." : "Boost quality: add vegetables and reduce processed carbs at your next meal.";
                  }
                }

                return (
                  <div
                    style={{
                      borderRadius: 20,
                      background: "linear-gradient(135deg, rgba(124,58,237,0.10), rgba(6,182,212,0.06))",
                      border: "1px solid rgba(124,58,237,0.22)",
                      padding: 18,
                      marginBottom: 16,
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10 }}>
                      <BrainIcon style={{ width: 12, height: 12, color: "#7c3aed" }} />
                      <span style={{
                        fontSize: 10, fontWeight: 700, textTransform: "uppercase",
                        letterSpacing: "0.08em", color: "#7c3aed",
                      }}>
                        Brain-Food Story
                      </span>
                    </div>
                    <p style={{ fontSize: 14, fontWeight: 700, color: "var(--foreground)", lineHeight: 1.35, margin: "0 0 8px" }}>
                      {headline}
                    </p>
                    <p style={{ fontSize: 12, color: "var(--muted-foreground)", lineHeight: 1.55, margin: "0 0 12px" }}>
                      {body}
                    </p>
                    <div style={{
                      display: "flex", alignItems: "flex-start", gap: 8,
                      borderRadius: 12, padding: "10px 12px",
                      background: "rgba(124,58,237,0.10)",
                      border: "1px solid rgba(124,58,237,0.18)",
                    }}>
                      <span style={{ fontSize: 10, fontWeight: 800, color: "#7c3aed", marginTop: 1, letterSpacing: "0.04em" }}>→</span>
                      <p style={{ fontSize: 12, fontWeight: 600, color: "#7c3aed", margin: 0, lineHeight: 1.4 }}>
                        {action}
                      </p>
                    </div>
                  </div>
                );
              })()}

              {/* ── 7-day nutrition trend chart ────────────────────────── */}
              {nutritionTrendData.some(d => d.cal !== null) && (
                <div style={{
                  background: "var(--card)", border: "1px solid var(--border)",
                  borderRadius: 16, padding: 14, marginBottom: 16,
                  boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
                }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                    <div>
                      <div style={{ fontSize: 12, fontWeight: 600, color: "var(--foreground)" }}>7-Day Nutrition Trend</div>
                      <div style={{ fontSize: 10, color: "var(--muted-foreground)" }}>Calories & protein · daily totals</div>
                    </div>
                    <div style={{ display: "flex", gap: 12 }}>
                      {[{ label: "Calories", color: "#e8b94a" }, { label: "Protein g", color: "#e879a8" }].map(({ label, color }) => (
                        <div key={label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                          <div style={{ width: 6, height: 6, borderRadius: "50%", background: color }} />
                          <span style={{ fontSize: 9, color: "var(--muted-foreground)" }}>{label}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <ResponsiveContainer width="100%" height={120}>
                    <AreaChart data={nutritionTrendData} margin={{ left: -20, right: 4, top: 4, bottom: 0 }}>
                      <defs>
                        <linearGradient id="nutGradCal" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#e8b94a" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#e8b94a" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="nutGradProt" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#e879a8" stopOpacity={0.25} />
                          <stop offset="95%" stopColor="#e879a8" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="label" tick={{ fontSize: 9, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} />
                      <YAxis yAxisId="cal" orientation="left" tick={{ fontSize: 9, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} tickCount={3} />
                      <YAxis yAxisId="prot" orientation="right" tick={{ fontSize: 9, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} tickCount={3} />
                      <Tooltip
                        contentStyle={{
                          background: "rgba(15,15,25,0.92)", border: "1px solid rgba(255,255,255,0.1)",
                          borderRadius: 8, fontSize: 11, padding: "4px 10px",
                        }}
                        formatter={(v: number, name: string) => [name === "cal" ? `${v} kcal` : `${v}g`, name === "cal" ? "Calories" : "Protein"]}
                        labelStyle={{ color: "rgba(255,255,255,0.5)", fontSize: 10 }}
                      />
                      <ReferenceLine yAxisId="cal" y={2000} stroke="#e8b94a" strokeDasharray="4 3" strokeOpacity={0.35} strokeWidth={1} />
                      <ReferenceLine yAxisId="prot" y={PROTEIN_GOAL} stroke="#e879a8" strokeDasharray="4 3" strokeOpacity={0.35} strokeWidth={1} />
                      <Area yAxisId="cal" type="monotone" dataKey="cal" name="cal" stroke="#e8b94a" strokeWidth={1.5} fill="url(#nutGradCal)" dot={false} connectNulls />
                      <Area yAxisId="prot" type="monotone" dataKey="prot" name="prot" stroke="#e879a8" strokeWidth={1.5} fill="url(#nutGradProt)" dot={false} connectNulls />
                    </AreaChart>
                  </ResponsiveContainer>
                  <div style={{ fontSize: 10, color: "var(--muted-foreground)", textAlign: "center", marginTop: 4 }}>
                    Dashed lines = goals · 2000 kcal · {PROTEIN_GOAL}g protein
                  </div>
                </div>
              )}

              {/* Nutrition Score + Food Quality */}
              <div style={{
                background: "var(--card)", border: "1px solid var(--border)",
                borderRadius: 16, padding: 14, marginBottom: 16,
                display: "flex", gap: 14, alignItems: "flex-start",
                boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
              }}>
                <div style={{ flexShrink: 0 }}>
                  <ScoreGauge
                    value={todayLogs.length > 0 ? foodQualityScore : (scores?.nutritionScore ?? null)}
                    label="Food Quality"
                    color="nutrition"
                    size="sm"
                  />
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "var(--foreground)", marginBottom: 6 }}>
                    Food Quality Score
                  </div>
                  {(qualityIndicators.positives.length > 0 || qualityIndicators.negatives.length > 0) ? (
                    <>
                      {qualityIndicators.positives.map((p, i) => (
                        <div key={`p-${i}`} style={{ fontSize: 11, color: "#06b6d4", marginBottom: 2, display: "flex", alignItems: "center", gap: 4 }}>
                          <span style={{ fontSize: 8 }}>+</span> {p}
                        </div>
                      ))}
                      {qualityIndicators.negatives.map((n, i) => (
                        <div key={`n-${i}`} style={{ fontSize: 11, color: "#e879a8", marginBottom: 2, display: "flex", alignItems: "center", gap: 4 }}>
                          <span style={{ fontSize: 8 }}>-</span> {n}
                        </div>
                      ))}
                    </>
                  ) : (
                    <div style={{ fontSize: 11, color: "var(--muted-foreground)" }}>
                      Log meals to see quality contributors
                    </div>
                  )}
                </div>
              </div>

              {/* Craving Analysis Card */}
              {(() => {
                const cravingColors: Record<string, string> = {
                  Stress: "#e879a8",
                  Comfort: "#d4a017",
                  Mindful: "#0891b2",
                  Balanced: "#06b6d4",
                };
                const cravingColor = cravingColors[craving.label] ?? "#06b6d4";
                const allLabels = ["Stress", "Comfort", "Mindful", "Balanced"] as const;

                return (
                  <div style={{
                    background: "var(--card)", border: "1px solid var(--border)",
                    borderRadius: 16, padding: 16, marginBottom: 16,
                    boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
                  }}>
                    <div style={{
                      fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
                      textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 12,
                    }}>
                      Eating Pattern
                    </div>

                    {/* Active eating type badge */}
                    <div style={{
                      display: "flex", alignItems: "center", gap: 12, marginBottom: 14,
                    }}>
                      <div style={{
                        width: 48, height: 48, borderRadius: 20, flexShrink: 0,
                        background: `linear-gradient(135deg, ${cravingColor}, ${cravingColor}88)`,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        boxShadow: `0 4px 16px ${cravingColor}33`,
                      }}>
                        <span style={{ fontSize: 22, fontWeight: 800, color: "white" }}>
                          {craving.label.charAt(0)}
                        </span>
                      </div>
                      <div>
                        <div style={{
                          fontSize: 18, fontWeight: 700,
                          background: `linear-gradient(135deg, ${cravingColor}, ${cravingColor}cc)`,
                          WebkitBackgroundClip: "text",
                          WebkitTextFillColor: "transparent",
                        }}>
                          {craving.label} Eating
                        </div>
                        <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 2 }}>
                          Based on voice stress & valence analysis
                        </div>
                      </div>
                    </div>

                    {/* Visual bar showing position across eating types */}
                    <div style={{
                      display: "flex", gap: 4, marginBottom: 12,
                    }}>
                      {allLabels.map((label) => {
                        const color = cravingColors[label];
                        const isActive = label === craving.label;
                        return (
                          <div key={label} style={{ flex: 1, textAlign: "center" }}>
                            <div style={{
                              height: isActive ? 6 : 4,
                              borderRadius: 3,
                              background: isActive ? `linear-gradient(90deg, ${color}, ${color}cc)` : "var(--border)",
                              transition: "all 0.3s ease",
                              marginBottom: 4,
                              boxShadow: isActive ? `0 2px 8px ${color}44` : "none",
                            }} />
                            <span style={{
                              fontSize: 9, fontWeight: isActive ? 700 : 500,
                              color: isActive ? color : "var(--muted-foreground)",
                              transition: "all 0.3s ease",
                            }}>
                              {label}
                            </span>
                          </div>
                        );
                      })}
                    </div>

                    {/* Description */}
                    <p style={{
                      fontSize: 12, color: "var(--muted-foreground)", lineHeight: 1.5, margin: 0,
                      padding: "8px 10px", background: "var(--muted)", borderRadius: 10,
                    }}>
                      {craving.text.charAt(0).toUpperCase() + craving.text.slice(1)}.
                      {" "}Track meals to see how emotions shape your eating.
                    </p>
                  </div>
                );
              })()}

              {/* AI Nutrition Insights — glass-card chips */}
              {insights.length > 0 && (
                <div style={{ marginBottom: 16 }}>
                  <div style={{
                    fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
                    textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 8,
                  }}>
                    Personalized Insights
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {insights.map((insight, i) => {
                      // Assign category color based on content keywords
                      const isProtein = /protein/i.test(insight);
                      const isHydration = /water|hydrat/i.test(insight);
                      const isVeg = /vegetable|veg|plant|fiber/i.test(insight);
                      const isMacro = /carb|fat|calori/i.test(insight);
                      const color = isProtein ? "#e879a8" : isHydration ? "#06b6d4" : isVeg ? "#22c55e" : isMacro ? "#e8b94a" : "#a78bfa";
                      const label = isProtein ? "Protein" : isHydration ? "Hydration" : isVeg ? "Plants" : isMacro ? "Macros" : "Pattern";
                      return (
                        <div key={i} style={{
                          borderRadius: 14,
                          background: "var(--card)",
                          border: `1px solid ${color}28`,
                          padding: "10px 14px",
                          display: "flex", alignItems: "flex-start", gap: 10,
                        }}>
                          <span style={{
                            fontSize: 9, fontWeight: 700, textTransform: "uppercase",
                            letterSpacing: "0.06em", color, padding: "2px 7px",
                            borderRadius: 6, background: `${color}18`,
                            flexShrink: 0, marginTop: 1,
                          }}>
                            {label}
                          </span>
                          <span style={{ fontSize: 12, color: "var(--foreground)", lineHeight: 1.5 }}>
                            {insight}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Brain Macro Summary */}
              <div style={{
                borderRadius: 16,
                background: "var(--card)",
                border: "1px solid var(--border)",
                padding: 14,
                boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
              }}>
                <div style={{
                  fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
                  textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 12,
                }}>
                  Today vs Goals
                </div>
                {todayLogs.length > 0 ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    {[
                      { label: "Calories", value: totalCalories, goal: 2000, unit: "kcal", color: "#e8b94a" },
                      { label: "Protein", value: Math.round(totalProtein), goal: PROTEIN_GOAL, unit: "g", color: "#e879a8" },
                      { label: "Carbs", value: Math.round(totalCarbs), goal: 250, unit: "g", color: "#06b6d4" },
                      { label: "Fat", value: Math.round(totalFat), goal: 65, unit: "g", color: "#7c3aed" },
                    ].map(({ label, value, goal, unit, color }) => {
                      const pct = Math.min(Math.round((value / goal) * 100), 100);
                      return (
                        <div key={label}>
                          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                            <span style={{ fontSize: 12, fontWeight: 500, color: "var(--foreground)" }}>{label}</span>
                            <span style={{ fontSize: 12, color: "var(--muted-foreground)" }}>
                              <span style={{ color, fontWeight: 700 }}>{value}</span>
                              {" / "}{goal}{unit}
                            </span>
                          </div>
                          <div style={{ height: 6, background: "var(--border)", borderRadius: 3, overflow: "hidden" }}>
                            <div style={{
                              height: "100%", width: `${pct}%`, background: color,
                              borderRadius: 3, transition: "width 0.5s ease",
                            }} />
                          </div>
                        </div>
                      );
                    })}
                    <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 2 }}>
                      {totalProtein >= PROTEIN_GOAL
                        ? "Protein goal met — your brain has sufficient amino acid supply."
                        : `${Math.round(PROTEIN_GOAL - totalProtein)}g protein remaining — add a protein source to your next meal.`}
                    </div>
                  </div>
                ) : (
                  <div style={{ fontSize: 12, color: "var(--muted-foreground)", lineHeight: 1.5 }}>
                    Start logging meals to track your daily nutrition against goals.
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {/* ═══════════ HISTORY TAB ═══════════ */}
          {activeTab === "history" && (
            <motion.div
              key="history"
              custom={tabDirection}
              variants={tabSlideVariants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
            >
              {/* ── Calorie Trend Chart (7 days) ────────────────────────────── */}
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                style={{
                  background: "var(--card)", border: "1px solid var(--border)",
                  borderRadius: 16, padding: 16, marginBottom: 16,
                  boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
                }}
              >
                <div style={{
                  fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
                  textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 12,
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                }}>
                  <span>Calorie Trend (7 Days)</span>
                  <span style={{ fontSize: 12, fontWeight: 700, color: "#d4a017" }}>
                    {Math.round(calorieTrendData.reduce((s, d) => s + d.calories, 0) / 7)} avg
                  </span>
                </div>
                <div style={{ width: "100%", height: 180 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={calorieTrendData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                      <defs>
                        <linearGradient id="calorieGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#d4a017" stopOpacity={0.4} />
                          <stop offset="100%" stopColor="#d4a017" stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                      <XAxis
                        dataKey="label"
                        tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis
                        tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "var(--card)",
                          border: "1px solid var(--border)",
                          borderRadius: 10,
                          fontSize: 12,
                          color: "var(--foreground)",
                          boxShadow: "0 4px 16px rgba(0,0,0,0.12)",
                        }}
                        labelStyle={{ fontWeight: 600, color: "var(--foreground)" }}
                        formatter={(value: number) => [`${value} kcal`, "Calories"]}
                      />
                      <Area
                        type="monotone"
                        dataKey="calories"
                        stroke="#d4a017"
                        strokeWidth={2}
                        fill="url(#calorieGradient)"
                        dot={{ fill: "#d4a017", r: 3, strokeWidth: 0 }}
                        activeDot={{ r: 5, fill: "#d4a017", stroke: "var(--card)", strokeWidth: 2 }}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                {/* Goal reference line label */}
                <div style={{
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 6, marginTop: 8,
                }}>
                  <div style={{ width: 12, height: 2, background: "#d4a017", borderRadius: 1 }} />
                  <span style={{ fontSize: 10, color: "var(--muted-foreground)" }}>
                    Daily goal: {CAL_GOAL} kcal
                  </span>
                </div>
              </motion.div>

              {/* ── Food-Mood Correlation Card ──────────────────────────────── */}
              {foodMoodInsight && (
                <motion.div
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1, duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                  style={{
                    background: "var(--card)", border: "1px solid var(--border)",
                    borderRadius: 16, padding: 16, marginBottom: 16,
                    boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
                  }}
                >
                  <div style={{
                    fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
                    textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 10,
                  }}>
                    Food-Mood Connection
                  </div>
                  <div style={{
                    display: "flex", alignItems: "flex-start", gap: 10,
                  }}>
                    <div style={{
                      width: 36, height: 36, borderRadius: 12, flexShrink: 0,
                      background: "linear-gradient(135deg, #d4a017, #ea580c)",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      boxShadow: "0 4px 12px rgba(212, 160, 23, 0.2)",
                    }}>
                      <span style={{ fontSize: 16 }}>&#x1f9e0;</span>
                    </div>
                    <p style={{
                      fontSize: 12, color: "var(--foreground)", lineHeight: 1.6, margin: 0,
                    }}>
                      {foodMoodInsight}
                    </p>
                  </div>
                  {moodCorrelation && moodCorrelation.matchedWithEmotion > 0 && (
                    <div style={{
                      marginTop: 10, display: "flex", gap: 8, flexWrap: "wrap",
                    }}>
                      <span style={{
                        fontSize: 10, padding: "3px 10px", borderRadius: 10,
                        background: "rgba(212, 160, 23, 0.1)", color: "#d4a017", fontWeight: 500,
                      }}>
                        {moodCorrelation.matchedWithEmotion} meals with emotion data
                      </span>
                      {moodCorrelation.matchedWithMood > 0 && (
                        <span style={{
                          fontSize: 10, padding: "3px 10px", borderRadius: 10,
                          background: "rgba(8, 145, 178, 0.1)", color: "#0891b2", fontWeight: 500,
                        }}>
                          {moodCorrelation.matchedWithMood} with mood logs
                        </span>
                      )}
                    </div>
                  )}
                </motion.div>
              )}

              {/* ── Meal Timeline (last 7 days) ──────────────────────────── */}
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2, duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
              >
                <div style={{
                  fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
                  textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 10,
                }}>
                  Meal Timeline
                </div>

                {timelineByDate.size === 0 ? (
                  <div style={{
                    background: "var(--card)", border: "1px solid var(--border)",
                    borderRadius: 16, padding: "24px 16px", textAlign: "center",
                    boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
                  }}>
                    <div style={{ fontSize: 13, color: "var(--muted-foreground)" }}>
                      No meals logged in the last 7 days. Start logging to see your timeline.
                    </div>
                  </div>
                ) : (
                  Array.from(timelineByDate.entries()).map(([dateKey, dayLogs], groupIdx) => {
                    const d = new Date(dateKey);
                    const now = new Date();
                    const isToday_ = d.toDateString() === now.toDateString();
                    const yesterday = new Date(now);
                    yesterday.setDate(yesterday.getDate() - 1);
                    const isYesterday = d.toDateString() === yesterday.toDateString();
                    const dateLabel = isToday_ ? "Today" : isYesterday ? "Yesterday" : d.toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" });
                    const dayCalories = dayLogs.reduce((s, l) => s + (l.totalCalories ?? 0), 0);

                    return (
                      <motion.div
                        key={dateKey}
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: groupIdx * 0.06, duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
                        style={{ marginBottom: 14 }}
                      >
                        {/* Date header */}
                        <div style={{
                          display: "flex", alignItems: "center", justifyContent: "space-between",
                          marginBottom: 6, padding: "0 2px",
                        }}>
                          <span style={{ fontSize: 12, fontWeight: 600, color: "var(--foreground)" }}>
                            {dateLabel}
                          </span>
                          <span style={{ fontSize: 11, fontWeight: 600, color: "#d4a017" }}>
                            {dayCalories} kcal
                          </span>
                        </div>

                        {/* Meal cards for this date */}
                        <div style={{
                          background: "var(--card)", border: "1px solid var(--border)",
                          borderRadius: 16, overflow: "hidden",
                          boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
                        }}>
                          {dayLogs.map((log, idx) => {
                            const moodInfo = moodByFoodId.get(log.id as unknown as number);
                            const items = log.foodItems ?? [];
                            const mealP = items.reduce((s, f) => s + (f.protein_g ?? 0), 0);

                            return (
                              <div
                                key={log.id}
                                style={{
                                  display: "flex", alignItems: "center", padding: "10px 14px",
                                  borderBottom: idx < dayLogs.length - 1 ? "1px solid var(--border)" : "none",
                                }}
                              >
                                <span style={{ fontSize: 16, marginRight: 10, flexShrink: 0 }}>
                                  {MEAL_ICONS[log.mealType ?? "snack"] ?? "\uD83C\uDF7D\uFE0F"}
                                </span>
                                <div style={{ flex: 1, minWidth: 0 }}>
                                  <div style={{
                                    fontSize: 12, fontWeight: 500, color: "var(--foreground)",
                                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                                  }}>
                                    {log.summary ?? "Meal"}
                                  </div>
                                  <div style={{ display: "flex", gap: 8, marginTop: 2, alignItems: "center" }}>
                                    <span style={{ fontSize: 10, color: "var(--muted-foreground)" }}>
                                      {formatTime(log.loggedAt)}
                                    </span>
                                    {mealP > 0 && (
                                      <span style={{ fontSize: 10, color: "#e879a8" }}>
                                        {Math.round(mealP)}g protein
                                      </span>
                                    )}
                                    {moodInfo?.emotion && (
                                      <span style={{
                                        fontSize: 9, padding: "1px 6px", borderRadius: 6,
                                        background: "rgba(212, 160, 23, 0.1)", color: "#d4a017", fontWeight: 500,
                                      }}>
                                        {moodInfo.emotion}
                                      </span>
                                    )}
                                  </div>
                                </div>
                                {log.totalCalories != null && (
                                  <span style={{ fontSize: 12, fontWeight: 600, color: "#e8b94a", flexShrink: 0 }}>
                                    {log.totalCalories} kcal
                                  </span>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </motion.div>
                    );
                  })
                )}
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Last 5 Meals Logged ── */}
        <div style={{
          background: "var(--card)",
          borderRadius: 20,
          border: "1px solid var(--border)",
          padding: "16px 20px",
          boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
          marginTop: 16,
        }}>
          <RecentReadings
            storageKey={`ndw_food_logs_${userId}`}
            title="Last 5 Meals"
            maxEntries={5}
            emptyMessage="Log a meal to see your history here"
            renderEntry={(entry: any) => (
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <UtensilsCrossed style={{ width: 12, height: 12, color: "#d4a017", flexShrink: 0 }} />
                <span style={{ fontSize: 12, color: "var(--foreground)", flex: 1, lineHeight: 1.3 }}>
                  {entry.summary ?? "Meal"}
                </span>
                {entry.totalCalories != null && (
                  <span style={{ fontSize: 11, color: "#e8b94a", fontWeight: 600, flexShrink: 0 }}>
                    {entry.totalCalories} kcal
                  </span>
                )}
                <span style={{ fontSize: 10, color: "var(--muted-foreground)", flexShrink: 0 }}>
                  {formatTimeAgo(entry.loggedAt)}
                </span>
              </div>
            )}
          />
        </div>
      </div>
    </main>
  );
}
