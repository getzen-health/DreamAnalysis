import { useState, useMemo, useRef, useCallback, useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { resolveUrl, apiRequest } from "@/lib/queryClient";
import { getParticipantId } from "@/lib/participant";
import { hapticSuccess } from "@/lib/haptics";
import { useVoiceData } from "@/hooks/use-voice-data";
import { useScores } from "@/hooks/use-scores";
import { ScoreGauge } from "@/components/score-gauge";
import { lookupBarcode, type BarcodeProduct } from "@/lib/barcode-api";
import { cardVariants, listItemVariants } from "@/lib/animations";

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
    const raw = localStorage.getItem(FAVORITES_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveFavorites(favs: FavoriteMeal[]) {
  try {
    localStorage.setItem(FAVORITES_KEY, JSON.stringify(favs));
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

// ── Calorie Ring (SVG) ────────────────────────────────────────────────────────

const CAL_GOAL = 2000;
const PROTEIN_GOAL = 50;
const CARBS_GOAL = 275;
const FAT_GOAL = 78;

function CalorieRing({ calories }: { calories: number }) {
  const r = 58;
  const stroke = 8;
  const circ = 2 * Math.PI * r;
  const pct = Math.min(calories / CAL_GOAL, 1);
  const dash = circ * pct;
  const size = 140;
  const cx = size / 2;

  const remaining = CAL_GOAL - calories;
  const isOver = remaining < 0;
  const goalColor = isOver
    ? Math.abs(remaining) > 300
      ? "#e879a8" // way over — rose
      : "#d4a017" // slightly over — amber
    : "#22c55e"; // on track — green

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      style={{ display: "flex", flexDirection: "column", alignItems: "center", marginBottom: 4 }}
    >
      <div style={{ position: "relative", width: size, height: size }}>
        <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
          {/* Background track */}
          <circle
            cx={cx}
            cy={cx}
            r={r}
            fill="none"
            stroke="var(--border)"
            strokeWidth={stroke}
          />
          {/* Fill arc */}
          <circle
            cx={cx}
            cy={cx}
            r={r}
            fill="none"
            stroke="url(#calGrad)"
            strokeWidth={stroke}
            strokeDasharray={`${dash} ${circ}`}
            strokeLinecap="round"
          />
          <defs>
            <linearGradient id="calGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#d4a017" />
              <stop offset="100%" stopColor="#ea580c" />
            </linearGradient>
          </defs>
        </svg>
        {/* Center text */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <span style={{ fontSize: 28, fontWeight: 700, color: "var(--foreground)", lineHeight: 1 }}>
            {calories}
          </span>
          <span style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 2 }}>of {CAL_GOAL} kcal</span>
        </div>
      </div>
      {/* Goal indicator */}
      <div
        style={{
          marginTop: 8,
          fontSize: 12,
          fontWeight: 600,
          color: goalColor,
          textAlign: "center",
        }}
      >
        {isOver
          ? `${Math.abs(remaining)} kcal over`
          : remaining === 0
            ? "Goal reached"
            : `${remaining} kcal remaining`}
      </div>
    </motion.div>
  );
}

// ── Macro Progress Bar ────────────────────────────────────────────────────────

function MacroBar({ value, goal, color }: { value: number; goal: number; color: string }) {
  const pct = Math.min((value / goal) * 100, 100);
  return (
    <div
      style={{
        height: 3,
        background: "var(--border)",
        borderRadius: 2,
        marginTop: 6,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          width: `${pct}%`,
          height: "100%",
          background: color,
          borderRadius: 2,
          transition: "width 0.4s ease",
        }}
      />
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
    <div
      style={{
        height: 4,
        background: "var(--border)",
        borderRadius: 2,
        overflow: "hidden",
        flex: 1,
      }}
    >
      <div
        style={{
          height: "100%",
          width: `${pct}%`,
          background: color,
          borderRadius: 2,
          transition: "width 0.5s ease",
        }}
      />
    </div>
  );
}

function VitaminTracker({ todayLogs }: { todayLogs: FoodLog[] }) {
  const vitaminTotals = useMemo(() => {
    const totals: VitaminData = {
      vitamin_d_mcg: 0, vitamin_b12_mcg: 0, vitamin_c_mg: 0,
      iron_mg: 0, magnesium_mg: 0, zinc_mg: 0, omega3_g: 0,
    };
    for (const log of todayLogs) {
      // Use GPT vitamins data if available, otherwise estimate from food names
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
    return totals;
  }, [todayLogs]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.12 }}
      style={{
        background: "var(--card)",
        border: "1px solid var(--border)",
        borderRadius: 14,
        padding: 14,
        marginBottom: 16,
      }}
    >
      <div style={{
        fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
        textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 10,
      }}>
        Micronutrients
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {VITAMIN_TARGETS.map((vt) => {
          const val = vitaminTotals[vt.key];
          const pct = vt.daily > 0 ? (val / vt.daily) * 100 : 0;
          const color = pct >= 80 ? "#22c55e" : pct >= 50 ? "#d4a017" : "#e879a8";
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
      {todayLogs.length === 0 && (
        <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 8 }}>
          Log meals to track micronutrients
        </div>
      )}
    </motion.div>
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

  // Score components (0-25 each, total 0-100)
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

  const scoreColor = glp1.score >= 70 ? "#22c55e" : glp1.score >= 40 ? "#0891b2" : "#94a3b8";

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.15 }}
      style={{
        background: "var(--card)",
        border: "1px solid var(--border)",
        borderRadius: 14,
        padding: 14,
        marginBottom: 16,
      }}
    >
      <div style={{
        fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
        textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 10,
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <span>GLP-1 Support</span>
        <span style={{
          fontSize: 18, fontWeight: 700, color: scoreColor,
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
            <div style={{ flex: 1, height: 3, background: "var(--border)", borderRadius: 2, overflow: "hidden" }}>
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
                fontSize: 10,
                padding: "2px 8px",
                borderRadius: 10,
                background: "hsl(180 65% 50% / 0.12)",
                color: "#0891b2",
                fontWeight: 500,
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
    </motion.div>
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
        <span style={{ color: "var(--muted-foreground)", fontSize: 14, transition: "transform 0.2s", transform: expanded ? "rotate(90deg)" : "none" }}>›</span>
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
            <span><span style={{ color: "#7ba7d9" }}>P</span> {Math.round(totalP)}g</span>
            <span><span style={{ color: "#e8b94a" }}>C</span> {Math.round(totalC)}g</span>
            <span><span style={{ color: "#e87676" }}>F</span> {Math.round(totalF)}g</span>
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
            <span>{isFav ? "★" : "☆"}</span>
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
        background: "var(--card)", borderRadius: 14, border: "1px solid var(--border)",
        padding: 14, marginBottom: 14,
      }}
    >
      <div style={{ fontSize: 12, fontWeight: 600, color: "var(--primary)", marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ fontSize: 14 }}>📊</span>
        Barcode Lookup
      </div>
      <p style={{ fontSize: 11, color: "var(--muted-foreground)", margin: "0 0 8px 0" }}>
        Enter the barcode number from the package (UPC/EAN).
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
          {loading ? "..." : "Look Up"}
        </button>
      </div>

      {/* Product found */}
      {product && (
        <div style={{
          background: "var(--muted)", borderRadius: 10, padding: 12, marginBottom: 8,
        }}>
          <div style={{ display: "flex", gap: 10, marginBottom: 8 }}>
            {product.imageUrl && (
              <img
                src={product.imageUrl}
                alt={product.name}
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
              <div style={{ fontSize: 12, fontWeight: 600, color: "#3b82f6" }}>{product.protein_g}g</div>
              <div style={{ fontSize: 9, color: "var(--muted-foreground)" }}>protein</div>
            </div>
            <div style={{ background: "var(--card)", borderRadius: 6, padding: "4px 0" }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#d4a017" }}>{product.carbs_g}g</div>
              <div style={{ fontSize: 9, color: "var(--muted-foreground)" }}>carbs</div>
            </div>
            <div style={{ background: "var(--card)", borderRadius: 6, padding: "4px 0" }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#e879a8" }}>{product.fat_g}g</div>
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
              type="number"
              min="0.25"
              step="0.25"
              value={servings}
              onChange={(e) => setServings(e.target.value)}
              style={{
                width: 60, background: "var(--card)", color: "var(--foreground)",
                border: "1px solid var(--border)", borderRadius: 6, padding: "4px 8px",
                fontSize: 12, textAlign: "center", outline: "none",
              }}
            />
          </div>
          <button
            onClick={handleLog}
            style={{
              width: "100%", background: "var(--primary)", color: "white",
              borderRadius: 10, padding: 10, fontSize: 13, fontWeight: 600,
              border: "none", cursor: "pointer",
            }}
          >
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

      <button
        onClick={onCancel}
        style={{
          width: "100%", background: "transparent", color: "var(--muted-foreground)",
          borderRadius: 10, padding: 8, fontSize: 12, border: "1px solid var(--border)",
          cursor: "pointer",
        }}
      >
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
      ? "#22c55e"
      : glucose > 120 && glucose <= 140
        ? "#d4a017"
        : "#e879a8";

  const trendArrow = glucoseData?.trend === "rising"
    ? "↑"
    : glucoseData?.trend === "falling"
      ? "↓"
      : "→";

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.15 }}
      style={{
        background: "var(--card)",
        border: "1px solid var(--border)",
        borderRadius: 14,
        padding: 14,
        marginBottom: 16,
      }}
    >
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
    </motion.div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function Nutrition() {
  const userId = getParticipantId();
  const qc = useQueryClient();
  const [captureMode, setCaptureMode] = useState<"none" | "camera" | "text" | "barcode">("none");
  const [mealText, setMealText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const [favorites, setFavorites] = useState<FavoriteMeal[]>(loadFavorites);

  const { scores } = useScores(userId);

  const { data: logs } = useQuery<FoodLog[]>({
    queryKey: ["/api/food/logs", userId],
    queryFn: async () => {
      const res = await fetch(resolveUrl(`/api/food/logs/${userId}`));
      if (!res.ok) return [];
      return res.json();
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
        mealType: autoMealType(),
        textDescription: summary,
      });
      await res.json();
      hapticSuccess();
      await new Promise((r) => setTimeout(r, 500));
      qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
    } catch (err) {
      setAnalysisError(err instanceof Error ? err.message : "Re-log failed");
    } finally {
      setIsAnalyzing(false);
    }
  }, [userId, qc]);

  // Handle barcode log
  const handleBarcodeLog = useCallback(async (items: FoodItem[], summary: string) => {
    setIsAnalyzing(true);
    setAnalysisError(null);
    setCaptureMode("none");
    try {
      const res = await apiRequest("POST", "/api/food/analyze", {
        userId,
        mealType: autoMealType(),
        textDescription: `${summary} - ${items.map((i) => `${i.name} (${i.calories} kcal, ${i.protein_g}g protein, ${i.carbs_g}g carbs, ${i.fat_g}g fat)`).join(", ")}`,
      });
      await res.json();
      hapticSuccess();
      await new Promise((r) => setTimeout(r, 500));
      qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
    } catch (err) {
      setAnalysisError(err instanceof Error ? err.message : "Barcode log failed");
    } finally {
      setIsAnalyzing(false);
    }
  }, [userId, qc]);

  return (
    <main
      style={{
        background: "var(--background)",
        minHeight: "100vh",
        padding: 16,
        paddingBottom: 100,
        color: "var(--foreground)",
        fontFamily: "Inter, system-ui, sans-serif",
      }}
    >
      {/* Header */}
      <motion.h1
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        style={{ fontSize: 18, fontWeight: 600, marginBottom: 20, marginTop: 4 }}
      >
        Nutrition
      </motion.h1>

      {/* Calorie Ring */}
      <CalorieRing calories={totalCalories} />

      {/* Macro Cards */}
      <motion.div
        initial="hidden"
        animate="visible"
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: 8,
          marginBottom: 16,
        }}
      >
        {/* Protein */}
        <motion.div
          custom={0}
          variants={cardVariants}
          style={{
            background: "var(--card)",
            borderRadius: 12,
            border: "1px solid var(--border)",
            padding: "12px 8px",
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: 16, fontWeight: 700, color: "#3b82f6" }}>
            {Math.round(totalProtein)}g
          </div>
          <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>Protein</div>
          <MacroBar value={totalProtein} goal={PROTEIN_GOAL} color="#3b82f6" />
        </motion.div>

        {/* Carbs */}
        <motion.div
          custom={1}
          variants={cardVariants}
          style={{
            background: "var(--card)",
            borderRadius: 12,
            border: "1px solid var(--border)",
            padding: "12px 8px",
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: 16, fontWeight: 700, color: "#d4a017" }}>
            {Math.round(totalCarbs)}g
          </div>
          <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>Carbs</div>
          <MacroBar value={totalCarbs} goal={CARBS_GOAL} color="#d4a017" />
        </motion.div>

        {/* Fat */}
        <motion.div
          custom={2}
          variants={cardVariants}
          style={{
            background: "var(--card)",
            borderRadius: 12,
            border: "1px solid var(--border)",
            padding: "12px 8px",
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: 16, fontWeight: 700, color: "#e879a8" }}>
            {Math.round(totalFat)}g
          </div>
          <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>Fat</div>
          <MacroBar value={totalFat} goal={FAT_GOAL} color="#e879a8" />
        </motion.div>
      </motion.div>

      {/* Vitamin Tracker */}
      <VitaminTracker todayLogs={todayLogs} />

      {/* GLP-1 Support Tracker */}
      <Glp1Tracker todayLogs={todayLogs} totalProtein={totalProtein} totalCalories={totalCalories} />

      {/* Nutrition Score + Food Quality */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
        style={{
          background: "var(--card)",
          border: "1px solid var(--border)",
          borderRadius: 14,
          padding: 14,
          marginBottom: 16,
          display: "flex",
          gap: 14,
          alignItems: "flex-start",
        }}
      >
        {/* Score gauge */}
        <div style={{ flexShrink: 0 }}>
          <ScoreGauge
            value={scores?.nutritionScore ?? null}
            label="Food Quality"
            color="nutrition"
            size="sm"
          />
        </div>
        {/* Contributors */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "var(--foreground)", marginBottom: 6 }}>
            Food Quality Score
          </div>
          {(qualityIndicators.positives.length > 0 || qualityIndicators.negatives.length > 0) ? (
            <>
              {qualityIndicators.positives.map((p, i) => (
                <div key={`p-${i}`} style={{ fontSize: 11, color: "#22c55e", marginBottom: 2, display: "flex", alignItems: "center", gap: 4 }}>
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
      </motion.div>

      {/* Glucose Section (conditional) */}
      <GlucoseSection userId={userId} />

      {/* Craving Analysis Card */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.15 }}
        style={{
          background: "var(--card)",
          border: "1px solid #2d2418",
          borderRadius: 14,
          padding: 14,
          marginBottom: 16,
        }}
      >
        <div
          style={{
            fontSize: 11,
            fontWeight: 600,
            color: "var(--primary)",
            marginBottom: 8,
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <span>🧠</span>
          <span>Craving Analysis</span>
        </div>
        <p style={{ fontSize: 13, color: "var(--foreground)", lineHeight: 1.5, margin: 0 }}>
          Right now you show signs of <strong style={{ color: "var(--foreground)" }}>{craving.text}</strong>.
          Track your meals to see how your emotional state shapes your eating patterns.
        </p>
      </motion.div>

      {/* Hidden file inputs */}
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
                resolve(result.split(",")[1]); // strip data:image/...;base64,
              };
              reader.onerror = reject;
              reader.readAsDataURL(file);
            });
            const res = await apiRequest("POST", "/api/food/analyze", {
              userId,
              mealType: autoMealType(),
              imageBase64: base64,
            });
            await res.json();
            hapticSuccess();
            try { localStorage.setItem("ndw_meal_logged", "true"); } catch {}
            await new Promise(r => setTimeout(r, 500));
            qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
            setCaptureMode("none");
          } catch (err) {
            setAnalysisError(err instanceof Error ? err.message : "Analysis failed");
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
            const res = await apiRequest("POST", "/api/food/analyze", {
              userId,
              mealType: autoMealType(),
              imageBase64: base64,
            });
            await res.json();
            hapticSuccess();
            await new Promise(r => setTimeout(r, 500));
            qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
            setCaptureMode("none");
          } catch (err) {
            setAnalysisError(err instanceof Error ? err.message : "Analysis failed");
          } finally {
            setIsAnalyzing(false);
            if (fileInputRef.current) fileInputRef.current.value = "";
          }
        }}
      />

      {/* Mindful Eating Prompt -- appears when emotional eating detected */}
      {captureMode === "none" && voiceData && (voiceData.stress_index ?? 0) > 0.4 && (
        <div style={{
          background: "var(--card)", border: "1px solid var(--border)",
          borderRadius: 14, padding: 14, marginBottom: 12,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <span style={{ fontSize: 16 }}>🧘</span>
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

      {/* Action Buttons -- Scan (primary) + Describe + Barcode */}
      {captureMode === "none" && (
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.2 }}
          style={{ marginBottom: 14 }}
        >
          {/* Primary: Camera scan -- large button */}
          <button
            onClick={() => cameraInputRef.current?.click()}
            style={{
              width: "100%", background: "var(--primary)",
              color: "white", borderRadius: 14, padding: 14, fontSize: 14, fontWeight: 700,
              border: "none", cursor: "pointer", marginBottom: 8,
              display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
            }}
          >
            Scan Your Meal
          </button>
          {/* Secondary row: Describe + Barcode */}
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
        </motion.div>
      )}

      {/* Analyzing state */}
      {isAnalyzing && (
        <div style={{
          background: "var(--card)", borderRadius: 14, border: "1px solid var(--border)",
          padding: 20, marginBottom: 14, textAlign: "center",
        }}>
          <div style={{ width: 28, height: 28, border: "3px solid var(--primary)", borderTopColor: "transparent",
            borderRadius: "50%", margin: "0 auto 8px", animation: "spin 0.8s linear infinite" }} />
          <p style={{ fontSize: 13, color: "var(--foreground)", margin: 0 }}>Analyzing your meal...</p>
          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </div>
      )}

      {/* Error */}
      {analysisError && (
        <div style={{
          background: "var(--card)", borderRadius: 14, border: "1px solid #2d1f18",
          padding: 14, marginBottom: 14,
        }}>
          <p style={{ fontSize: 12, color: "#e879a8", margin: 0 }}>
            {analysisError.includes("authentication") || analysisError.includes("API key") || analysisError.includes("not configured") || analysisError.includes("503")
              ? "AI-powered meal suggestions require an active API key. Please check your configuration."
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

      {/* Describe mode -- text input */}
      {captureMode === "text" && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
          style={{
            background: "var(--card)", borderRadius: 14, border: "1px solid var(--border)",
            padding: 14, marginBottom: 14,
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
                  const res = await apiRequest("POST", "/api/food/analyze", {
                    userId,
                    mealType: autoMealType(),
                    textDescription: mealText.trim(),
                  });
                  await res.json();
                  hapticSuccess();
                  setMealText("");
                  await new Promise(r => setTimeout(r, 500));
                  qc.invalidateQueries({ queryKey: ["/api/food/logs", userId] });
                  setCaptureMode("none");
                } catch (err) {
                  setAnalysisError(err instanceof Error ? err.message : "Analysis failed");
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

      {/* Recent Meals + Favorites Quick Re-log */}
      {captureMode === "none" && (recentMeals.length > 0 || favorites.length > 0) && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.25 }}
          style={{ marginBottom: 16 }}
        >
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
                {favorites.slice(0, 5).map((fav, i) => (
                  <button
                    key={fav.id}
                    onClick={() => handleRelog(fav.summary, fav.foodItems)}
                    disabled={isAnalyzing}
                    style={{
                      flexShrink: 0, background: "var(--card)", border: "1px solid var(--border)",
                      borderRadius: 10, padding: "8px 12px", cursor: "pointer", minWidth: 120, maxWidth: 160,
                      textAlign: "left",
                    }}
                  >
                    <div style={{ fontSize: 11, fontWeight: 500, color: "var(--foreground)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {fav.summary}
                    </div>
                    <div style={{ fontSize: 9, color: "var(--muted-foreground)", marginTop: 2 }}>
                      {fav.totalCalories} kcal · Tap to re-log
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
                {recentMeals.map((meal, i) => (
                  <button
                    key={meal.id}
                    onClick={() => handleRelog(meal.summary ?? "Meal", meal.foodItems ?? [])}
                    disabled={isAnalyzing}
                    style={{
                      flexShrink: 0, background: "var(--card)", border: "1px solid var(--border)",
                      borderRadius: 10, padding: "8px 12px", cursor: "pointer", minWidth: 120, maxWidth: 160,
                      textAlign: "left",
                    }}
                  >
                    <div style={{ fontSize: 11, fontWeight: 500, color: "var(--foreground)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {meal.summary ?? "Meal"}
                    </div>
                    <div style={{ fontSize: 9, color: "var(--muted-foreground)", marginTop: 2 }}>
                      {meal.totalCalories ?? 0} kcal · Tap to re-log
                    </div>
                  </button>
                ))}
              </div>
            </>
          )}
        </motion.div>
      )}

      {/* Today's Meals */}
      <div>
        <div
          style={{
            fontSize: 11,
            fontWeight: 600,
            color: "var(--muted-foreground)",
            textTransform: "uppercase",
            letterSpacing: "0.5px",
            marginBottom: 8,
          }}
        >
          Today's Meals
        </div>

        <div
          style={{
            background: "var(--card)",
            borderRadius: 14,
            border: "1px solid var(--border)",
            overflow: "hidden",
          }}
        >
          {todayLogs.length === 0 ? (
            <div
              style={{
                padding: "24px 16px",
                textAlign: "center",
                fontSize: 13,
                color: "var(--muted-foreground)",
              }}
            >
              Log your first meal to start tracking
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

      {/* AI Nutrition Insights */}
      {insights.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.3 }}
          style={{
            background: "var(--card)",
            border: "1px solid var(--border)",
            borderRadius: 14,
            padding: 14,
            marginTop: 16,
          }}
        >
          <div style={{
            fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
            textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 8,
          }}>
            Nutrition Insights
          </div>
          {insights.map((insight, i) => (
            <motion.div
              key={i}
              custom={i}
              initial="hidden"
              animate="visible"
              variants={listItemVariants}
              style={{
                fontSize: 12,
                color: "var(--foreground)",
                lineHeight: 1.5,
                padding: "6px 0",
                borderBottom: i < insights.length - 1 ? "1px solid var(--border)" : "none",
              }}
            >
              {insight}
            </motion.div>
          ))}
        </motion.div>
      )}
    </main>
  );
}
