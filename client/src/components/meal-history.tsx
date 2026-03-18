/**
 * MealHistory — lists past meals grouped by date.
 *
 * Features:
 *  - Grouped by date (Today / Yesterday / date string)
 *  - Each meal: thumbnail, food items list, total calories, time
 *  - Favorite toggle (star icon)
 *  - "Re-log" button that pre-fills food items from a favorited (or any) meal
 *
 * Data contract: the parent page owns the list fetching; this component
 * receives the array and callbacks for mutation.
 */

import { useState } from "react";
import {
  Star,
  RotateCcw,
  Flame,
  ChevronDown,
  ChevronRight,
  ImageIcon,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

// ── Types ──────────────────────────────────────────────────────────────────────

export interface MealHistoryFoodItem {
  name:      string;
  portion:   string;
  calories:  number;
  protein_g: number;
  carbs_g:   number;
  fat_g:     number;
  fiber_g?:  number;
}

export interface MealHistoryEntry {
  id:            string;
  userId?:       string;
  images:        string[] | null;
  foodItems:     MealHistoryFoodItem[] | null;
  totalCalories: number | null;
  totalProtein:  number | null;
  totalCarbs:    number | null;
  totalFat:      number | null;
  totalFiber:    number | null;
  mealType:      string | null;
  isFavorite:    boolean;
  createdAt:     string;
}

export interface MealHistoryProps {
  meals:           MealHistoryEntry[];
  onToggleFavorite?: (id: string, current: boolean) => void;
  onRelog?:          (items: MealHistoryFoodItem[], mealType: string | null) => void;
  className?:        string;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

const MEAL_ICONS: Record<string, string> = {
  breakfast: "🌅",
  lunch:     "☀️",
  dinner:    "🌙",
  snack:     "🍎",
};

const MACRO_COLOR: Record<string, string> = {
  carbs:    "border-amber-500/40 text-amber-400 bg-amber-500/10",
  protein:  "border-indigo-500/40 text-indigo-400 bg-indigo-500/10",
  fat:      "border-rose-500/40 text-rose-400 bg-rose-500/10",
  balanced: "border-cyan-500/40 text-cyan-400 bg-cyan-600/10",
};

function groupByDate(meals: MealHistoryEntry[]): [string, MealHistoryEntry[]][] {
  const today     = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);

  const groups = new Map<string, MealHistoryEntry[]>();
  for (const meal of meals) {
    const d = new Date(meal.createdAt);
    let label: string;
    if (d.toDateString() === today.toDateString())     label = "Today";
    else if (d.toDateString() === yesterday.toDateString()) label = "Yesterday";
    else label = d.toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" });

    if (!groups.has(label)) groups.set(label, []);
    groups.get(label)!.push(meal);
  }
  return Array.from(groups.entries());
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function dominantMacro(
  protein: number | null,
  carbs: number | null,
  fat: number | null
): string {
  const p = protein ?? 0;
  const c = carbs   ?? 0;
  const f = fat     ?? 0;
  const protCal = p * 4;
  const carbCal = c * 4;
  const fatCal  = f * 9;
  if (protCal === 0 && carbCal === 0 && fatCal === 0) return "balanced";
  const max = Math.max(protCal, carbCal, fatCal);
  if (max === protCal) return "protein";
  if (max === carbCal) return "carbs";
  return "fat";
}

// ── Sub-component: single meal card ──────────────────────────────────────────

interface MealCardProps {
  meal:            MealHistoryEntry;
  onToggleFavorite?: (id: string, current: boolean) => void;
  onRelog?:          (items: MealHistoryFoodItem[], mealType: string | null) => void;
}

function MealCard({ meal, onToggleFavorite, onRelog }: MealCardProps) {
  const [expanded, setExpanded] = useState(false);

  const thumbnail = meal.images?.[0] ?? null;
  const items     = meal.foodItems ?? [];
  const macro     = dominantMacro(meal.totalProtein, meal.totalCarbs, meal.totalFat);

  return (
    <div
      className="rounded-xl border border-border/50 bg-muted/20 overflow-hidden"
      style={{ contain: "content" }}
    >
      {/* Main row */}
      <div className="flex items-center gap-3 p-3">
        {/* Thumbnail */}
        <div className="w-12 h-12 rounded-lg bg-muted/40 shrink-0 overflow-hidden flex items-center justify-center border border-border/30">
          {thumbnail ? (
            <img
              src={thumbnail}
              alt="Meal thumbnail"
              className="w-full h-full object-cover"
              loading="lazy"
            />
          ) : (
            <ImageIcon className="w-5 h-5 text-muted-foreground/40" />
          )}
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-xs" aria-hidden="true">
              {MEAL_ICONS[meal.mealType ?? "snack"] ?? "🍽️"}
            </span>
            <p className="text-sm font-medium leading-snug truncate">
              {items.length > 0
                ? items.slice(0, 2).map(it => it.name).join(", ") +
                  (items.length > 2 ? ` +${items.length - 2} more` : "")
                : "Meal"}
            </p>
          </div>
          <p className="text-xs text-muted-foreground mt-0.5">
            {formatTime(meal.createdAt)}
            {meal.totalCalories ? ` · ${meal.totalCalories} kcal` : ""}
          </p>
        </div>

        {/* Macro badge */}
        <Badge
          variant="outline"
          className={`text-[10px] shrink-0 ${MACRO_COLOR[macro] ?? ""}`}
        >
          {macro}
        </Badge>

        {/* Expand toggle */}
        <button
          onClick={() => setExpanded(v => !v)}
          aria-label={expanded ? "Collapse meal details" : "Expand meal details"}
          className="text-muted-foreground hover:text-foreground transition-colors ml-1"
        >
          {expanded
            ? <ChevronDown className="h-4 w-4" />
            : <ChevronRight className="h-4 w-4" />}
        </button>
      </div>

      {/* Expanded: full item list + macros + actions */}
      {expanded && (
        <div className="border-t border-border/30 px-3 pb-3 pt-2 space-y-3">
          {/* Image strip (if multi-image) */}
          {(meal.images?.length ?? 0) > 1 && (
            <div className="flex gap-1.5 overflow-x-auto pb-1">
              {meal.images!.map((src, i) => (
                <img
                  key={i}
                  src={src}
                  alt={`Meal photo ${i + 1}`}
                  className="h-16 w-16 rounded object-cover shrink-0 border border-border/30"
                  loading="lazy"
                />
              ))}
            </div>
          )}

          {/* Food items */}
          {items.length > 0 && (
            <ul className="space-y-1">
              {items.map((item, i) => (
                <li key={i} className="flex justify-between items-baseline text-xs">
                  <span className="text-foreground/80 capitalize">{item.name}</span>
                  <span className="font-mono text-muted-foreground ml-2 shrink-0">
                    {Math.round(item.calories)} kcal · {item.portion}
                  </span>
                </li>
              ))}
            </ul>
          )}

          {/* Macro totals */}
          <div className="grid grid-cols-4 gap-1 text-center text-xs">
            <div className="rounded bg-muted/30 py-1">
              <p className="font-semibold text-orange-400">{meal.totalCalories ?? 0}</p>
              <p className="text-[10px] text-muted-foreground">kcal</p>
            </div>
            <div className="rounded bg-muted/30 py-1">
              <p className="font-semibold text-indigo-400">{meal.totalProtein ?? 0}g</p>
              <p className="text-[10px] text-muted-foreground">protein</p>
            </div>
            <div className="rounded bg-muted/30 py-1">
              <p className="font-semibold text-amber-400">{meal.totalCarbs ?? 0}g</p>
              <p className="text-[10px] text-muted-foreground">carbs</p>
            </div>
            <div className="rounded bg-muted/30 py-1">
              <p className="font-semibold text-rose-400">{meal.totalFat ?? 0}g</p>
              <p className="text-[10px] text-muted-foreground">fat</p>
            </div>
          </div>

          {/* Actions row */}
          <div className="flex gap-2">
            {/* Favorite toggle */}
            <Button
              variant="outline"
              size="sm"
              className={`flex-1 gap-1.5 ${meal.isFavorite ? "border-amber-500/50 text-amber-400" : ""}`}
              onClick={() => onToggleFavorite?.(meal.id, meal.isFavorite)}
              aria-label={meal.isFavorite ? "Remove from favorites" : "Add to favorites"}
            >
              <Star
                className={`h-3.5 w-3.5 ${meal.isFavorite ? "fill-amber-400 text-amber-400" : ""}`}
              />
              {meal.isFavorite ? "Favorited" : "Favorite"}
            </Button>

            {/* Re-log */}
            {items.length > 0 && (
              <Button
                variant="outline"
                size="sm"
                className="flex-1 gap-1.5"
                onClick={() => onRelog?.(items, meal.mealType)}
                aria-label="Re-log this meal"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Re-log
              </Button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export function MealHistory({
  meals,
  onToggleFavorite,
  onRelog,
  className,
}: MealHistoryProps) {
  if (meals.length === 0) {
    return (
      <div className={`text-center py-6 ${className ?? ""}`}>
        <Flame className="w-6 h-6 text-muted-foreground/30 mx-auto mb-2" />
        <p className="text-xs text-muted-foreground">No meal history yet.</p>
      </div>
    );
  }

  const groups = groupByDate(meals);

  return (
    <div className={`space-y-4 ${className ?? ""}`}>
      {groups.map(([dateLabel, dayMeals]) => (
        <section key={dateLabel}>
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            {dateLabel}
          </p>
          <div className="space-y-2">
            {dayMeals.map(meal => (
              <MealCard
                key={meal.id}
                meal={meal}
                onToggleFavorite={onToggleFavorite}
                onRelog={onRelog}
              />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
