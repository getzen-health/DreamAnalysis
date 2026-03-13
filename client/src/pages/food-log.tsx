import { getParticipantId } from "@/lib/participant";
import { useState, useRef, useEffect } from "react";
import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import { resolveUrl } from "@/lib/queryClient";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import {
  Camera,
  CheckCircle2,
  ImageIcon,
  Loader2,
  Utensils,
  Moon,
  Zap,
  TrendingUp,
  X,
  PenLine,
} from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import {
  MealHistory,
  type MealHistoryEntry,
  type MealHistoryFoodItem,
} from "@/components/meal-history";

// ── Types ─────────────────────────────────────────────────────────────────────

interface FoodItem {
  name: string;
  portion: string;
  calories: number;
  carbs_g: number;
  protein_g: number;
  fat_g: number;
}

interface FoodAnalysis {
  id: string;
  loggedAt: string;
  foodItems: FoodItem[];
  totalCalories: number;
  dominantMacro: string;
  glycemicImpact: string;
  moodImpact: string;
  dreamRelevance: string;
  summary: string;
}

interface FoodLog {
  id: string;
  loggedAt: string;
  mealType: string | null;
  summary: string | null;
  totalCalories: number | null;
  dominantMacro: string | null;
  glycemicImpact: string | null;
  foodItems: FoodItem[] | null;
}

type InputMode = "photo" | "text";
type FilterType = "all" | "breakfast" | "lunch" | "dinner" | "snack";

// ── Helpers ───────────────────────────────────────────────────────────────────

const GI_STYLE: Record<string, string> = {
  low:    "border-green-500/40 text-green-400 bg-green-500/10",
  medium: "border-amber-500/40 text-amber-400 bg-amber-500/10",
  high:   "border-rose-500/40 text-rose-400 bg-rose-500/10",
};

const MACRO_COLOR: Record<string, string> = {
  carbs:    "text-amber-400",
  protein:  "text-blue-400",
  fat:      "text-rose-400",
  balanced: "text-green-400",
};

const MEAL_TYPES = ["breakfast", "lunch", "dinner", "snack"] as const;
const MEAL_ICONS: Record<string, string> = {
  breakfast: "🌅",
  lunch:     "☀️",
  dinner:    "🌙",
  snack:     "🍎",
};
const MEAL_PROMPTS: Record<string, string> = {
  breakfast: "What did you have for breakfast?\ne.g. 2 scrambled eggs, whole-wheat toast with butter, orange juice",
  lunch:     "What did you eat for lunch?\ne.g. grilled chicken salad, whole-grain roll, sparkling water",
  dinner:    "What did you have for dinner?\ne.g. salmon with roasted vegetables, brown rice, glass of wine",
  snack:     "Any snacks today?\ne.g. apple with almond butter, handful of almonds, yogurt",
};
const MEAL_SUGGESTIONS: Record<string, string[]> = {
  breakfast: ["Oatmeal with berries", "Eggs & toast", "Smoothie bowl", "Avocado toast", "Cereal & milk", "Pancakes"],
  lunch:     ["Chicken salad", "Sandwich", "Rice bowl", "Soup & bread", "Pasta", "Burrito"],
  dinner:    ["Grilled chicken & veggies", "Pasta with sauce", "Stir fry & rice", "Curry & naan", "Salmon & salad", "Pizza"],
  snack:     ["Apple & peanut butter", "Mixed nuts", "Yogurt", "Protein bar", "Banana", "Cheese & crackers"],
};

function formatTime(iso: string) {
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatDate(iso: string) {
  const d = new Date(iso);
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  if (d.toDateString() === today.toDateString()) return "Today";
  if (d.toDateString() === yesterday.toDateString()) return "Yesterday";
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

function autoMealType(): string {
  const h = new Date().getHours();
  if (h >= 5 && h < 10) return "breakfast";
  if (h >= 11 && h < 15) return "lunch";
  if (h >= 17 && h < 22) return "dinner";
  return "snack";
}

// Resize image to max dimension and convert to base64
async function compressToBase64(file: File, maxPx = 800): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      const scale = Math.min(1, maxPx / Math.max(img.width, img.height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      const ctx = canvas.getContext("2d");
      if (!ctx) return reject(new Error("No canvas context"));
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      resolve(canvas.toDataURL("image/jpeg", 0.75).split(",")[1]);
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("Image load failed")); };
    img.src = url;
  });
}

// ── Main component ────────────────────────────────────────────────────────────

export default function FoodLog() {
  const USER_ID = getParticipantId();
  const { toast } = useToast();
  const qc = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  const [inputMode, setInputMode] = useState<InputMode>("photo");
  const [mealType, setMealType] = useState(autoMealType());
  const [filterType, setFilterType] = useState<FilterType>("all");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<FoodAnalysis | null>(null);

  // When a specific filter tab is selected, also switch the log form to that meal type
  useEffect(() => {
    setAnalysis(null);
    setDescription("");
    setPhotoDescription("");
    if (filterType !== "all") {
      setMealType(filterType);
    }
  }, [filterType]);

  // Photo mode state
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [photoDescription, setPhotoDescription] = useState("");
  const [pendingFile, setPendingFile] = useState<File | null>(null);

  // Text mode state
  const [description, setDescription] = useState("");

  // History
  const { data: history } = useQuery<FoodLog[]>({
    queryKey: ["/api/food/logs", USER_ID],
    queryFn: async () => {
      const res = await fetch(resolveUrl(`/api/food/logs/${USER_ID}`), { credentials: "include" });
      if (!res.ok) return [];
      return res.json();
    },
  });

  // ── Meal history (issues #367 + #378) ──────────────────────────────────────
  const { data: mealHistoryData } = useQuery<MealHistoryEntry[]>({
    queryKey: ["/api/meal-history", USER_ID],
    queryFn: async () => {
      const res = await fetch(resolveUrl(`/api/meal-history/${USER_ID}`), { credentials: "include" });
      if (!res.ok) return [];
      return res.json();
    },
  });

  const toggleFavoriteMutation = useMutation({
    mutationFn: async ({ id, current }: { id: string; current: boolean }) => {
      await apiRequest("PATCH", `/api/meal-history/${id}/favorite`, { isFavorite: !current });
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["/api/meal-history", USER_ID] }),
    onError: () => toast({ title: "Could not update favorite", variant: "destructive" }),
  });

  function handleToggleFavorite(id: string, current: boolean) {
    toggleFavoriteMutation.mutate({ id, current });
  }

  function handleRelog(items: MealHistoryFoodItem[], relogMealType: string | null) {
    // Pre-fill the text description from the food items list
    const desc = items.map(it => `${it.name} (${it.portion})`).join(", ");
    setInputMode("text");
    setDescription(desc);
    if (relogMealType) setMealType(relogMealType);
    setAnalysis(null);
    // Scroll to top so the user sees the pre-filled form
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  // ── Shared submit logic ──────────────────────────────────────────────────────
  async function submitAnalysis(payload: { imageBase64?: string; textDescription?: string }) {
    setIsAnalyzing(true);
    setAnalysis(null);
    try {
      const res = await apiRequest("POST", "/api/food/analyze", {
        userId: USER_ID,
        mealType,
        ...payload,
      });
      const data: FoodAnalysis = await res.json();
      setAnalysis(data);
      qc.invalidateQueries({ queryKey: ["/api/food/logs", USER_ID] });
      qc.invalidateQueries({ queryKey: ["/api/research/correlation", USER_ID] });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Analysis failed";
      toast({ title: "Could not analyze meal", description: msg, variant: "destructive" });
    } finally {
      setIsAnalyzing(false);
    }
  }

  // ── Photo mode handlers ──────────────────────────────────────────────────────
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setPendingFile(file);
    setMealType(autoMealType());
    setPhotoDescription("");
    setAnalysis(null);
  };

  const handlePhotoSubmit = async () => {
    if (!pendingFile && !photoDescription.trim()) return;
    const payload: { imageBase64?: string; textDescription?: string } = {};
    if (pendingFile) {
      try {
        payload.imageBase64 = await compressToBase64(pendingFile);
      } catch {
        // Image compression failed — continue with text-only
      }
    }
    if (photoDescription.trim()) {
      payload.textDescription = photoDescription.trim();
    }
    if (!payload.imageBase64 && !payload.textDescription) return;
    await submitAnalysis(payload);
  };

  const clearPhoto = () => {
    setPreviewUrl(null);
    setPendingFile(null);
    setPhotoDescription("");
    setAnalysis(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (cameraInputRef.current) cameraInputRef.current.value = "";
  };

  // ── Text mode handler ────────────────────────────────────────────────────────
  const handleTextSubmit = async () => {
    if (!description.trim()) return;
    await submitAnalysis({ textDescription: description.trim() });
  };

  const clearText = () => {
    setDescription("");
    setAnalysis(null);
  };

  // ── Mode switch ──────────────────────────────────────────────────────────────
  const switchMode = (mode: InputMode) => {
    setInputMode(mode);
    setAnalysis(null);
    setPreviewUrl(null);
    setPendingFile(null);
    setPhotoDescription("");
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (cameraInputRef.current) cameraInputRef.current.value = "";
    if (mode === "text") setDescription("");
  };

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="max-w-lg mx-auto py-4 px-4 space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3 pt-1">
        <Utensils className="w-5 h-5 text-amber-400 shrink-0" />
        <div>
          <h1 className="text-[18px] font-bold leading-tight">Meal Log</h1>
          <p className="text-[12px] text-muted-foreground">Food shapes mood &amp; dreams</p>
        </div>
      </div>

      {/* ── Input mode toggle ─────────────────────────────────────────────────── */}
      <div
        className="flex rounded-2xl p-1 gap-1"
        role="group"
        aria-label="Input method"
        style={{
          background: "hsl(222,28%,9%,0.7)",
          border: "1px solid hsl(220,18%,17%,0.6)",
        }}
      >
        <button
          onClick={() => switchMode("photo")}
          aria-label="Log meal by photo"
          aria-pressed={inputMode === "photo"}
          className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl text-[13px] font-semibold transition-colors ${
            inputMode === "photo"
              ? "bg-amber-500 text-white shadow-md"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <Camera className="w-3.5 h-3.5" aria-hidden="true" />
          Photo
        </button>
        <button
          onClick={() => switchMode("text")}
          aria-label="Log meal by text description"
          aria-pressed={inputMode === "text"}
          className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl text-[13px] font-semibold transition-colors ${
            inputMode === "text"
              ? "bg-amber-500 text-white shadow-md"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <PenLine className="w-3.5 h-3.5" aria-hidden="true" />
          Describe
        </button>
      </div>

      {/* ── Capture card ─────────────────────────────────────────────────────── */}
      <div
        className="rounded-2xl p-4 space-y-4"
        style={{
          background: "hsl(222,28%,9%,0.7)",
          border: "1px solid hsl(220,18%,17%,0.6)",
          boxShadow: "0 1px 3px hsl(222,30%,3%,0.4)",
        }}
      >

          {/* Hidden file inputs — one for gallery, one with capture for camera */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileChange}
          />
          <input
            ref={cameraInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            className="hidden"
            onChange={handleFileChange}
          />

          {/* Meal type selector */}
          <div className="flex gap-1.5" role="group" aria-label="Meal type">
            {MEAL_TYPES.map(t => (
              <button
                key={t}
                onClick={() => { setMealType(t); setDescription(""); setPhotoDescription(""); setAnalysis(null); }}
                aria-label={`Meal type: ${t.charAt(0).toUpperCase() + t.slice(1)}`}
                aria-pressed={mealType === t}
                className={`flex-1 py-1.5 rounded-xl text-[11px] font-semibold border transition-colors ${
                  mealType === t
                    ? "border-amber-500/60 bg-amber-500/15 text-amber-300"
                    : "border-border/40 bg-muted/10 text-muted-foreground hover:text-foreground"
                }`}
              >
                <span aria-hidden="true">{MEAL_ICONS[t]}</span> {t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>

          {/* ── PHOTO MODE ─────────────────────────────────────────────────── */}
          {inputMode === "photo" && (
            !previewUrl ? (
              <div className="space-y-3">
                {/* Large camera CTA — primary action on mobile */}
                <button
                  onClick={() => cameraInputRef.current?.click()}
                  aria-label="Take food photo"
                  className="w-full flex flex-col items-center justify-center gap-3 py-8 rounded-2xl border-2 border-dashed border-amber-500/35 bg-amber-500/5 active:bg-amber-500/12 transition-all active:scale-[0.99]"
                >
                  <div className="w-16 h-16 rounded-full bg-amber-500 flex items-center justify-center shadow-lg shadow-amber-500/25">
                    <Camera className="w-8 h-8 text-white" aria-hidden="true" />
                  </div>
                  <div className="text-center">
                    <p className="text-[15px] font-bold text-foreground">Take a Photo</p>
                    <p className="text-[12px] text-muted-foreground mt-0.5">Tap to open camera</p>
                  </div>
                </button>
                {/* Choose from Gallery — secondary action */}
                <button
                  onClick={() => fileInputRef.current?.click()}
                  aria-label="Choose food photo from gallery"
                  className="w-full flex items-center justify-center gap-2 py-3 rounded-xl border border-border/50 bg-muted/15 active:bg-muted/30 transition-colors"
                >
                  <ImageIcon className="w-4 h-4 text-muted-foreground" aria-hidden="true" />
                  <span className="text-[13px] font-medium text-muted-foreground">
                    Choose from Gallery
                  </span>
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="relative rounded-xl overflow-hidden">
                  <img src={previewUrl} alt="Meal preview" className="w-full h-48 object-cover" />
                  <button
                    onClick={clearPhoto}
                    className="absolute top-2 right-2 w-7 h-7 rounded-full bg-background/60 flex items-center justify-center hover:bg-background/80 transition-colors"
                  >
                    <X className="w-3.5 h-3.5 text-foreground" />
                  </button>
                  {isAnalyzing && (
                    <div className="absolute inset-0 bg-background/50 flex items-center justify-center gap-2">
                      <Loader2 className="w-5 h-5 animate-spin text-foreground" />
                      <span className="text-foreground text-sm font-medium">Analyzing…</span>
                    </div>
                  )}
                </div>
                {!isAnalyzing && !analysis && (
                  <div className="space-y-2">
                    <p className="text-xs text-muted-foreground">
                      Photo captured. Add a description to improve accuracy (optional):
                    </p>
                    <Textarea
                      placeholder={MEAL_PROMPTS[mealType] ?? "e.g. grilled chicken with rice and vegetables"}
                      value={photoDescription}
                      onChange={e => setPhotoDescription(e.target.value)}
                      className="min-h-[72px] resize-none text-sm"
                      disabled={isAnalyzing}
                    />
                    {/* Quick suggestion chips for photo mode */}
                    <div className="flex flex-wrap gap-1.5">
                      {(MEAL_SUGGESTIONS[mealType] ?? []).slice(0, 4).map(suggestion => (
                        <button
                          key={suggestion}
                          onClick={() => setPhotoDescription(prev => prev ? `${prev}, ${suggestion.toLowerCase()}` : suggestion)}
                          className="px-2 py-0.5 rounded-full text-[10px] border border-amber-500/30 bg-amber-500/5 text-amber-300 hover:bg-amber-500/15 transition-colors"
                        >
                          {suggestion}
                        </button>
                      ))}
                    </div>
                    <div className="flex gap-2">
                      <Button
                        className="flex-1 bg-amber-500 hover:bg-amber-600 text-white"
                        onClick={handlePhotoSubmit}
                        disabled={isAnalyzing || !pendingFile}
                      >
                        {isAnalyzing ? (
                          <><Loader2 className="w-4 h-4 animate-spin mr-2" />Analyzing…</>
                        ) : (
                          "Log meal"
                        )}
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => cameraInputRef.current?.click()} className="gap-1.5">
                        <Camera className="w-3.5 h-3.5" />
                        Retake
                      </Button>
                    </div>
                  </div>
                )}
                {!isAnalyzing && analysis && (
                  <Button variant="outline" size="sm" onClick={clearPhoto} className="gap-1.5">
                    <Camera className="w-3.5 h-3.5" />
                    Log another meal
                  </Button>
                )}
              </div>
            )
          )}

          {/* ── TEXT MODE ──────────────────────────────────────────────────── */}
          {inputMode === "text" && (
            <div className="space-y-3">
              <Textarea
                placeholder={MEAL_PROMPTS[mealType] ?? "Describe your meal…"}
                value={description}
                onChange={e => setDescription(e.target.value)}
                className={`min-h-[100px] resize-none text-sm ${
                  description.length > 0 && !description.trim()
                    ? "border-red-500/50 focus-visible:ring-red-500/30"
                    : ""
                }`}
                disabled={isAnalyzing}
              />
              {/* Quick suggestion chips */}
              {!isAnalyzing && !analysis && (
                <div className="space-y-1.5">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Quick add</p>
                  <div className="flex flex-wrap gap-1.5">
                    {(MEAL_SUGGESTIONS[mealType] ?? []).map(suggestion => (
                      <button
                        key={suggestion}
                        onClick={() => setDescription(prev => prev ? `${prev}, ${suggestion.toLowerCase()}` : suggestion)}
                        className="px-2.5 py-1 rounded-full text-xs border border-amber-500/30 bg-amber-500/5 text-amber-300 hover:bg-amber-500/15 hover:border-amber-500/50 transition-colors"
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              <div className="flex gap-2">
                <Button
                  className="flex-1 bg-amber-500 hover:bg-amber-600 text-white"
                  onClick={handleTextSubmit}
                  disabled={isAnalyzing || !description.trim()}
                >
                  {isAnalyzing ? (
                    <><Loader2 className="w-4 h-4 animate-spin mr-2" />Analyzing…</>
                  ) : (
                    "Log meal"
                  )}
                </Button>
                {(description || analysis) && !isAnalyzing && (
                  <Button variant="outline" size="icon" onClick={clearText}>
                    <X className="w-4 h-4" />
                  </Button>
                )}
              </div>
            </div>
          )}
        </div>

      {/* ── Analysis result ───────────────────────────────────────────────────── */}
      {analysis && !isAnalyzing && (
        <div
          className="rounded-2xl p-4 space-y-4"
          style={{
            background: "hsl(152,25%,8%,0.7)",
            border: "1px solid hsl(152,30%,18%,0.6)",
          }}
        >
            {/* Header */}
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <CheckCircle2 className="w-4 h-4 text-green-400 shrink-0" />
                  <span className="text-sm font-semibold text-green-400">Logged</span>
                  <Badge variant="outline" className="text-xs">{MEAL_ICONS[mealType]} {mealType}</Badge>
                </div>
                <p className="text-sm font-medium">{analysis.summary}</p>
              </div>
              <div className="text-right shrink-0">
                <p className="text-2xl font-bold text-amber-400">{analysis.totalCalories}</p>
                <p className="text-xs text-muted-foreground">calories</p>
              </div>
            </div>

            {/* Macro + GI badges */}
            <div className="flex gap-2 flex-wrap">
              {analysis.glycemicImpact && (
                <Badge variant="outline" className={`text-xs ${GI_STYLE[analysis.glycemicImpact] ?? ""}`}>
                  GI: {analysis.glycemicImpact}
                </Badge>
              )}
              {analysis.dominantMacro && (
                <Badge variant="outline" className={`text-xs border-border ${MACRO_COLOR[analysis.dominantMacro] ?? ""}`}>
                  {analysis.dominantMacro.charAt(0).toUpperCase() + analysis.dominantMacro.slice(1)} dominant
                </Badge>
              )}
            </div>

            {/* Food items with macro breakdown */}
            {analysis.foodItems?.length > 0 && (
              <div className="space-y-2">
                {analysis.foodItems.map((item, i) => (
                  <div key={i} className="space-y-0.5">
                    <div className="flex justify-between items-baseline text-sm">
                      <span className="text-muted-foreground">
                        {item.name} <span className="text-xs opacity-70">({item.portion})</span>
                      </span>
                      <span className="text-xs font-medium shrink-0 ml-2">
                        {item.calories} kcal
                      </span>
                    </div>
                    <div className="flex gap-3 text-[10px] text-muted-foreground/70 pl-0">
                      <span><span className="text-blue-400">P</span> {item.protein_g}g</span>
                      <span><span className="text-amber-400">C</span> {item.carbs_g}g</span>
                      <span><span className="text-rose-400">F</span> {item.fat_g}g</span>
                    </div>
                  </div>
                ))}
                {/* Meal macro totals bar */}
                {(() => {
                  const totalP = analysis.foodItems.reduce((s, f) => s + (f.protein_g ?? 0), 0);
                  const totalC = analysis.foodItems.reduce((s, f) => s + (f.carbs_g ?? 0), 0);
                  const totalF = analysis.foodItems.reduce((s, f) => s + (f.fat_g ?? 0), 0);
                  const totalCal = totalP * 4 + totalC * 4 + totalF * 9 || 1;
                  const pPct = Math.round(totalP * 4 / totalCal * 100);
                  const cPct = Math.round(totalC * 4 / totalCal * 100);
                  const fPct = 100 - pPct - cPct;
                  return (
                    <div className="mt-2 space-y-1">
                      <div className="flex justify-between text-[10px] text-muted-foreground">
                        <span><span className="text-blue-400">{totalP.toFixed(1)}g protein</span></span>
                        <span><span className="text-amber-400">{totalC.toFixed(1)}g carbs</span></span>
                        <span><span className="text-rose-400">{totalF.toFixed(1)}g fat</span></span>
                      </div>
                      <div className="flex h-2 rounded-full overflow-hidden gap-px">
                        <div style={{ width: `${pPct}%` }} className="bg-blue-500/70 rounded-l-full" />
                        <div style={{ width: `${cPct}%` }} className="bg-amber-500/70" />
                        <div style={{ width: `${Math.max(fPct, 0)}%` }} className="bg-rose-500/70 rounded-r-full" />
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}

            {/* Insights */}
            <div className="border-t border-border/40 pt-3 space-y-3">
              <div className="flex items-start gap-2">
                <Zap className="w-3.5 h-3.5 text-violet-400 shrink-0 mt-0.5" />
                <div>
                  <p className="text-xs font-medium text-violet-400 mb-0.5">Mood & energy prediction</p>
                  <p className="text-xs text-muted-foreground leading-relaxed">{analysis.moodImpact}</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <Moon className="w-3.5 h-3.5 text-blue-400 shrink-0 mt-0.5" />
                <div>
                  <p className="text-xs font-medium text-blue-400 mb-0.5">Tonight's sleep & dreams</p>
                  <p className="text-xs text-muted-foreground leading-relaxed">{analysis.dreamRelevance}</p>
                </div>
              </div>
            </div>
        </div>
      )}

      {/* ── Daily nutrition totals ────────────────────────────────────────────── */}
      {history && history.length > 0 && (() => {
        const todayStr = new Date().toDateString();
        const todayLogs = history.filter(l => new Date(l.loggedAt).toDateString() === todayStr);
        if (todayLogs.length === 0) return null;
        const totalCal = todayLogs.reduce((s, l) => s + (l.totalCalories ?? 0), 0);
        // Sum macros from foodItems arrays
        let totalP = 0, totalC = 0, totalFat = 0, totalFib = 0;
        for (const l of todayLogs) {
          for (const fi of (l.foodItems ?? [])) {
            totalP += fi.protein_g ?? 0;
            totalC += fi.carbs_g ?? 0;
            totalFat += fi.fat_g ?? 0;
          }
        }
        // Daily targets (rough averages)
        const CAL_GOAL = 2000, P_GOAL = 50, C_GOAL = 275, FAT_GOAL = 78, FIB_GOAL = 28;
        const ring = (val: number, goal: number, color: string, label: string, unit: string) => {
          const pct = Math.min(val / goal * 100, 100);
          const r = 18, circ = 2 * Math.PI * r;
          const dash = circ * pct / 100;
          return (
            <div className="flex flex-col items-center gap-1">
              <div className="relative w-12 h-12">
                <svg
                  className="w-full h-full -rotate-90"
                  viewBox="0 0 44 44"
                  role="img"
                  aria-label={`${label}: ${val % 1 === 0 ? val : val.toFixed(0)}${unit} (${Math.round(pct)}% of daily goal)`}
                >
                  <circle cx="22" cy="22" r={r} fill="none" stroke="currentColor" className="text-muted/30" strokeWidth="4" />
                  <circle
                    cx="22" cy="22" r={r} fill="none"
                    stroke="currentColor" className={color} strokeWidth="4"
                    strokeDasharray={`${dash} ${circ}`}
                    strokeLinecap="round"
                  />
                </svg>
                <span className="absolute inset-0 flex items-center justify-center text-[9px] font-bold" aria-hidden="true">
                  {Math.round(pct)}%
                </span>
              </div>
              <span className="text-[9px] text-muted-foreground text-center leading-tight" aria-hidden="true">
                {label}<br />{val % 1 === 0 ? val : val.toFixed(0)}{unit}
              </span>
            </div>
          );
        };
        return (
          <Card className="border-amber-500/20 bg-amber-500/5">
            <CardContent className="pt-4 pb-3">
              <p className="text-xs font-semibold text-amber-400 mb-3">Today's nutrition</p>
              <div className="flex justify-around">
                {ring(Math.round(totalCal), CAL_GOAL, "text-amber-400", "Calories", " kcal")}
                {ring(Math.round(totalP), P_GOAL, "text-blue-400", "Protein", "g")}
                {ring(Math.round(totalC), C_GOAL, "text-green-400", "Carbs", "g")}
                {ring(Math.round(totalFat), FAT_GOAL, "text-rose-400", "Fat", "g")}
                {ring(Math.round(totalFib), FIB_GOAL, "text-violet-400", "Fiber", "g")}
              </div>
              <p className="text-[10px] text-muted-foreground mt-2 text-center">
                Based on {todayLogs.length} meal{todayLogs.length !== 1 ? "s" : ""} logged today · targets are population averages
              </p>
            </CardContent>
          </Card>
        );
      })()}

      {/* ── History ───────────────────────────────────────────────────────────── */}
      {history && history.length > 0 && (
        <div className="space-y-3">
          {/* Filter tabs */}
          <div className="flex gap-1 overflow-x-auto pb-1">
            {(["all", ...MEAL_TYPES] as FilterType[]).map(t => (
              <button
                key={t}
                onClick={() => setFilterType(t)}
                className={`shrink-0 px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
                  filterType === t
                    ? "border-amber-500/60 bg-amber-500/15 text-amber-300"
                    : "border-border/50 bg-muted/20 text-muted-foreground hover:text-foreground"
                }`}
              >
                {t === "all" ? "All" : `${MEAL_ICONS[t]} ${t.charAt(0).toUpperCase() + t.slice(1)}`}
              </button>
            ))}
          </div>

          {/* Filtered list */}
          {(() => {
            const filtered = filterType === "all"
              ? history
              : history.filter(l => l.mealType === filterType);
            if (filtered.length === 0) return (
              <p className="text-xs text-muted-foreground text-center py-4">
                No {filterType} entries logged yet.
              </p>
            );
            return (
              <div className="space-y-2">
                {filtered.map(log => (
              <div
                key={log.id}
                className="flex items-center gap-3 rounded-lg border border-border/50 bg-muted/20 p-3 text-sm"
              >
                <div className="w-8 h-8 rounded-lg bg-amber-500/15 flex items-center justify-center text-base shrink-0">
                  {MEAL_ICONS[log.mealType ?? "snack"] ?? "🍽️"}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{log.summary ?? "Meal"}</p>
                  <p className="text-xs text-muted-foreground">
                    {formatDate(log.loggedAt)} · {formatTime(log.loggedAt)}
                    {log.totalCalories ? ` · ${log.totalCalories} cal` : ""}
                    {log.dominantMacro && (
                      <span className={` · ${MACRO_COLOR[log.dominantMacro] ?? ""}`}>
                        {log.dominantMacro}
                      </span>
                    )}
                  </p>
                </div>
                {log.glycemicImpact && (
                  <Badge variant="outline" className={`text-[10px] shrink-0 ${GI_STYLE[log.glycemicImpact] ?? ""}`}>
                    GI {log.glycemicImpact}
                  </Badge>
                )}
              </div>
                ))}
              </div>
            );
          })()}
        </div>
      )}

      {/* Empty state */}
      {(!history || history.length === 0) && !analysis && !isAnalyzing && (
        <div className="text-center py-8 space-y-2">
          <TrendingUp className="w-8 h-8 text-muted-foreground/30 mx-auto" />
          <p className="text-sm text-muted-foreground">
            Log your first meal to start tracking how food affects your mood and dreams.
          </p>
        </div>
      )}

      {/* ── Meal history section (#378) ──────────────────────────────────────── */}
      {mealHistoryData && mealHistoryData.length > 0 && (
        <div className="space-y-2">
          <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Meal history
          </h2>
          <MealHistory
            meals={mealHistoryData}
            onToggleFavorite={handleToggleFavorite}
            onRelog={handleRelog}
          />
        </div>
      )}
    </div>
  );
}
