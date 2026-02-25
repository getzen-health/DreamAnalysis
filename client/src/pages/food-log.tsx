import { getParticipantId } from "@/lib/participant";
import { useState, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import {
  Camera,
  CheckCircle2,
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

const USER_ID = getParticipantId();

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
  const { toast } = useToast();
  const qc = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [inputMode, setInputMode] = useState<InputMode>("photo");
  const [mealType, setMealType] = useState(autoMealType());
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<FoodAnalysis | null>(null);

  // Photo mode state
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // Text mode state
  const [description, setDescription] = useState("");

  // History
  const { data: history } = useQuery<FoodLog[]>({
    queryKey: ["/api/food/logs", USER_ID],
    queryFn: async () => {
      const res = await fetch(`/api/food/logs/${USER_ID}`, { credentials: "include" });
      if (!res.ok) return [];
      return res.json();
    },
  });

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
    setMealType(autoMealType());
    const b64 = await compressToBase64(file);
    await submitAnalysis({ imageBase64: b64 });
  };

  const clearPhoto = () => {
    setPreviewUrl(null);
    setAnalysis(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
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
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (mode === "text") setDescription("");
  };

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="max-w-lg mx-auto py-6 px-4 space-y-5">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Utensils className="w-6 h-6 text-amber-400" />
        <div>
          <h1 className="text-xl font-bold">Meal Log</h1>
          <p className="text-xs text-muted-foreground">Track what you eat · see how food shapes mood & dreams</p>
        </div>
      </div>

      {/* ── Input mode toggle ─────────────────────────────────────────────────── */}
      <div className="flex rounded-lg border border-border/60 p-1 gap-1 bg-muted/20">
        <button
          onClick={() => switchMode("photo")}
          className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-md text-sm font-medium transition-colors ${
            inputMode === "photo"
              ? "bg-background shadow-sm text-foreground"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <Camera className="w-4 h-4" />
          Photo
        </button>
        <button
          onClick={() => switchMode("text")}
          className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-md text-sm font-medium transition-colors ${
            inputMode === "text"
              ? "bg-background shadow-sm text-foreground"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <PenLine className="w-4 h-4" />
          Describe
        </button>
      </div>

      {/* ── Meal type chips ───────────────────────────────────────────────────── */}
      <div className="flex gap-2">
        {MEAL_TYPES.map(t => (
          <button
            key={t}
            onClick={() => setMealType(t)}
            className={`flex-1 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              mealType === t
                ? "border-amber-500/60 bg-amber-500/15 text-amber-300"
                : "border-border/50 bg-muted/20 text-muted-foreground hover:text-foreground"
            }`}
          >
            {MEAL_ICONS[t]} {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {/* ── Capture card ─────────────────────────────────────────────────────── */}
      <Card>
        <CardContent className="pt-5 space-y-4">

          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            className="hidden"
            onChange={handleFileChange}
          />

          {/* ── PHOTO MODE ─────────────────────────────────────────────────── */}
          {inputMode === "photo" && (
            !previewUrl ? (
              <button
                onClick={() => fileInputRef.current?.click()}
                className="w-full h-40 rounded-xl border-2 border-dashed border-amber-500/30 hover:border-amber-500/60 bg-amber-500/5 hover:bg-amber-500/10 transition-all flex flex-col items-center justify-center gap-3 group"
              >
                <div className="w-12 h-12 rounded-full bg-amber-500/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <Camera className="w-6 h-6 text-amber-400" />
                </div>
                <div className="text-center">
                  <p className="text-sm font-medium">Photograph your meal</p>
                  <p className="text-xs text-muted-foreground">Tap to use camera or pick a photo</p>
                </div>
              </button>
            ) : (
              <div className="space-y-3">
                <div className="relative rounded-xl overflow-hidden">
                  <img src={previewUrl} alt="Meal preview" className="w-full h-48 object-cover" />
                  <button
                    onClick={clearPhoto}
                    className="absolute top-2 right-2 w-7 h-7 rounded-full bg-black/60 flex items-center justify-center hover:bg-black/80 transition-colors"
                  >
                    <X className="w-3.5 h-3.5 text-white" />
                  </button>
                  {isAnalyzing && (
                    <div className="absolute inset-0 bg-black/50 flex items-center justify-center gap-2">
                      <Loader2 className="w-5 h-5 animate-spin text-white" />
                      <span className="text-white text-sm font-medium">Analyzing…</span>
                    </div>
                  )}
                </div>
                {!isAnalyzing && (
                  <Button variant="outline" size="sm" onClick={() => fileInputRef.current?.click()} className="gap-1.5">
                    <Camera className="w-3.5 h-3.5" />
                    Retake photo
                  </Button>
                )}
              </div>
            )
          )}

          {/* ── TEXT MODE ──────────────────────────────────────────────────── */}
          {inputMode === "text" && (
            <div className="space-y-3">
              <Textarea
                placeholder={"Describe your meal…\ne.g. 2 scrambled eggs, one slice whole-wheat toast with butter, a glass of orange juice"}
                value={description}
                onChange={e => setDescription(e.target.value)}
                className="min-h-[100px] resize-none text-sm"
                disabled={isAnalyzing}
              />
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
        </CardContent>
      </Card>

      {/* ── Analysis result ───────────────────────────────────────────────────── */}
      {analysis && !isAnalyzing && (
        <Card className="border-green-500/30 bg-green-500/5">
          <CardContent className="pt-5 space-y-4">
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

            {/* Food items */}
            {analysis.foodItems?.length > 0 && (
              <div className="space-y-1">
                {analysis.foodItems.map((item, i) => (
                  <div key={i} className="flex justify-between items-baseline text-sm">
                    <span className="text-muted-foreground">
                      {item.name} <span className="text-xs">({item.portion})</span>
                    </span>
                    <span className="text-xs text-muted-foreground/70 shrink-0 ml-2">
                      {item.calories} kcal
                    </span>
                  </div>
                ))}
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
          </CardContent>
        </Card>
      )}

      {/* ── History ───────────────────────────────────────────────────────────── */}
      {history && history.length > 0 && (
        <div className="space-y-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Recent meals</p>
          <div className="space-y-2">
            {history.map(log => (
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
    </div>
  );
}
