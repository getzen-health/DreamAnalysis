/**
 * FoodCapture — mobile-first food image capture component.
 *
 * On native Capacitor (iOS/Android): uses @capacitor/camera plugin.
 * On web: falls back to <input type="file" accept="image/*" capture="environment">.
 *
 * Flow:
 *   1. User taps "Capture Meal" → camera/file picker opens
 *   2. Image preview shown with "Analyze" / "Retake" buttons
 *   3. On "Analyze": calls /food/analyze-image via ml-api
 *   4. Results displayed: food items, calories, macros, glycemic impact
 */

import { useState, useRef, useCallback } from "react";
import { Camera, Flame, Beef, Wheat, Droplet, Leaf, AlertCircle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { analyzeFoodImage, type FoodImageAnalysisResult } from "@/lib/ml-api";
import { hapticSuccess } from "@/lib/haptics";

// ── Helpers ───────────────────────────────────────────────────────────────────

function isCapacitorNative(): boolean {
  try {
    const cap = (window as unknown as { Capacitor?: { isNativePlatform?: () => boolean } }).Capacitor;
    return !!cap?.isNativePlatform?.();
  } catch {
    return false;
  }
}

function glycemicColor(impact: string): string {
  if (impact === "low") return "bg-emerald-500/15 text-emerald-400 border-emerald-500/30";
  if (impact === "high") return "bg-rose-500/15 text-rose-400 border-rose-500/30";
  return "bg-amber-500/15 text-amber-400 border-amber-500/30";
}

function macroColor(macro: string): string {
  if (macro === "protein") return "bg-blue-500/15 text-blue-400 border-blue-500/30";
  if (macro === "fat") return "bg-yellow-500/15 text-yellow-400 border-yellow-500/30";
  if (macro === "carbs") return "bg-orange-500/15 text-orange-400 border-orange-500/30";
  return "bg-muted/50 text-muted-foreground border-border/40";
}

// ── Props ─────────────────────────────────────────────────────────────────────

export interface FoodCaptureProps {
  /** Called with the full analysis result after a successful scan */
  onAnalyzed?: (result: FoodImageAnalysisResult) => void;
  className?: string;
}

// ── Component ─────────────────────────────────────────────────────────────────

type CaptureState = "idle" | "preview" | "analyzing" | "done" | "error";

export function FoodCapture({ onAnalyzed, className }: FoodCaptureProps) {
  const [captureState, setCaptureState] = useState<CaptureState>("idle");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [result, setResult] = useState<FoodImageAnalysisResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Image capture ──────────────────────────────────────────────────────────

  const captureWithCapacitor = useCallback(async () => {
    try {
      // Dynamic import so the module isn't bundled on web-only builds
      const { Camera: CapCamera, CameraResultType, CameraSource } = await import(
        "@capacitor/camera"
      );
      const photo = await CapCamera.getPhoto({
        resultType: CameraResultType.Base64,
        source: CameraSource.Camera,
        quality: 80,
      });

      if (!photo.base64String) return;
      const b64 = photo.base64String;
      const mime = photo.format === "png" ? "image/png" : "image/jpeg";
      setPreviewUrl(`data:${mime};base64,${b64}`);
      setImageBase64(`data:${mime};base64,${b64}`);
      setCaptureState("preview");
      hapticSuccess();
    } catch (err) {
      // User cancelled — no error needed
      if ((err as Error)?.message?.toLowerCase().includes("cancel")) return;
      setErrorMsg((err as Error)?.message ?? "Camera unavailable");
      setCaptureState("error");
    }
  }, []);

  const triggerFileInput = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = () => {
        const dataUrl = reader.result as string;
        setPreviewUrl(dataUrl);
        setImageBase64(dataUrl);
        setCaptureState("preview");
        hapticSuccess();
      };
      reader.readAsDataURL(file);
      // Reset so selecting the same file again triggers onChange
      e.target.value = "";
    },
    []
  );

  const handleCapture = useCallback(() => {
    setErrorMsg(null);
    if (isCapacitorNative()) {
      captureWithCapacitor();
    } else {
      triggerFileInput();
    }
  }, [captureWithCapacitor, triggerFileInput]);

  // ── Analyze ────────────────────────────────────────────────────────────────

  const handleAnalyze = useCallback(async () => {
    if (!imageBase64) return;
    setCaptureState("analyzing");
    try {
      const analysis = await analyzeFoodImage(imageBase64);
      setResult(analysis);
      setCaptureState("done");
      onAnalyzed?.(analysis);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Analysis failed");
      setCaptureState("error");
    }
  }, [imageBase64, onAnalyzed]);

  const handleRetake = useCallback(() => {
    setPreviewUrl(null);
    setImageBase64(null);
    setResult(null);
    setErrorMsg(null);
    setCaptureState("idle");
  }, []);

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className={className}>
      {/* Hidden file input for web fallback */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        onChange={handleFileChange}
        aria-label="Select food photo"
      />

      {/* ── Idle — camera button ── */}
      {captureState === "idle" && (
        <Button
          variant="outline"
          size="sm"
          onClick={handleCapture}
          className="w-full"
        >
          <Camera className="h-4 w-4 mr-2" />
          Capture Meal
        </Button>
      )}

      {/* ── Preview — image + action buttons ── */}
      {(captureState === "preview" || captureState === "analyzing") && previewUrl && (
        <Card>
          <CardContent className="p-3 space-y-3">
            <img
              src={previewUrl}
              alt="Meal preview"
              className="w-full rounded-md object-cover max-h-56"
            />
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={handleRetake}
                disabled={captureState === "analyzing"}
                className="flex-1"
              >
                Retake
              </Button>
              <Button
                size="sm"
                onClick={handleAnalyze}
                disabled={captureState === "analyzing"}
                className="flex-1"
              >
                {captureState === "analyzing" ? (
                  <>
                    <span className="h-3 w-3 mr-2 rounded-full border-2 border-current border-t-transparent animate-spin inline-block" />
                    Analyzing…
                  </>
                ) : (
                  "Analyze"
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Done — nutritional results ── */}
      {captureState === "done" && result && (
        <Card>
          <CardHeader className="pb-2 pt-3 px-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Flame className="h-4 w-4 text-orange-400" />
              Nutritional Breakdown
            </CardTitle>
          </CardHeader>
          <CardContent className="px-3 pb-3 space-y-3">
            {/* Summary */}
            <p className="text-xs text-muted-foreground leading-relaxed">{result.summary}</p>

            {/* Badges */}
            <div className="flex flex-wrap gap-1.5">
              <Badge variant="outline" className={glycemicColor(result.glycemic_impact)}>
                GI: {result.glycemic_impact}
              </Badge>
              <Badge variant="outline" className={macroColor(result.dominant_macro)}>
                {result.dominant_macro}-dominant
              </Badge>
              <Badge variant="outline" className="bg-muted/30 text-muted-foreground border-border/40 text-xs">
                {Math.round(result.confidence * 100)}% confidence
              </Badge>
            </div>

            {/* Macro summary row */}
            <div className="grid grid-cols-4 gap-1 text-center">
              <div className="rounded-md bg-muted/30 p-2">
                <div className="flex justify-center mb-0.5">
                  <Flame className="h-3 w-3 text-orange-400" />
                </div>
                <p className="text-xs font-semibold tabular-nums">{Math.round(result.total_calories)}</p>
                <p className="text-[10px] text-muted-foreground">kcal</p>
              </div>
              <div className="rounded-md bg-muted/30 p-2">
                <div className="flex justify-center mb-0.5">
                  <Beef className="h-3 w-3 text-blue-400" />
                </div>
                <p className="text-xs font-semibold tabular-nums">{result.total_protein_g}g</p>
                <p className="text-[10px] text-muted-foreground">protein</p>
              </div>
              <div className="rounded-md bg-muted/30 p-2">
                <div className="flex justify-center mb-0.5">
                  <Wheat className="h-3 w-3 text-orange-300" />
                </div>
                <p className="text-xs font-semibold tabular-nums">{result.total_carbs_g}g</p>
                <p className="text-[10px] text-muted-foreground">carbs</p>
              </div>
              <div className="rounded-md bg-muted/30 p-2">
                <div className="flex justify-center mb-0.5">
                  <Droplet className="h-3 w-3 text-yellow-400" />
                </div>
                <p className="text-xs font-semibold tabular-nums">{result.total_fat_g}g</p>
                <p className="text-[10px] text-muted-foreground">fat</p>
              </div>
            </div>

            {/* Fiber row */}
            {result.total_fiber_g > 0 && (
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Leaf className="h-3 w-3 text-emerald-400" />
                <span>{result.total_fiber_g}g fiber</span>
              </div>
            )}

            {/* Food items list */}
            {result.food_items.length > 0 && (
              <div className="space-y-1">
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Items detected
                </p>
                <ul className="space-y-1">
                  {result.food_items.map((item, i) => (
                    <li
                      key={i}
                      className="flex justify-between items-baseline text-xs text-muted-foreground"
                    >
                      <span className="font-medium text-foreground/80 capitalize">{item.name}</span>
                      <span className="font-mono ml-2 shrink-0">
                        {Math.round(item.calories)} kcal · {item.portion}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Retake button */}
            <Button
              size="sm"
              variant="outline"
              onClick={handleRetake}
              className="w-full"
            >
              <Camera className="h-3.5 w-3.5 mr-2" />
              Capture Another
            </Button>
          </CardContent>
        </Card>
      )}

      {/* ── Error state ── */}
      {captureState === "error" && (
        <div className="rounded-md border border-destructive/30 bg-destructive/10 p-3 space-y-2">
          <div className="flex items-center gap-2 text-destructive text-sm">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>{errorMsg ?? "Something went wrong"}</span>
          </div>
          <Button size="sm" variant="outline" onClick={handleRetake} className="w-full">
            Try Again
          </Button>
        </div>
      )}
    </div>
  );
}
