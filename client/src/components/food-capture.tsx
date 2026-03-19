/**
 * FoodCapture — mobile-first food image capture component.
 *
 * Supports:
 *   - Camera / file-picker capture (up to 4 images per meal — #378)
 *   - Barcode scanning via manual entry with OpenFoodFacts lookup (#367)
 *   - Vision-AI fallback when barcode is not found
 *
 * On native Capacitor (iOS/Android): uses @capacitor/camera plugin.
 * On web: falls back to <input type="file" accept="image/*" capture="environment">.
 *
 * Flow (camera):
 *   1. User taps "Capture Meal" → camera/file picker opens
 *   2. Up to 4 images shown as thumbnails; "+" to add more
 *   3. On "Analyze": calls analyzeFoodImage for each image, aggregates results
 *   4. Combined nutrition totals shown
 *
 * Flow (barcode):
 *   1. User taps "Scan Barcode"
 *   2. Text input for manual barcode entry (web fallback — native camera barcode requires plugin)
 *   3. lookupBarcode() fetches OpenFoodFacts; portion size input shown
 *   4. If not found, offers fallback to camera/AI analysis
 */

import { useState, useRef, useCallback } from "react";
import {
  Camera,
  Flame,
  Beef,
  Wheat,
  Droplet,
  Leaf,
  AlertCircle,
  Barcode,
  Plus,
  X,
  ChevronRight,
  Loader2,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { analyzeFoodImage, type FoodImageAnalysisResult } from "@/lib/ml-api";
import { lookupBarcode, type BarcodeProduct } from "@/lib/barcode-api";
import { hapticSuccess } from "@/lib/haptics";

// ── Helpers ───────────────────────────────────────────────────────────────────

const MAX_IMAGES = 4;

function isCapacitorNative(): boolean {
  try {
    const cap = (window as unknown as { Capacitor?: { isNativePlatform?: () => boolean } }).Capacitor;
    return !!cap?.isNativePlatform?.();
  } catch {
    return false;
  }
}

function glycemicColor(impact: string): string {
  if (impact === "low")  return "bg-cyan-500/15 text-cyan-400 border-cyan-500/30";
  if (impact === "high") return "bg-rose-500/15 text-rose-400 border-rose-500/30";
  return "bg-amber-500/15 text-amber-400 border-amber-500/30";
}

function macroColor(macro: string): string {
  if (macro === "protein") return "bg-indigo-500/15 text-indigo-400 border-indigo-500/30";
  if (macro === "fat")     return "bg-yellow-500/15 text-yellow-400 border-yellow-500/30";
  if (macro === "carbs")   return "bg-orange-500/15 text-orange-400 border-orange-500/30";
  return "bg-muted/50 text-muted-foreground border-border/40";
}

/** Aggregate multiple FoodImageAnalysisResult objects into one combined result. */
function aggregateResults(results: FoodImageAnalysisResult[]): FoodImageAnalysisResult {
  if (results.length === 1) return results[0];
  const all = results.reduce(
    (acc, r) => ({
      food_items:      [...acc.food_items, ...r.food_items],
      total_calories:  acc.total_calories  + r.total_calories,
      total_protein_g: acc.total_protein_g + r.total_protein_g,
      total_carbs_g:   acc.total_carbs_g   + r.total_carbs_g,
      total_fat_g:     acc.total_fat_g     + r.total_fat_g,
      total_fiber_g:   acc.total_fiber_g   + r.total_fiber_g,
      // pick the "worst" (highest) glycemic and lowest confidence
      glycemic_impact:  r.total_calories > acc.total_calories ? r.glycemic_impact : acc.glycemic_impact,
      dominant_macro:   r.total_calories > acc.total_calories ? r.dominant_macro  : acc.dominant_macro,
      confidence:      Math.min(acc.confidence, r.confidence),
      analysis_method: "combined",
      summary:         `Combined meal (${results.length} images)`,
    }),
    {
      food_items:      [] as FoodImageAnalysisResult["food_items"],
      total_calories:  0,
      total_protein_g: 0,
      total_carbs_g:   0,
      total_fat_g:     0,
      total_fiber_g:   0,
      glycemic_impact: results[0].glycemic_impact,
      dominant_macro:  results[0].dominant_macro,
      confidence:      results[0].confidence,
      analysis_method: results[0].analysis_method,
      summary:         "",
    }
  );
  return all;
}

/** Convert a BarcodeProduct + portion multiplier to FoodImageAnalysisResult. */
function barcodeToAnalysisResult(
  product: BarcodeProduct,
  servings: number
): FoodImageAnalysisResult {
  const s = servings || 1;
  const cal  = Math.round(product.calories  * s);
  const prot = Math.round(product.protein_g * s * 10) / 10;
  const carb = Math.round(product.carbs_g   * s * 10) / 10;
  const fat  = Math.round(product.fat_g     * s * 10) / 10;
  const fib  = Math.round(product.fiber_g   * s * 10) / 10;

  // Determine dominant macro by calories contribution
  const protCal = prot * 4;
  const carbCal = carb * 4;
  const fatCal  = fat  * 9;
  const maxMacro = Math.max(protCal, carbCal, fatCal);
  const dominant_macro =
    maxMacro === protCal ? "protein" :
    maxMacro === carbCal ? "carbs"   : "fat";

  // Rough glycemic estimate from carb fraction
  const carbFrac = cal > 0 ? carbCal / cal : 0;
  const glycemic_impact = carbFrac > 0.6 ? "high" : carbFrac > 0.35 ? "medium" : "low";

  return {
    food_items: [{
      name:      `${product.name}${product.brand ? ` (${product.brand})` : ""}`,
      portion:   product.servingSize
        ? `${s === 1 ? "" : `${s}× `}${product.servingSize}`
        : `${s} serving${s !== 1 ? "s" : ""}`,
      calories:  cal,
      protein_g: prot,
      carbs_g:   carb,
      fat_g:     fat,
      fiber_g:   fib,
    }],
    total_calories:  cal,
    total_protein_g: prot,
    total_carbs_g:   carb,
    total_fat_g:     fat,
    total_fiber_g:   fib,
    glycemic_impact,
    dominant_macro,
    confidence:      1.0,
    analysis_method: "barcode",
    summary: `${product.name}${product.servingSize ? ` · ${product.servingSize}` : ""}`,
  };
}

// ── Props ─────────────────────────────────────────────────────────────────────

export interface FoodCaptureProps {
  /** Called with the full analysis result after a successful scan */
  onAnalyzed?: (result: FoodImageAnalysisResult) => void;
  className?: string;
}

// ── State types ───────────────────────────────────────────────────────────────

type CaptureState = "idle" | "preview" | "analyzing" | "done" | "error" | "barcode-scan";

interface CapturedImage {
  previewUrl:  string;
  base64:      string;
}

// ── Component ─────────────────────────────────────────────────────────────────

export function FoodCapture({ onAnalyzed, className }: FoodCaptureProps) {
  const [captureState, setCaptureState]   = useState<CaptureState>("idle");
  const [images, setImages]               = useState<CapturedImage[]>([]);
  const [result, setResult]               = useState<FoodImageAnalysisResult | null>(null);
  const [errorMsg, setErrorMsg]           = useState<string | null>(null);

  // Barcode state
  const [barcodeInput, setBarcodeInput]   = useState("");
  const [barcodeProduct, setBarcodeProduct] = useState<BarcodeProduct | null>(null);
  const [barcodeServings, setBarcodeServings] = useState("1");
  const [barcodeLoading, setBarcodeLoading]   = useState(false);
  const [barcodeNotFound, setBarcodeNotFound] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // ── Reset ──────────────────────────────────────────────────────────────────

  const reset = useCallback(() => {
    setImages([]);
    setResult(null);
    setErrorMsg(null);
    setBarcodeInput("");
    setBarcodeProduct(null);
    setBarcodeServings("1");
    setBarcodeLoading(false);
    setBarcodeNotFound(false);
    setCaptureState("idle");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  // ── Image capture ──────────────────────────────────────────────────────────

  const addBase64Image = useCallback((base64: string, mime: string) => {
    const previewUrl = `data:${mime};base64,${base64}`;
    const entry: CapturedImage = { previewUrl, base64: previewUrl };
    setImages(prev => {
      const next = [...prev, entry].slice(0, MAX_IMAGES);
      return next;
    });
    setCaptureState("preview");
    hapticSuccess();
  }, []);

  const captureWithCapacitor = useCallback(async () => {
    try {
      const { Camera: CapCamera, CameraResultType, CameraSource } = await import(
        "@capacitor/camera"
      );
      const photo = await CapCamera.getPhoto({
        resultType: CameraResultType.Base64,
        source:     CameraSource.Camera,
        quality:    80,
      });
      if (!photo.base64String) return;
      const mime = photo.format === "png" ? "image/png" : "image/jpeg";
      addBase64Image(photo.base64String, mime);
    } catch (err) {
      if ((err as Error)?.message?.toLowerCase().includes("cancel")) return;
      setErrorMsg((err as Error)?.message ?? "Camera unavailable");
      setCaptureState("error");
    }
  }, [addBase64Image]);

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
        const entry: CapturedImage = { previewUrl: dataUrl, base64: dataUrl };
        setImages(prev => [...prev, entry].slice(0, MAX_IMAGES));
        setCaptureState("preview");
        hapticSuccess();
      };
      reader.readAsDataURL(file);
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

  const removeImage = useCallback((index: number) => {
    setImages(prev => {
      const next = prev.filter((_, i) => i !== index);
      if (next.length === 0) setCaptureState("idle");
      return next;
    });
  }, []);

  // ── Analyze (multi-image) ──────────────────────────────────────────────────

  const ANALYSIS_TIMEOUT_MS = 30_000;

  const handleAnalyze = useCallback(async () => {
    if (images.length === 0) return;
    setCaptureState("analyzing");
    setErrorMsg(null);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const promises = images.map(img => {
        const timeout = new Promise<never>((_, reject) => {
          const id = setTimeout(() => {
            reject(new DOMException("Analysis timed out", "TimeoutError"));
            controller.abort();
          }, ANALYSIS_TIMEOUT_MS);
          // Clear timeout if the controller is aborted externally (cancel button)
          controller.signal.addEventListener("abort", () => clearTimeout(id), { once: true });
        });
        return Promise.race([analyzeFoodImage(img.base64), timeout]);
      });

      const results  = await Promise.all(promises);
      const combined = aggregateResults(results);
      setResult(combined);
      setCaptureState("done");
      onAnalyzed?.(combined);
    } catch (err) {
      // Don't show error if user explicitly cancelled
      if (controller.signal.aborted && !(err instanceof DOMException && err.name === "TimeoutError")) {
        setCaptureState("preview");
        return;
      }

      if (err instanceof DOMException && err.name === "TimeoutError") {
        setErrorMsg("Analysis timed out — ML backend may be offline. Try again later.");
      } else if (err instanceof TypeError || (err instanceof Error && /fetch|network|ECONNREFUSED/i.test(err.message))) {
        setErrorMsg("Could not reach the analysis server. Check your connection.");
      } else {
        setErrorMsg(err instanceof Error ? err.message : "Analysis failed");
      }
      setCaptureState("error");
    } finally {
      abortRef.current = null;
    }
  }, [images, onAnalyzed]);

  const handleCancelAnalysis = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  // ── Barcode lookup ─────────────────────────────────────────────────────────

  const handleBarcodeSubmit = useCallback(async () => {
    if (!barcodeInput.trim()) return;
    setBarcodeLoading(true);
    setBarcodeNotFound(false);
    setBarcodeProduct(null);
    try {
      const product = await lookupBarcode(barcodeInput.trim());
      if (!product) {
        setBarcodeNotFound(true);
      } else {
        setBarcodeProduct(product);
        setBarcodeServings("1");
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Barcode lookup failed");
      setCaptureState("error");
    } finally {
      setBarcodeLoading(false);
    }
  }, [barcodeInput]);

  const handleBarcodeLog = useCallback(() => {
    if (!barcodeProduct) return;
    const servings = parseFloat(barcodeServings) || 1;
    const analysisResult = barcodeToAnalysisResult(barcodeProduct, servings);
    setResult(analysisResult);
    setCaptureState("done");
    onAnalyzed?.(analysisResult);
  }, [barcodeProduct, barcodeServings, onAnalyzed]);

  const handleBarcodeFallback = useCallback(() => {
    // User wants to use camera/AI instead after barcode not found
    setBarcodeNotFound(false);
    setBarcodeInput("");
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

      {/* ── Idle — capture + barcode buttons ── */}
      {captureState === "idle" && (
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleCapture}
            className="flex-1"
          >
            <Camera className="h-4 w-4 mr-2" />
            Capture Meal
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setCaptureState("barcode-scan")}
            className="flex-1"
          >
            <Barcode className="h-4 w-4 mr-2" />
            Scan Barcode
          </Button>
        </div>
      )}

      {/* ── Barcode scan — manual entry (web fallback) ── */}
      {captureState === "barcode-scan" && (
        <Card>
          <CardHeader className="pb-2 pt-3 px-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Barcode className="h-4 w-4 text-primary" />
              Barcode Lookup
            </CardTitle>
          </CardHeader>
          <CardContent className="px-3 pb-3 space-y-3">
            <p className="text-xs text-muted-foreground">
              Enter the barcode number from the package (UPC/EAN).
            </p>
            <div className="space-y-1">
              <Label htmlFor="barcode-input" className="text-xs">Barcode number</Label>
              <div className="flex gap-2">
                <Input
                  id="barcode-input"
                  value={barcodeInput}
                  onChange={e => {
                    setBarcodeInput(e.target.value);
                    setBarcodeNotFound(false);
                    setBarcodeProduct(null);
                  }}
                  placeholder="e.g. 012345678901"
                  className="text-sm"
                  onKeyDown={e => { if (e.key === "Enter") handleBarcodeSubmit(); }}
                  disabled={barcodeLoading}
                />
                <Button
                  size="sm"
                  onClick={handleBarcodeSubmit}
                  disabled={barcodeLoading || !barcodeInput.trim()}
                >
                  {barcodeLoading ? (
                    <span className="h-3 w-3 rounded-full border-2 border-current border-t-transparent animate-spin inline-block" />
                  ) : (
                    <ChevronRight className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>

            {/* Product found — show info + servings */}
            {barcodeProduct && (
              <div className="space-y-3 rounded-md bg-muted/20 p-3 border border-border/40">
                <div className="flex items-start gap-3">
                  {barcodeProduct.imageUrl && (
                    <img
                      src={barcodeProduct.imageUrl}
                      alt={barcodeProduct.name}
                      className="w-14 h-14 rounded object-cover shrink-0 border border-border/30"
                    />
                  )}
                  <div className="min-w-0">
                    <p className="text-sm font-semibold leading-snug truncate">{barcodeProduct.name}</p>
                    {barcodeProduct.brand && (
                      <p className="text-xs text-muted-foreground">{barcodeProduct.brand}</p>
                    )}
                    {barcodeProduct.servingSize && (
                      <p className="text-xs text-muted-foreground">Per serving: {barcodeProduct.servingSize}</p>
                    )}
                  </div>
                </div>

                {/* Nutrition per serving */}
                <div className="grid grid-cols-4 gap-1 text-center">
                  <div className="rounded bg-muted/30 p-1.5">
                    <p className="text-xs font-semibold tabular-nums">{barcodeProduct.calories}</p>
                    <p className="text-[10px] text-muted-foreground">kcal</p>
                  </div>
                  <div className="rounded bg-muted/30 p-1.5">
                    <p className="text-xs font-semibold tabular-nums">{barcodeProduct.protein_g}g</p>
                    <p className="text-[10px] text-muted-foreground">protein</p>
                  </div>
                  <div className="rounded bg-muted/30 p-1.5">
                    <p className="text-xs font-semibold tabular-nums">{barcodeProduct.carbs_g}g</p>
                    <p className="text-[10px] text-muted-foreground">carbs</p>
                  </div>
                  <div className="rounded bg-muted/30 p-1.5">
                    <p className="text-xs font-semibold tabular-nums">{barcodeProduct.fat_g}g</p>
                    <p className="text-[10px] text-muted-foreground">fat</p>
                  </div>
                </div>

                {barcodeProduct.allergens && (
                  <p className="text-[10px] text-amber-400/80">
                    Contains: {barcodeProduct.allergens}
                  </p>
                )}

                {/* Servings input */}
                <div className="flex items-center gap-3">
                  <Label htmlFor="servings-input" className="text-xs shrink-0 text-muted-foreground">
                    How many servings?
                  </Label>
                  <Input
                    id="servings-input"
                    type="number"
                    min="0.25"
                    step="0.25"
                    value={barcodeServings}
                    onChange={e => setBarcodeServings(e.target.value)}
                    className="w-20 h-7 text-sm text-center"
                  />
                </div>

                <Button size="sm" onClick={handleBarcodeLog} className="w-full">
                  Log this meal
                </Button>
              </div>
            )}

            {/* Not found — offer fallback */}
            {barcodeNotFound && (
              <div className="rounded-md bg-amber-500/10 border border-amber-500/30 p-3 space-y-2">
                <p className="text-xs text-amber-400">
                  Barcode not found in OpenFoodFacts database.
                </p>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleBarcodeFallback}
                  className="w-full"
                >
                  <Camera className="h-3.5 w-3.5 mr-2" />
                  Use camera + AI instead
                </Button>
              </div>
            )}

            {/* Cancel */}
            <Button size="sm" variant="ghost" onClick={reset} className="w-full text-muted-foreground">
              Cancel
            </Button>
          </CardContent>
        </Card>
      )}

      {/* ── Preview — thumbnails row + analyze/add buttons ── */}
      {(captureState === "preview" || captureState === "analyzing") && images.length > 0 && (
        <Card>
          <CardContent className="p-3 space-y-3">
            {/* Thumbnails row */}
            <div className="flex gap-2 flex-wrap">
              {images.map((img, i) => (
                <div key={i} className="relative w-20 h-20 shrink-0">
                  <img
                    src={img.previewUrl}
                    alt={`Meal photo ${i + 1}`}
                    className="w-full h-full rounded-md object-cover border border-border/30"
                  />
                  {captureState !== "analyzing" && (
                    <button
                      onClick={() => removeImage(i)}
                      className="absolute -top-2 -right-2 h-6 w-6 min-w-[44px] min-h-[44px] rounded-full bg-background border border-border/60 flex items-center justify-center"
                      aria-label={`Remove photo ${i + 1}`}
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              ))}

              {/* "+" button to add another image */}
              {images.length < MAX_IMAGES && captureState !== "analyzing" && (
                <button
                  onClick={handleCapture}
                  className="w-20 h-20 rounded-md border-2 border-dashed border-border/50 flex flex-col items-center justify-center gap-1 text-muted-foreground hover:border-border hover:text-foreground transition-colors"
                  aria-label="Add another photo"
                >
                  <Plus className="h-5 w-5" />
                  <span className="text-[10px]">Add</span>
                </button>
              )}
            </div>

            {images.length > 1 && (
              <p className="text-xs text-muted-foreground">
                {images.length} photos — nutrition will be combined
              </p>
            )}

            {captureState === "analyzing" ? (
              <div className="space-y-2">
                <div className="flex items-center justify-center gap-2 py-2">
                  <Loader2 className="h-4 w-4 animate-spin text-primary" />
                  <span className="text-sm font-medium animate-pulse">Analyzing your meal...</span>
                </div>
                <div className="w-full h-1.5 rounded-full bg-muted/30 overflow-hidden">
                  <div className="h-full rounded-full bg-primary/60 animate-pulse" style={{ width: "100%" }} />
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleCancelAnalysis}
                  className="w-full"
                >
                  <X className="h-3.5 w-3.5 mr-2" />
                  Cancel
                </Button>
              </div>
            ) : (
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={reset}
                  className="flex-1"
                >
                  Retake
                </Button>
                <Button
                  size="sm"
                  onClick={handleAnalyze}
                  className="flex-1"
                >
                  Analyze
                </Button>
              </div>
            )}
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
                  <Beef className="h-3 w-3 text-indigo-400" />
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
                <Leaf className="h-3 w-3 text-cyan-400" />
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

            {/* Capture another */}
            <Button
              size="sm"
              variant="outline"
              onClick={reset}
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
          <Button size="sm" variant="outline" onClick={reset} className="w-full">
            Try Again
          </Button>
        </div>
      )}
    </div>
  );
}
