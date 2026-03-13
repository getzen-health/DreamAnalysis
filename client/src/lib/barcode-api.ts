/**
 * barcode-api.ts — OpenFoodFacts barcode lookup.
 *
 * Uses the public OpenFoodFacts v2 API (no key required).
 * Returns null for products not found; never throws on 404 or missing fields.
 */

export interface BarcodeProduct {
  name: string;
  brand: string;
  servingSize: string;
  calories: number;
  protein_g: number;
  carbs_g: number;
  fat_g: number;
  fiber_g: number;
  ingredients: string;
  allergens: string;
  imageUrl: string | null;
}

interface OFFNutriments {
  "energy-kcal_serving"?: number;
  "energy-kcal_100g"?: number;
  proteins_serving?: number;
  proteins_100g?: number;
  carbohydrates_serving?: number;
  carbohydrates_100g?: number;
  fat_serving?: number;
  fat_100g?: number;
  fiber_serving?: number;
  fiber_100g?: number;
}

interface OFFProduct {
  product_name?: string;
  brands?: string;
  serving_size?: string;
  nutriments?: OFFNutriments;
  ingredients_text?: string;
  allergens_tags?: string[];
  image_front_url?: string;
  image_url?: string;
}

interface OFFResponse {
  status: number; // 1 = found, 0 = not found
  product?: OFFProduct;
}

/**
 * Look up a packaged food product by barcode using the OpenFoodFacts API.
 *
 * @param barcode - UPC-A, EAN-13, or other standard barcode string
 * @returns BarcodeProduct on success, null if not found
 * @throws Error only on network failure (not on 404 / empty product)
 */
export async function lookupBarcode(barcode: string): Promise<BarcodeProduct | null> {
  const trimmed = barcode.trim();
  if (!trimmed) return null;

  const url = `https://world.openfoodfacts.org/api/v2/product/${encodeURIComponent(trimmed)}.json`;

  let response: Response;
  try {
    response = await fetch(url, {
      headers: { "User-Agent": "NeuralDreamWorkshop/1.0 (https://neuraldream.app)" },
    });
  } catch (err) {
    throw new Error(`Network error looking up barcode: ${(err as Error).message}`);
  }

  if (!response.ok) {
    if (response.status === 404) return null;
    throw new Error(`OpenFoodFacts returned HTTP ${response.status}`);
  }

  let data: OFFResponse;
  try {
    data = await response.json() as OFFResponse;
  } catch {
    throw new Error("Failed to parse OpenFoodFacts response");
  }

  // status 0 means product not found in the database
  if (data.status === 0 || !data.product) return null;

  const p = data.product;
  const n = p.nutriments ?? {};

  // Prefer per-serving values; fall back to per-100g if serving data absent
  const calories  = n["energy-kcal_serving"]    ?? n["energy-kcal_100g"]      ?? 0;
  const protein   = n["proteins_serving"]        ?? n["proteins_100g"]         ?? 0;
  const carbs     = n["carbohydrates_serving"]   ?? n["carbohydrates_100g"]    ?? 0;
  const fat       = n["fat_serving"]             ?? n["fat_100g"]              ?? 0;
  const fiber     = n["fiber_serving"]           ?? n["fiber_100g"]            ?? 0;

  // Clean allergen tags: strip "en:" prefix and join
  const allergens = (p.allergens_tags ?? [])
    .map(tag => tag.replace(/^[a-z]{2}:/, ""))
    .filter(Boolean)
    .join(", ");

  return {
    name:        p.product_name?.trim() || "Unknown product",
    brand:       p.brands?.trim()       || "",
    servingSize: p.serving_size?.trim() || "",
    calories:    Math.round(calories),
    protein_g:   Math.round(protein  * 10) / 10,
    carbs_g:     Math.round(carbs    * 10) / 10,
    fat_g:       Math.round(fat      * 10) / 10,
    fiber_g:     Math.round(fiber    * 10) / 10,
    ingredients: p.ingredients_text?.trim() || "",
    allergens,
    imageUrl:    p.image_front_url ?? p.image_url ?? null,
  };
}
