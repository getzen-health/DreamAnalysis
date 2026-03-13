import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { FoodCapture } from "@/components/food-capture";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { FoodImageAnalysisResult } from "@/lib/ml-api";

// ── Module mocks ──────────────────────────────────────────────────────────────

vi.mock("@/lib/ml-api", () => ({
  analyzeFoodImage: vi.fn(),
}));

vi.mock("@/lib/haptics", () => ({
  hapticSuccess: vi.fn(),
}));

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false } } });
}

function renderComponent(props: Parameters<typeof FoodCapture>[0] = {}) {
  const client = makeClient();
  return render(
    <QueryClientProvider client={client}>
      <FoodCapture {...props} />
    </QueryClientProvider>
  );
}

const MOCK_RESULT: FoodImageAnalysisResult = {
  food_items: [
    {
      name: "Grilled Chicken",
      portion: "150g",
      calories: 248,
      protein_g: 46.5,
      carbs_g: 0,
      fat_g: 5.4,
      fiber_g: 0,
    },
    {
      name: "Brown Rice",
      portion: "1 cup",
      calories: 216,
      protein_g: 5,
      carbs_g: 45,
      fat_g: 1.8,
      fiber_g: 3.5,
    },
  ],
  total_calories: 464,
  total_protein_g: 51.5,
  total_carbs_g: 45,
  total_fat_g: 7.2,
  total_fiber_g: 3.5,
  dominant_macro: "protein",
  glycemic_impact: "low",
  confidence: 0.85,
  analysis_method: "vision_ai",
  summary: "Grilled chicken with brown rice — 464 kcal estimated.",
};

/**
 * Simulates selecting a file via the hidden file input.
 * Stubs FileReader so onload is called synchronously in the same tick,
 * avoiding flaky timeouts inside jsdom.
 *
 * The component reads `reader.result` (not the onload event argument), so the
 * stub must set `this.result` before calling `this.onload()`.
 */
async function simulateFileSelect(
  fileInput: HTMLInputElement,
  dataUrl = "data:image/jpeg;base64,/9j/fakeimage"
) {
  // Stub FileReader BEFORE triggering the event so the component picks it up
  class SyncFileReader {
    result: string | null = null;
    onload: (() => void) | null = null;
    readAsDataURL(_file: File) {
      // Set result first — the component reads `reader.result` in the callback
      this.result = dataUrl;
      this.onload?.();
    }
  }

  vi.stubGlobal("FileReader", SyncFileReader);

  const file = new File(["fake"], "meal.jpg", { type: "image/jpeg" });
  Object.defineProperty(fileInput, "files", {
    value: [file],
    configurable: true,
  });

  await act(async () => {
    fireEvent.change(fileInput);
  });
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("FoodCapture", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it("renders the camera button initially", () => {
    renderComponent();
    expect(screen.getByRole("button", { name: /capture meal/i })).toBeInTheDocument();
  });

  it("shows image preview after file selection", async () => {
    renderComponent();

    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    expect(fileInput).toBeInTheDocument();

    await simulateFileSelect(fileInput);

    expect(screen.getByAltText("Meal preview")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^analyze$/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /retake/i })).toBeInTheDocument();
  });

  it("displays nutritional results after successful analysis", async () => {
    const { analyzeFoodImage } = await import("@/lib/ml-api");
    vi.mocked(analyzeFoodImage).mockResolvedValueOnce(MOCK_RESULT);

    renderComponent();

    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    await simulateFileSelect(fileInput);

    expect(screen.getByRole("button", { name: /^analyze$/i })).toBeInTheDocument();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^analyze$/i }));
    });

    await waitFor(() => {
      expect(screen.getByText(/nutritional breakdown/i)).toBeInTheDocument();
    });

    // Calories
    expect(screen.getByText("464")).toBeInTheDocument();
    // Summary
    expect(screen.getByText(/grilled chicken with brown rice/i)).toBeInTheDocument();
    // Badges
    expect(screen.getByText(/protein-dominant/i)).toBeInTheDocument();
    expect(screen.getByText(/gi: low/i)).toBeInTheDocument();
    // Food items
    expect(screen.getByText("Grilled Chicken")).toBeInTheDocument();
    expect(screen.getByText("Brown Rice")).toBeInTheDocument();
  });

  it("handles API errors gracefully", async () => {
    const { analyzeFoodImage } = await import("@/lib/ml-api");
    vi.mocked(analyzeFoodImage).mockRejectedValueOnce(
      new Error("Could not identify any food items.")
    );

    renderComponent();

    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    await simulateFileSelect(fileInput);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^analyze$/i }));
    });

    await waitFor(() => {
      expect(screen.getByText(/could not identify any food items/i)).toBeInTheDocument();
    });

    // Try Again button should appear
    expect(screen.getByRole("button", { name: /try again/i })).toBeInTheDocument();
  });

  it("calls onAnalyzed callback with result", async () => {
    const { analyzeFoodImage } = await import("@/lib/ml-api");
    vi.mocked(analyzeFoodImage).mockResolvedValueOnce(MOCK_RESULT);

    const onAnalyzed = vi.fn();
    renderComponent({ onAnalyzed });

    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    await simulateFileSelect(fileInput);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^analyze$/i }));
    });

    await waitFor(() => {
      expect(onAnalyzed).toHaveBeenCalledWith(MOCK_RESULT);
    });
  });
});
