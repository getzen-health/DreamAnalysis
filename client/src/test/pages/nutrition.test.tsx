import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import { screen, fireEvent, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import Nutrition from "@/pages/nutrition";

// ResizeObserver polyfill for Recharts
beforeAll(() => {
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

// Mock framer-motion to avoid animation complexity in tests
vi.mock("framer-motion", () => ({
  motion: {
    div: ({ children, ...rest }: any) => <div {...rest}>{children}</div>,
    path: (props: any) => <path {...props} />,
    circle: (props: any) => <circle {...props} />,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

vi.mock("@/lib/participant", () => ({
  getParticipantId: () => "test-user-nutrition",
}));

vi.mock("@/lib/haptics", () => ({
  hapticSuccess: vi.fn(),
}));

vi.mock("@/hooks/use-voice-data", () => ({
  useVoiceData: () => null,
}));

vi.mock("@/hooks/use-scores", () => ({
  useScores: () => ({ scores: { nutritionScore: 72 } }),
}));

vi.mock("@/lib/ml-api", () => ({
  syncFoodLogToML: vi.fn().mockResolvedValue({}),
}));

vi.mock("@/lib/barcode-api", () => ({
  lookupBarcode: vi.fn(),
}));

vi.mock("@/lib/animations", () => ({
  cardVariants: {},
  listItemVariants: {},
}));

vi.mock("@/components/score-gauge", () => ({
  ScoreGauge: ({ value, label }: { value: number | null; label: string }) => (
    <div data-testid="score-gauge">
      <span>{label}</span>
      <span>{value ?? "--"}</span>
    </div>
  ),
}));

vi.mock("recharts", () => ({
  AreaChart: ({ children }: any) => <div data-testid="area-chart">{children}</div>,
  Area: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  ResponsiveContainer: ({ children }: any) => <div>{children}</div>,
}));

const STORAGE_KEY = "ndw_food_logs_test-user-nutrition";
const SUPPLEMENTS_KEY = "ndw_supplements";
const GLP1_KEY = "ndw_glp1_injections";

function makeFoodLog(overrides: Record<string, unknown> = {}) {
  return {
    id: `log_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    loggedAt: new Date().toISOString(),
    mealType: "lunch",
    summary: "Chicken and rice",
    totalCalories: 450,
    dominantMacro: "protein",
    foodItems: [
      { name: "Chicken breast", portion: "1 serving", calories: 230, protein_g: 30, carbs_g: 0, fat_g: 10 },
      { name: "Rice", portion: "1 serving", calories: 210, protein_g: 4, carbs_g: 45, fat_g: 1 },
    ],
    vitamins: null,
    ...overrides,
  };
}

describe("Nutrition page", () => {
  beforeEach(() => {
    // Clear localStorage before each test
    localStorage.clear();

    // Mock fetch to return empty arrays by default
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [],
    }) as unknown as typeof fetch;
  });

  // ── Task 1: Meal history display ──────────────────────────────────────────

  describe("Task 1: Meal history display", () => {
    it("shows '0 calories' and 'No meals logged today' when no meals exist", () => {
      renderWithProviders(<Nutrition />);
      expect(screen.getByText("0 calories")).toBeInTheDocument();
      expect(screen.getByText("No meals logged today")).toBeInTheDocument();
    });

    it("displays meals with timestamps and calorie breakdown when meals are logged", async () => {
      const log1 = makeFoodLog({ id: "meal_1", summary: "Chicken and rice", totalCalories: 450 });
      const log2 = makeFoodLog({ id: "meal_2", summary: "Oatmeal with berries", totalCalories: 280, mealType: "breakfast" });
      localStorage.setItem(STORAGE_KEY, JSON.stringify([log1, log2]));

      renderWithProviders(<Nutrition />);

      await waitFor(() => {
        // Meals may appear in both Today's Meals and Recent Meals, so use getAllByText
        const chickenElements = screen.getAllByText("Chicken and rice");
        expect(chickenElements.length).toBeGreaterThanOrEqual(1);
        const oatmealElements = screen.getAllByText("Oatmeal with berries");
        expect(oatmealElements.length).toBeGreaterThanOrEqual(1);
      });

      // Should show calorie amounts
      expect(screen.getAllByText("450 kcal").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("280 kcal").length).toBeGreaterThanOrEqual(1);
    });

    it("shows daily running totals when meals are logged", async () => {
      const log1 = makeFoodLog({ id: "meal_t1", totalCalories: 450 });
      const log2 = makeFoodLog({ id: "meal_t2", totalCalories: 280 });
      localStorage.setItem(STORAGE_KEY, JSON.stringify([log1, log2]));

      renderWithProviders(<Nutrition />);

      await waitFor(() => {
        // Daily totals section shows combined calories -- may appear in ring + totals
        const calElements = screen.getAllByText("730");
        expect(calElements.length).toBeGreaterThanOrEqual(1);
        // Should show meal count
        expect(screen.getByText("2 meals logged today")).toBeInTheDocument();
      });
    });
  });

  // ── Task 2 & 5: Vitamins tab — combined food + supplement sources ─────────

  describe("Task 2 & 5: Vitamins tab with food + supplement sources", () => {
    it("shows the Vitamins tab button", () => {
      renderWithProviders(<Nutrition />);
      expect(screen.getByText("Vitamins")).toBeInTheDocument();
    });

    it("shows 'No supplements taken today' message when no supplements are logged", async () => {
      // Log a meal so food sources exist
      const log = makeFoodLog({
        id: "vit_test_1",
        foodItems: [{ name: "Salmon", portion: "1 serving", calories: 200, protein_g: 28, carbs_g: 0, fat_g: 8 }],
      });
      localStorage.setItem(STORAGE_KEY, JSON.stringify([log]));

      renderWithProviders(<Nutrition />);

      // Switch to Vitamins tab
      fireEvent.click(screen.getByText("Vitamins"));

      await waitFor(() => {
        expect(screen.getByText("Micronutrients")).toBeInTheDocument();
        expect(screen.getByText(/No supplements taken today/)).toBeInTheDocument();
      });
    });

    it("shows supplement source indicator when supplements are taken", async () => {
      // Set up a supplement and mark it as taken
      const supp = {
        id: "supp_vitd",
        name: "Vitamin D",
        dosage: "2000 IU",
        timeOfDay: "morning",
        type: "vitamin",
      };
      localStorage.setItem(SUPPLEMENTS_KEY, JSON.stringify([supp]));
      const todayKey = `ndw_supplement_log_${new Date().toISOString().slice(0, 10)}`;
      localStorage.setItem(todayKey, JSON.stringify({ supp_vitd: true }));

      renderWithProviders(<Nutrition />);

      // Switch to Vitamins tab
      fireEvent.click(screen.getByText("Vitamins"));

      await waitFor(() => {
        expect(screen.getByText("Supplements (active)")).toBeInTheDocument();
      });
    });
  });

  // ── Task 3: Food Quality Score — auto-calculate ───────────────────────────

  describe("Task 3: Food Quality Score auto-calculation", () => {
    it("displays quality score when meals are logged", async () => {
      const log = makeFoodLog({ id: "quality_1", totalCalories: 450 });
      localStorage.setItem(STORAGE_KEY, JSON.stringify([log]));

      renderWithProviders(<Nutrition />);

      await waitFor(() => {
        // Should show the Quality badge in daily totals
        expect(screen.getByText("Quality")).toBeInTheDocument();
      });
    });

    it("does not show quality score section when no meals logged", () => {
      renderWithProviders(<Nutrition />);
      // The daily totals section (which contains the quality score) should not render
      expect(screen.queryByText("Daily Totals")).not.toBeInTheDocument();
    });

    it("uses calculated score in the Insights tab ScoreGauge", async () => {
      const log = makeFoodLog({
        id: "insights_q1",
        foodItems: [
          { name: "Chicken breast", portion: "1 serving", calories: 230, protein_g: 30, carbs_g: 0, fat_g: 10 },
          { name: "Broccoli", portion: "1 serving", calories: 60, protein_g: 3, carbs_g: 10, fat_g: 1 },
        ],
        totalCalories: 290,
      });
      localStorage.setItem(STORAGE_KEY, JSON.stringify([log]));

      renderWithProviders(<Nutrition />);

      // Switch to Insights tab
      fireEvent.click(screen.getByText("Insights"));

      await waitFor(() => {
        const gauge = screen.getByTestId("score-gauge");
        expect(gauge).toBeInTheDocument();
        expect(gauge.textContent).toContain("Food Quality");
      });
    });

    it("persists quality score to localStorage", async () => {
      const log = makeFoodLog({ id: "persist_q1" });
      localStorage.setItem(STORAGE_KEY, JSON.stringify([log]));

      renderWithProviders(<Nutrition />);

      await waitFor(() => {
        const stored = localStorage.getItem("ndw_food_quality_score");
        expect(stored).not.toBeNull();
        const parsed = JSON.parse(stored!);
        expect(parsed.score).toBeGreaterThan(0);
        expect(parsed.date).toBeDefined();
      });
    });
  });

  // ── Task 4: GLP-1 injection tracker ───────────────────────────────────────

  describe("Task 4: GLP-1 injection tracker", () => {
    it("shows GLP-1 injection tracker in Supplements tab", () => {
      renderWithProviders(<Nutrition />);

      // Switch to Supplements tab
      fireEvent.click(screen.getByText("Supplements"));

      expect(screen.getByText("GLP-1 Injection Tracker")).toBeInTheDocument();
      expect(screen.getByText("+ Log GLP-1 Injection")).toBeInTheDocument();
    });

    it("can log a GLP-1 injection", async () => {
      renderWithProviders(<Nutrition />);
      fireEvent.click(screen.getByText("Supplements"));

      // Click the add button
      fireEvent.click(screen.getByText("+ Log GLP-1 Injection"));

      // The form should appear with a Log Injection button
      expect(screen.getByText("Log Injection")).toBeInTheDocument();

      // Click Log Injection
      fireEvent.click(screen.getByText("Log Injection"));

      await waitFor(() => {
        // Should show at least 1 logged
        expect(screen.getByText("1 logged")).toBeInTheDocument();
        // Should persist to localStorage
        const stored = localStorage.getItem(GLP1_KEY);
        expect(stored).not.toBeNull();
        const injections = JSON.parse(stored!);
        expect(injections).toHaveLength(1);
        expect(injections[0].medication).toBe("Ozempic"); // default first option
      });
    });

    it("shows decay visualization and next injection date after logging", async () => {
      // Pre-populate with an injection from 3 days ago
      const threeDaysAgo = new Date();
      threeDaysAgo.setDate(threeDaysAgo.getDate() - 3);
      const injection = {
        id: "glp1_test_1",
        medication: "Ozempic",
        dose: "0.5 mg",
        date: threeDaysAgo.toISOString(),
      };
      localStorage.setItem(GLP1_KEY, JSON.stringify([injection]));

      renderWithProviders(<Nutrition />);
      fireEvent.click(screen.getByText("Supplements"));

      await waitFor(() => {
        // Should show Ozempic section with decay level
        expect(screen.getByText("Ozempic")).toBeInTheDocument();
        // Should show next injection info
        expect(screen.getByText(/Next injection|Due/)).toBeInTheDocument();
        // Should show the half-life info
        expect(screen.getByText(/half-life/)).toBeInTheDocument();
      });
    });

    it("shows overdue reminder when injection is past schedule", async () => {
      // Pre-populate with an injection from 10 days ago (overdue for weekly)
      const tenDaysAgo = new Date();
      tenDaysAgo.setDate(tenDaysAgo.getDate() - 10);
      const injection = {
        id: "glp1_overdue_1",
        medication: "Wegovy",
        dose: "1.7 mg",
        date: tenDaysAgo.toISOString(),
      };
      localStorage.setItem(GLP1_KEY, JSON.stringify([injection]));

      renderWithProviders(<Nutrition />);
      fireEvent.click(screen.getByText("Supplements"));

      await waitFor(() => {
        expect(screen.getByText(/Overdue/)).toBeInTheDocument();
      });
    });

    it("shows informational text when no injections logged", () => {
      renderWithProviders(<Nutrition />);
      fireEvent.click(screen.getByText("Supplements"));

      expect(screen.getByText(/Track GLP-1 injections/)).toBeInTheDocument();
    });
  });

  // ── General page rendering ─────────────────────────────────────────────────

  describe("General rendering", () => {
    it("renders the page title", () => {
      renderWithProviders(<Nutrition />);
      expect(screen.getByText("Nutrition")).toBeInTheDocument();
    });

    it("shows all five tabs", () => {
      renderWithProviders(<Nutrition />);
      expect(screen.getByText("Log")).toBeInTheDocument();
      expect(screen.getByText("Vitamins")).toBeInTheDocument();
      expect(screen.getByText("Supplements")).toBeInTheDocument();
      expect(screen.getByText("Insights")).toBeInTheDocument();
      expect(screen.getByText("History")).toBeInTheDocument();
    });

    it("shows calorie ring with 0 when no meals logged", () => {
      renderWithProviders(<Nutrition />);
      // The calorie ring displays "0" and "kcal remaining" shows full goal
      expect(screen.getByText("2000 kcal remaining")).toBeInTheDocument();
    });

    it("can switch between tabs", () => {
      renderWithProviders(<Nutrition />);

      fireEvent.click(screen.getByText("Supplements"));
      expect(screen.getByText("GLP-1 Injection Tracker")).toBeInTheDocument();

      fireEvent.click(screen.getByText("Vitamins"));
      expect(screen.getByText("Micronutrients")).toBeInTheDocument();

      fireEvent.click(screen.getByText("Log"));
      expect(screen.getByText("Today's Meals")).toBeInTheDocument();
    });

    it("shows Scan Your Meal button on Log tab", () => {
      renderWithProviders(<Nutrition />);
      expect(screen.getByText("Scan Your Meal")).toBeInTheDocument();
    });

    it("shows Describe Meal and Scan Barcode buttons", () => {
      renderWithProviders(<Nutrition />);
      expect(screen.getByText("Describe Meal")).toBeInTheDocument();
      expect(screen.getByText("Scan Barcode")).toBeInTheDocument();
    });
  });

  // ── Food log persistence ────────────────────────────────────────────────────

  describe("Food log persistence", () => {
    it("persists food logs to localStorage when API analyze succeeds", async () => {
      // Mock fetch: GET /api/food/logs returns empty; POST /api/food/analyze returns a log
      const analyzeResponse = {
        id: 42,
        loggedAt: new Date().toISOString(),
        mealType: "lunch",
        summary: "Chicken and rice",
        totalCalories: 450,
        dominantMacro: "protein",
        foodItems: [
          { name: "Chicken breast", portion: "1 serving", calories: 230, protein_g: 30, carbs_g: 0, fat_g: 10 },
          { name: "Rice", portion: "1 serving", calories: 210, protein_g: 4, carbs_g: 45, fat_g: 1 },
        ],
        vitamins: null,
      };

      global.fetch = vi.fn().mockImplementation((url: string, opts?: RequestInit) => {
        if (opts?.method === "POST" || (typeof url === "string" && url.includes("/api/food/analyze"))) {
          return Promise.resolve({
            ok: true,
            json: async () => analyzeResponse,
          });
        }
        // GET for food logs
        return Promise.resolve({
          ok: true,
          json: async () => [],
        });
      }) as unknown as typeof fetch;

      renderWithProviders(<Nutrition />);

      // Open text description mode
      fireEvent.click(screen.getByText("Describe Meal"));

      await waitFor(() => {
        expect(screen.getByPlaceholderText(/rice bowl/i)).toBeInTheDocument();
      });

      // Type a meal description
      const textarea = screen.getByPlaceholderText(/rice bowl/i);
      fireEvent.change(textarea, { target: { value: "chicken and rice" } });

      // Click Log Meal
      fireEvent.click(screen.getByText("Log Meal"));

      // Wait for the async operation to complete
      await waitFor(() => {
        const stored = localStorage.getItem(STORAGE_KEY);
        expect(stored).not.toBeNull();
        const logs = JSON.parse(stored!);
        expect(logs.length).toBeGreaterThanOrEqual(1);
        const savedLog = logs.find((l: any) => String(l.id) === "42");
        expect(savedLog).toBeDefined();
        expect(savedLog.totalCalories).toBe(450);
        expect(savedLog.summary).toBe("Chicken and rice");
      });
    });

    it("persists food logs to localStorage when API fails (fallback path)", async () => {
      // Mock fetch: all POST calls fail, GET returns empty
      global.fetch = vi.fn().mockImplementation((_url: string, opts?: RequestInit) => {
        if (opts?.method === "POST") {
          return Promise.resolve({
            ok: false,
            status: 503,
            json: async () => ({ message: "OpenAI not configured" }),
            text: async () => "OpenAI not configured",
            statusText: "Service Unavailable",
          });
        }
        return Promise.resolve({
          ok: true,
          json: async () => [],
        });
      }) as unknown as typeof fetch;

      renderWithProviders(<Nutrition />);

      // Open text description mode
      fireEvent.click(screen.getByText("Describe Meal"));

      await waitFor(() => {
        expect(screen.getByPlaceholderText(/rice bowl/i)).toBeInTheDocument();
      });

      // Type a meal description
      const textarea = screen.getByPlaceholderText(/rice bowl/i);
      fireEvent.change(textarea, { target: { value: "chicken and rice" } });

      // Click Log Meal
      fireEvent.click(screen.getByText("Log Meal"));

      // Wait for the fallback to persist to localStorage
      await waitFor(() => {
        const stored = localStorage.getItem(STORAGE_KEY);
        expect(stored).not.toBeNull();
        const logs = JSON.parse(stored!);
        expect(logs.length).toBeGreaterThanOrEqual(1);
        // Fallback uses local estimation — should have a local_ prefix id
        expect(logs[0].id).toMatch(/^local_/);
        expect(logs[0].totalCalories).toBeGreaterThan(0);
        expect(logs[0].summary).toBeDefined();
      });
    });

    it("does not create duplicate entries in localStorage", async () => {
      // Pre-populate with an existing log
      const existingLog = makeFoodLog({ id: "42" });
      localStorage.setItem(STORAGE_KEY, JSON.stringify([existingLog]));

      const analyzeResponse = {
        id: 42,
        loggedAt: new Date().toISOString(),
        mealType: "lunch",
        summary: "Same meal",
        totalCalories: 450,
        dominantMacro: "protein",
        foodItems: [],
        vitamins: null,
      };

      global.fetch = vi.fn().mockImplementation((_url: string, opts?: RequestInit) => {
        if (opts?.method === "POST") {
          return Promise.resolve({
            ok: true,
            json: async () => analyzeResponse,
          });
        }
        return Promise.resolve({
          ok: true,
          json: async () => [existingLog],
        });
      }) as unknown as typeof fetch;

      renderWithProviders(<Nutrition />);

      // Open text description mode
      fireEvent.click(screen.getByText("Describe Meal"));

      await waitFor(() => {
        expect(screen.getByPlaceholderText(/rice bowl/i)).toBeInTheDocument();
      });

      const textarea = screen.getByPlaceholderText(/rice bowl/i);
      fireEvent.change(textarea, { target: { value: "same meal again" } });
      fireEvent.click(screen.getByText("Log Meal"));

      await waitFor(() => {
        const stored = localStorage.getItem(STORAGE_KEY);
        const logs = JSON.parse(stored!);
        // Should not have duplicates with same id
        const id42Count = logs.filter((l: any) => String(l.id) === "42").length;
        expect(id42Count).toBe(1);
      });
    });

    it("loads and displays meals from localStorage when API is unavailable", async () => {
      // Pre-populate localStorage with meals
      const log = makeFoodLog({ id: "offline_1", summary: "Offline pasta", totalCalories: 350 });
      localStorage.setItem(STORAGE_KEY, JSON.stringify([log]));

      // API returns error
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({ message: "Server error" }),
        text: async () => "Server error",
        statusText: "Internal Server Error",
      }) as unknown as typeof fetch;

      renderWithProviders(<Nutrition />);

      await waitFor(() => {
        // Should still display the localStorage meal
        const pastaElements = screen.getAllByText("Offline pasta");
        expect(pastaElements.length).toBeGreaterThanOrEqual(1);
        expect(screen.getAllByText("350 kcal").length).toBeGreaterThanOrEqual(1);
      });
    });

    it("shows 0 calories and empty state message when nothing is logged", () => {
      renderWithProviders(<Nutrition />);
      expect(screen.getByText("0 calories")).toBeInTheDocument();
      expect(screen.getByText("No meals logged today")).toBeInTheDocument();
    });

    it("merges API logs with localStorage logs without duplicates", async () => {
      // localStorage has one log
      const localLog = makeFoodLog({ id: "local_111", summary: "Local meal", totalCalories: 200 });
      localStorage.setItem(STORAGE_KEY, JSON.stringify([localLog]));

      // API returns a different log
      const apiLog = makeFoodLog({ id: "api_222", summary: "API meal", totalCalories: 300 });

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => [apiLog],
      }) as unknown as typeof fetch;

      renderWithProviders(<Nutrition />);

      await waitFor(() => {
        // Both meals should appear
        const localElements = screen.getAllByText("Local meal");
        expect(localElements.length).toBeGreaterThanOrEqual(1);
        const apiElements = screen.getAllByText("API meal");
        expect(apiElements.length).toBeGreaterThanOrEqual(1);
      });
    });
  });
});
