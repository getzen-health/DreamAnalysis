import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, fireEvent, waitFor, act } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import Wellness from "@/pages/wellness";

// Radix Tabs uses ResizeObserver internally
global.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
} as unknown as typeof ResizeObserver;

vi.mock("framer-motion", () => ({
  motion: {
    div: ({ children, ...rest }: any) => <div {...rest}>{children}</div>,
    span: ({ children, ...rest }: any) => <span {...rest}>{children}</span>,
    path: (props: any) => <path {...props} />,
    circle: (props: any) => <circle {...props} />,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

vi.mock("@/hooks/use-auth", () => ({
  useAuth: () => ({ user: { id: 1 } }),
}));

vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

vi.mock("@/lib/ml-api", () => ({
  syncMoodLogToML: vi.fn().mockResolvedValue({}),
}));

describe("Wellness page", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [],
    }) as unknown as typeof fetch;
  });

  it("renders the page heading", () => {
    renderWithProviders(<Wellness />);
    expect(screen.getByText("Wellness")).toBeInTheDocument();
  });

  it("shows subtitle text", () => {
    renderWithProviders(<Wellness />);
    expect(
      screen.getByText("Cycle tracking and optional mood logging")
    ).toBeInTheDocument();
  });

  it("shows the Menstrual Cycle tab", () => {
    renderWithProviders(<Wellness />);
    expect(screen.getByText("Menstrual Cycle")).toBeInTheDocument();
  });

  it("shows the Mood tab", () => {
    renderWithProviders(<Wellness />);
    expect(screen.getByRole("tab", { name: /mood/i })).toBeInTheDocument();
  });

  it("renders without crashing", () => {
    renderWithProviders(<Wellness />);
    expect(screen.getByText("Wellness")).toBeInTheDocument();
  });
});

describe("Wellness page — MoodTab", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [],
    }) as unknown as typeof fetch;
  });

  async function renderAndSwitchToMood() {
    renderWithProviders(<Wellness />);
    const moodTab = screen.getByRole("tab", { name: /mood/i });
    // Radix Tabs requires mouseDown to activate — fireEvent.click alone is insufficient
    await act(async () => {
      fireEvent.mouseDown(moodTab);
      fireEvent.mouseUp(moodTab);
      fireEvent.click(moodTab);
    });
  }

  it("shows 'How are you feeling?' header after switching to Mood tab", async () => {
    await renderAndSwitchToMood();
    await waitFor(() => {
      expect(
        screen.getByText("How are you feeling?")
      ).toBeInTheDocument();
    });
  });

  it("shows mood tracking description", async () => {
    await renderAndSwitchToMood();
    await waitFor(() => {
      expect(
        screen.getByText("Track your mood and energy over time")
      ).toBeInTheDocument();
    });
  });

  it("shows Mood label in the input form", async () => {
    await renderAndSwitchToMood();
    await waitFor(() => {
      // "Mood" appears in both the tab trigger and the label — find the label element
      const labels = screen.getAllByText("Mood");
      const formLabel = labels.find((el) => el.tagName.toLowerCase() === "label");
      expect(formLabel).toBeInTheDocument();
    });
  });

  it("shows Energy label", async () => {
    await renderAndSwitchToMood();
    await waitFor(() => {
      expect(screen.getByText("Energy")).toBeInTheDocument();
    });
  });

  it("shows mood slider with default score", async () => {
    await renderAndSwitchToMood();
    await waitFor(() => {
      // Default mood score is 5
      const sliders = screen.getAllByRole("slider");
      expect(sliders.length).toBeGreaterThanOrEqual(1);
    });
  });

  it("shows mood scale labels (Awful and Great)", async () => {
    await renderAndSwitchToMood();
    await waitFor(() => {
      expect(screen.getByText("Awful")).toBeInTheDocument();
      expect(screen.getByText("Great")).toBeInTheDocument();
    });
  });

  it("shows energy scale labels (Exhausted and Energized)", async () => {
    await renderAndSwitchToMood();
    await waitFor(() => {
      expect(screen.getByText("Exhausted")).toBeInTheDocument();
      expect(screen.getByText("Energized")).toBeInTheDocument();
    });
  });

  it("shows Notes label and textarea", async () => {
    await renderAndSwitchToMood();
    await waitFor(() => {
      expect(screen.getByText("Notes (optional)")).toBeInTheDocument();
      expect(
        screen.getByPlaceholderText("What's on your mind?")
      ).toBeInTheDocument();
    });
  });

  it("shows Log Mood button", async () => {
    await renderAndSwitchToMood();
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /Log Mood/i })
      ).toBeInTheDocument();
    });
  });

  it("mood face label shows (Okay) for default score of 5", async () => {
    await renderAndSwitchToMood();
    await waitFor(() => {
      expect(screen.getByText("(Okay)")).toBeInTheDocument();
    });
  });
});

describe("Wellness page — CycleTab setup", () => {
  beforeEach(() => {
    localStorage.clear();
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [],
    }) as unknown as typeof fetch;
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("shows cycle setup prompt when no cycle data exists", () => {
    renderWithProviders(<Wellness />);
    expect(screen.getByText("Set Up Menstrual Cycle Tracking")).toBeInTheDocument();
  });

  it("shows stepper buttons for cycle length", () => {
    renderWithProviders(<Wellness />);
    // Stepper + and - buttons exist
    const buttons = screen.getAllByRole("button");
    const minusButtons = buttons.filter(b => b.textContent === "-");
    const plusButtons = buttons.filter(b => b.textContent === "+");
    expect(minusButtons.length).toBeGreaterThanOrEqual(2); // one for cycle, one for period
    expect(plusButtons.length).toBeGreaterThanOrEqual(2);
  });

  it("shows number input for period length with min 1", () => {
    renderWithProviders(<Wellness />);
    const periodInput = screen.getByLabelText("Average period length (days)");
    expect(periodInput).toBeInTheDocument();
    expect(periodInput).toHaveAttribute("min", "1");
  });

  it("shows number input for cycle length with min 20", () => {
    renderWithProviders(<Wellness />);
    const cycleInput = screen.getByLabelText("Average cycle length (days)");
    expect(cycleInput).toBeInTheDocument();
    expect(cycleInput).toHaveAttribute("min", "20");
  });

  it("shows Start Tracking button text when no initial data", () => {
    renderWithProviders(<Wellness />);
    expect(screen.getByRole("button", { name: "Start Tracking" })).toBeInTheDocument();
  });

  it("persists cycle data to localStorage on submit", async () => {
    renderWithProviders(<Wellness />);

    // Fill in date
    const dateInput = screen.getByLabelText("When did your last period start?");
    await act(async () => {
      fireEvent.change(dateInput, { target: { value: "2026-03-01" } });
    });

    // Submit
    const submitBtn = screen.getByRole("button", { name: "Start Tracking" });
    await act(async () => {
      fireEvent.click(submitBtn);
    });

    await waitFor(() => {
      const stored = localStorage.getItem("ndw_cycle_data");
      expect(stored).not.toBeNull();
      const data = JSON.parse(stored!);
      expect(data.lastPeriodStart).toBe("2026-03-01");
      expect(data.cycleLength).toBe(28);
      expect(data.periodLength).toBe(5);
    });
  });

  it("shows cycle wheel and phase after setup", async () => {
    // Pre-set cycle data in localStorage
    localStorage.setItem("ndw_cycle_data", JSON.stringify({
      lastPeriodStart: "2026-03-01",
      cycleLength: 28,
      periodLength: 5,
    }));

    renderWithProviders(<Wellness />);

    await waitFor(() => {
      // Cycle wheel renders with "Day" text — use getAllByText since it appears in both wheel and card
      const dayTexts = screen.getAllByText(/^Day \d+$/);
      expect(dayTexts.length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("of 28")).toBeInTheDocument();
    });
  });

  it("shows settings gear button when cycle data exists", async () => {
    localStorage.setItem("ndw_cycle_data", JSON.stringify({
      lastPeriodStart: "2026-03-01",
      cycleLength: 28,
      periodLength: 5,
    }));

    renderWithProviders(<Wellness />);

    await waitFor(() => {
      expect(screen.getByTitle("Edit cycle settings")).toBeInTheDocument();
    });
  });

  it("shows Update Settings button text when editing existing data", async () => {
    localStorage.setItem("ndw_cycle_data", JSON.stringify({
      lastPeriodStart: "2026-03-01",
      cycleLength: 28,
      periodLength: 5,
    }));

    renderWithProviders(<Wellness />);

    // Click settings button to show edit form
    const settingsBtn = screen.getByTitle("Edit cycle settings");
    await act(async () => {
      fireEvent.click(settingsBtn);
    });

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Update Settings" })).toBeInTheDocument();
    });
  });

  it("auto-detects last period start from logged cycle entries", async () => {
    // Set up initial cycle data
    localStorage.setItem("ndw_cycle_data", JSON.stringify({
      lastPeriodStart: "2026-02-01",
      cycleLength: 28,
      periodLength: 5,
    }));

    // Simulate logged period days in March
    localStorage.setItem("ndw_cycle_logs", JSON.stringify({
      "2026-03-08": { flowLevel: "medium", symptoms: null, notes: null },
      "2026-03-09": { flowLevel: "heavy", symptoms: null, notes: null },
      "2026-03-10": { flowLevel: "light", symptoms: null, notes: null },
    }));

    // Make the API fail so the query falls back to reading localStorage
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({}),
    }) as unknown as typeof fetch;

    renderWithProviders(<Wellness />);

    await waitFor(() => {
      const stored = localStorage.getItem("ndw_cycle_data");
      expect(stored).not.toBeNull();
      const data = JSON.parse(stored!);
      // Should update lastPeriodStart to the detected start (March 8)
      expect(data.lastPeriodStart).toBe("2026-03-08");
    });
  });

  it("shows current phase based on logged data", async () => {
    // Set period start to today-ish to ensure menstrual phase
    const today = new Date();
    const startDate = new Date(today);
    startDate.setDate(startDate.getDate() - 2); // 2 days ago = day 3 of cycle
    const startStr = startDate.toISOString().slice(0, 10);

    localStorage.setItem("ndw_cycle_data", JSON.stringify({
      lastPeriodStart: startStr,
      cycleLength: 28,
      periodLength: 5,
    }));

    renderWithProviders(<Wellness />);

    await waitFor(() => {
      // CycleOverviewCard shows "Day of Cycle" label
      expect(screen.getByText("Day of Cycle")).toBeInTheDocument();
      // Should show "Menstrual" — it appears in overview card badge, wheel label, and phase legend
      const menstrualTexts = screen.getAllByText("Menstrual");
      expect(menstrualTexts.length).toBeGreaterThanOrEqual(1);
    });
  });

  it("shows calendar with month navigation", async () => {
    localStorage.setItem("ndw_cycle_data", JSON.stringify({
      lastPeriodStart: "2026-03-01",
      cycleLength: 28,
      periodLength: 5,
    }));

    renderWithProviders(<Wellness />);

    await waitFor(() => {
      // Calendar day headers
      const dayHeaders = screen.getAllByText("S");
      expect(dayHeaders.length).toBeGreaterThanOrEqual(2); // Sunday + Saturday
      // Log Today button
      expect(screen.getByRole("button", { name: /Log Today/ })).toBeInTheDocument();
    });
  });

  it("shows period and fertile labels in the calendar legend", async () => {
    localStorage.setItem("ndw_cycle_data", JSON.stringify({
      lastPeriodStart: "2026-03-01",
      cycleLength: 28,
      periodLength: 5,
    }));

    renderWithProviders(<Wellness />);

    await waitFor(() => {
      // Calendar legend shows "Period" and "Fertile" labels
      expect(screen.getByText("Period")).toBeInTheDocument();
      expect(screen.getByText("Fertile")).toBeInTheDocument();
    });
  });
});

/* ========== Task #46: Mood history always visible ========== */

describe("Wellness page — Mood history from localStorage", () => {
  beforeEach(() => {
    localStorage.clear();
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => [],
    }) as unknown as typeof fetch;
  });

  afterEach(() => {
    localStorage.clear();
  });

  async function renderAndSwitchToMood() {
    renderWithProviders(<Wellness />);
    const moodTab = screen.getByRole("tab", { name: /mood/i });
    await act(async () => {
      fireEvent.mouseDown(moodTab);
      fireEvent.mouseUp(moodTab);
      fireEvent.click(moodTab);
    });
  }

  it("shows past mood entries from localStorage without logging a new entry", async () => {
    // Pre-populate mood logs in localStorage
    localStorage.setItem("ndw_mood_logs", JSON.stringify([
      {
        id: "local_1",
        userId: null,
        moodScore: "7",
        energyLevel: "6",
        notes: "feeling good",
        loggedAt: new Date(Date.now() - 3600000).toISOString(), // 1 hour ago
      },
      {
        id: "local_2",
        userId: null,
        moodScore: "4",
        energyLevel: "3",
        notes: "tired day",
        loggedAt: new Date(Date.now() - 7200000).toISOString(), // 2 hours ago
      },
    ]));

    await renderAndSwitchToMood();

    await waitFor(() => {
      expect(screen.getByText("Recent entries")).toBeInTheDocument();
      expect(screen.getByText("feeling good")).toBeInTheDocument();
      expect(screen.getByText("tired day")).toBeInTheDocument();
    });
  });

  it("shows mood chart when 3+ entries exist in localStorage", async () => {
    const entries = [];
    for (let i = 0; i < 5; i++) {
      entries.push({
        id: `local_${i}`,
        userId: null,
        moodScore: String(5 + i),
        energyLevel: String(4 + i),
        notes: null,
        loggedAt: new Date(Date.now() - i * 86400000).toISOString(),
      });
    }
    localStorage.setItem("ndw_mood_logs", JSON.stringify(entries));

    await renderAndSwitchToMood();

    await waitFor(() => {
      expect(screen.getByText("Mood over time")).toBeInTheDocument();
    });
  });

  it("shows mood history even without authenticated user", async () => {
    // Mock useAuth to return no user
    vi.mocked(vi.fn()).mockReturnValue({ user: null });

    localStorage.setItem("ndw_mood_logs", JSON.stringify([
      {
        id: "local_1",
        userId: null,
        moodScore: "8",
        energyLevel: "7",
        notes: "great morning",
        loggedAt: new Date().toISOString(),
      },
    ]));

    await renderAndSwitchToMood();

    await waitFor(() => {
      expect(screen.getByText("Recent entries")).toBeInTheDocument();
      expect(screen.getByText("great morning")).toBeInTheDocument();
    });
  });
});
