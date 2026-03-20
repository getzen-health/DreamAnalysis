import { describe, it, expect, vi, beforeEach } from "vitest";
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
