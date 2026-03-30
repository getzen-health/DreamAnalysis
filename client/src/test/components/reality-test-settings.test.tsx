import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { RealityTestSettings } from "@/components/reality-test-settings";

// ── Polyfill ResizeObserver for jsdom (required by Radix Slider) ─────────────

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = globalThis.ResizeObserver ?? ResizeObserverStub as any;

// ── Mock supabase-store ──────────────────────────────────────────────────────

const mockStore: Record<string, string> = {};

vi.mock("@/lib/supabase-store", () => ({
  sbGetSetting: vi.fn((key: string) => mockStore[key] ?? null),
  sbSaveSetting: vi.fn((key: string, value: string) => {
    mockStore[key] = value;
  }),
}));

beforeEach(() => {
  Object.keys(mockStore).forEach((k) => delete mockStore[k]);
});

// ── Tests ────────────────────────────────────────────────────────────────────

describe("RealityTestSettings", () => {
  it("renders with data-testid", () => {
    renderWithProviders(<RealityTestSettings />);
    expect(screen.getByTestId("reality-test-settings")).toBeInTheDocument();
  });

  it("shows header text", () => {
    renderWithProviders(<RealityTestSettings />);
    expect(screen.getByText("Reality Testing Reminders")).toBeInTheDocument();
  });

  it("has a toggle switch", () => {
    renderWithProviders(<RealityTestSettings />);
    expect(screen.getByTestId("reality-test-toggle")).toBeInTheDocument();
  });

  it("does not show settings when disabled (default)", () => {
    renderWithProviders(<RealityTestSettings />);
    expect(screen.queryByTestId("frequency-slider")).not.toBeInTheDocument();
    expect(screen.queryByTestId("schedule-preview")).not.toBeInTheDocument();
  });

  it("shows settings panel after toggling on", () => {
    renderWithProviders(<RealityTestSettings />);
    const toggle = screen.getByTestId("reality-test-toggle");
    fireEvent.click(toggle);
    expect(screen.getByTestId("frequency-slider")).toBeInTheDocument();
    expect(screen.getByTestId("frequency-value")).toBeInTheDocument();
    expect(screen.getByTestId("start-hour-select")).toBeInTheDocument();
    expect(screen.getByTestId("end-hour-select")).toBeInTheDocument();
    expect(screen.getByTestId("randomize-toggle")).toBeInTheDocument();
  });

  it("shows schedule preview when enabled", () => {
    renderWithProviders(<RealityTestSettings />);
    fireEvent.click(screen.getByTestId("reality-test-toggle"));
    expect(screen.getByTestId("schedule-preview")).toBeInTheDocument();
    const slots = screen.getAllByTestId("schedule-slot");
    // Default frequency is 5
    expect(slots.length).toBe(5);
  });

  it("loads saved config from store", () => {
    mockStore["ndw_reality_test_config"] = JSON.stringify({
      enabled: true,
      frequency: 7,
      startHour: 10,
      endHour: 20,
      randomize: false,
    });
    renderWithProviders(<RealityTestSettings />);
    // Should show settings since enabled is true
    expect(screen.getByTestId("frequency-slider")).toBeInTheDocument();
    expect(screen.getByTestId("frequency-value")).toHaveTextContent("7");
    const slots = screen.getAllByTestId("schedule-slot");
    expect(slots.length).toBe(7);
  });

  it("displays frequency label", () => {
    renderWithProviders(<RealityTestSettings />);
    fireEvent.click(screen.getByTestId("reality-test-toggle"));
    expect(screen.getByText("Tests per day")).toBeInTheDocument();
  });

  it("displays active hours label", () => {
    renderWithProviders(<RealityTestSettings />);
    fireEvent.click(screen.getByTestId("reality-test-toggle"));
    expect(screen.getByText("Active hours")).toBeInTheDocument();
  });

  it("displays randomize label", () => {
    renderWithProviders(<RealityTestSettings />);
    fireEvent.click(screen.getByTestId("reality-test-toggle"));
    expect(screen.getByText("Randomize timing")).toBeInTheDocument();
  });

  it("each schedule slot contains a time and prompt", () => {
    mockStore["ndw_reality_test_config"] = JSON.stringify({
      enabled: true,
      frequency: 3,
      startHour: 9,
      endHour: 21,
      randomize: true,
    });
    renderWithProviders(<RealityTestSettings />);
    const slots = screen.getAllByTestId("schedule-slot");
    expect(slots.length).toBe(3);
    for (const slot of slots) {
      // Each slot should have text content (time + prompt)
      expect(slot.textContent!.length).toBeGreaterThan(0);
    }
  });
});
