import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import { screen, fireEvent, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import StudySessionFood from "@/pages/study/StudySessionFood";

beforeAll(() => {
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

const navigateMock = vi.fn();
vi.mock("wouter", () => ({
  useLocation: () => ["/study/session/food", navigateMock],
  useSearch: () => "?code=P102",
}));

vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

vi.mock("@/lib/queryClient", () => ({
  apiRequest: vi.fn().mockImplementation((_method: string, url: string) => {
    if (url === "/api/study/session/start") {
      return Promise.resolve({ json: () => Promise.resolve({ session_id: 55 }) });
    }
    if (url === "/api/study/session/complete") {
      return Promise.resolve({ json: () => Promise.resolve({ ok: true }) });
    }
    return Promise.resolve({ json: () => Promise.resolve({}) });
  }),
}));

beforeEach(() => {
  navigateMock.mockClear();
  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: () =>
      Promise.resolve({
        alpha: 0.35,
        beta: 0.25,
        theta: 0.12,
        delta: 0.07,
        gamma: 0.03,
        stress_level: 0.4,
      }),
  });
});

describe("StudySessionFood", () => {
  it("starts on pre-survey phase", async () => {
    renderWithProviders(<StudySessionFood />);
    await waitFor(() => {
      expect(screen.getByText(/pre-meal survey/i)).toBeTruthy();
    });
  });

  it("shows hunger and mood sliders", async () => {
    renderWithProviders(<StudySessionFood />);
    await waitFor(() => {
      expect(screen.getByText(/how hungry are you right now/i)).toBeTruthy();
      expect(screen.getByText(/what is your current mood/i)).toBeTruthy();
    });
  });

  it("shows eating duration selector", async () => {
    renderWithProviders(<StudySessionFood />);
    await waitFor(() => {
      expect(screen.getByText("15 min")).toBeTruthy();
      expect(screen.getByText("20 min")).toBeTruthy();
      expect(screen.getByText("25 min")).toBeTruthy();
      expect(screen.getByText("30 min")).toBeTruthy();
    });
  });

  it("has start baseline button", async () => {
    renderWithProviders(<StudySessionFood />);
    await waitFor(() => {
      expect(screen.getByText(/start baseline recording/i)).toBeTruthy();
    });
  });

  it("transitions to baseline phase when clicking start", async () => {
    renderWithProviders(<StudySessionFood />);
    await waitFor(() => {
      expect(screen.getByText(/start baseline recording/i)).toBeTruthy();
    });
    fireEvent.click(screen.getByText(/start baseline recording/i));
    await waitFor(() => {
      expect(screen.getByText(/pre-meal baseline/i)).toBeTruthy();
    });
  });

  it("shows eyes closed instruction in baseline", async () => {
    renderWithProviders(<StudySessionFood />);
    await waitFor(() => {
      fireEvent.click(screen.getByText(/start baseline recording/i));
    });
    await waitFor(() => {
      expect(screen.getByText(/sit still with your eyes closed/i)).toBeTruthy();
    });
  });

  it("displays 5-step stepper", async () => {
    renderWithProviders(<StudySessionFood />);
    await waitFor(() => {
      expect(screen.getByText("Pre-survey")).toBeTruthy();
      expect(screen.getByText("Baseline")).toBeTruthy();
      expect(screen.getByText("Eat")).toBeTruthy();
      expect(screen.getByText("Post EEG")).toBeTruthy();
      expect(screen.getByText("Post-survey")).toBeTruthy();
    });
  });

  it("shows participant code badge", async () => {
    renderWithProviders(<StudySessionFood />);
    await waitFor(() => {
      expect(screen.getByText("P102")).toBeTruthy();
    });
  });
});
