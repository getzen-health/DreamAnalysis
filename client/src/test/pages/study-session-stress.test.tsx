import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import StudySessionStress from "@/pages/study/StudySessionStress";

beforeAll(() => {
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

// Mock wouter
const navigateMock = vi.fn();
vi.mock("wouter", () => ({
  useLocation: () => ["/study/session/stress", navigateMock],
  useSearch: () => "?code=P101",
}));

// Mock toast
vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

// Mock apiRequest — session start returns id, complete resolves
vi.mock("@/lib/queryClient", () => ({
  apiRequest: vi.fn().mockImplementation((_method: string, url: string) => {
    if (url === "/api/study/session/start") {
      return Promise.resolve({ json: () => Promise.resolve({ session_id: 42 }) });
    }
    if (url === "/api/study/session/complete") {
      return Promise.resolve({ json: () => Promise.resolve({ ok: true }) });
    }
    return Promise.resolve({ json: () => Promise.resolve({}) });
  }),
}));

// Mock fetch for EEG simulation
beforeEach(() => {
  navigateMock.mockClear();
  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: () =>
      Promise.resolve({
        alpha: 0.4,
        beta: 0.3,
        theta: 0.15,
        delta: 0.08,
        gamma: 0.03,
        stress_level: 0.5,
      }),
  });
});

describe("StudySessionStress", () => {
  it("renders loading state initially", () => {
    renderWithProviders(<StudySessionStress />);
    expect(screen.getByText(/setting up your session/i)).toBeTruthy();
  });

  it("shows baseline phase after session starts", async () => {
    renderWithProviders(<StudySessionStress />);
    await waitFor(() => {
      expect(screen.getByText(/baseline recording/i)).toBeTruthy();
    });
  });

  it("shows eyes closed instruction during baseline", async () => {
    renderWithProviders(<StudySessionStress />);
    await waitFor(() => {
      expect(screen.getByText(/sit still and close your eyes/i)).toBeTruthy();
    });
  });

  it("displays stepper with 5 phases", async () => {
    renderWithProviders(<StudySessionStress />);
    await waitFor(() => {
      expect(screen.getByText("Baseline")).toBeTruthy();
      expect(screen.getByText("Work")).toBeTruthy();
      expect(screen.getByText("Breathing")).toBeTruthy();
      expect(screen.getByText("Post-session")).toBeTruthy();
      expect(screen.getByText("Survey")).toBeTruthy();
    });
  });

  it("shows ~28 minute session label", async () => {
    renderWithProviders(<StudySessionStress />);
    await waitFor(() => {
      expect(screen.getByText(/~28 minute session/i)).toBeTruthy();
    });
  });

  it("shows sample count during baseline", async () => {
    renderWithProviders(<StudySessionStress />);
    await waitFor(() => {
      expect(screen.getByText(/samples/i)).toBeTruthy();
    });
  });
});
