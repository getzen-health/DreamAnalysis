import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import TPBMSessionPage from "@/pages/tpbm-session";

// Mock wouter
vi.mock("wouter", () => ({
  useLocation: () => ["/tpbm", vi.fn()],
  Link: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

describe("TPBMSessionPage", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("renders the page header", () => {
    renderWithProviders(<TPBMSessionPage />);
    expect(screen.getByText("tPBM Sessions")).toBeDefined();
    expect(
      screen.getByText(/Track transcranial photobiomodulation/)
    ).toBeDefined();
  });

  it("shows empty state when no sessions", () => {
    renderWithProviders(<TPBMSessionPage />);
    expect(screen.getByText(/No tPBM sessions logged/)).toBeDefined();
  });

  it("opens the log form when clicking the button", () => {
    renderWithProviders(<TPBMSessionPage />);
    const logBtn = screen.getByRole("button", { name: /Log New.*Session/i });
    fireEvent.click(logBtn);
    expect(screen.getByText("Log New Session")).toBeDefined();
  });

  it("saves a session to localStorage", () => {
    renderWithProviders(<TPBMSessionPage />);

    // Open form
    fireEvent.click(screen.getByRole("button", { name: /Log New.*Session/i }));

    // Fill and save
    fireEvent.click(screen.getByRole("button", { name: /Save tPBM session/i }));

    // Verify session saved to localStorage
    const raw = localStorage.getItem("ndw_tpbm_sessions");
    expect(raw).not.toBeNull();
    const sessions = JSON.parse(raw!);
    expect(sessions).toHaveLength(1);
    expect(sessions[0].device).toBe("Vielight Neuro Gamma");
    expect(sessions[0].durationMinutes).toBe(20);
  });

  it("displays saved sessions", () => {
    // Pre-seed a session
    const sessions = [
      {
        id: "test_1",
        date: new Date().toISOString(),
        durationMinutes: 15,
        device: "Joovv Go",
        preMood: 4,
        postMood: 7,
      },
    ];
    localStorage.setItem("ndw_tpbm_sessions", JSON.stringify(sessions));

    renderWithProviders(<TPBMSessionPage />);
    expect(screen.getByText("Joovv Go")).toBeDefined();
    // "15 min" appears in the session card badge
    expect(screen.getAllByText("15 min").length).toBeGreaterThanOrEqual(1);
  });

  it("shows stats when sessions exist", () => {
    const sessions = [
      {
        id: "test_1",
        date: new Date().toISOString(),
        durationMinutes: 20,
        device: "Vielight Neuro Gamma",
        preMood: 4,
        postMood: 7,
      },
    ];
    localStorage.setItem("ndw_tpbm_sessions", JSON.stringify(sessions));

    renderWithProviders(<TPBMSessionPage />);
    expect(screen.getByText("Total Sessions")).toBeDefined();
    expect(screen.getByText("Avg Duration")).toBeDefined();
  });
});
