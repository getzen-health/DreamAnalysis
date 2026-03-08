import { describe, it, expect, vi, beforeAll } from "vitest";
import { screen, fireEvent, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import StudyProfile from "@/pages/study/StudyProfile";

beforeAll(() => {
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

const navigateMock = vi.fn();
vi.mock("wouter", () => ({
  useLocation: () => ["/study/profile", navigateMock],
  useSearch: () => "?code=P200",
}));

vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

vi.mock("@/lib/queryClient", () => ({
  apiRequest: vi.fn().mockResolvedValue({
    json: () => Promise.resolve({ ok: true }),
  }),
}));

describe("StudyProfile", () => {
  it("renders the profile form", () => {
    renderWithProviders(<StudyProfile />);
    expect(screen.getByText(/your profile/i)).toBeTruthy();
  });

  it("shows participant code badge", () => {
    renderWithProviders(<StudyProfile />);
    expect(screen.getByText("P200")).toBeTruthy();
  });

  it("has age input field", () => {
    renderWithProviders(<StudyProfile />);
    expect(screen.getByLabelText(/age/i)).toBeTruthy();
  });

  it("has apple watch toggle", () => {
    renderWithProviders(<StudyProfile />);
    expect(screen.getAllByText(/apple watch/i).length).toBeGreaterThan(0);
  });

  it("has both session buttons", () => {
    renderWithProviders(<StudyProfile />);
    expect(screen.getByText(/start stress session/i)).toBeTruthy();
    expect(screen.getByText(/start food session/i)).toBeTruthy();
  });

  it("shows correct session durations", () => {
    renderWithProviders(<StudyProfile />);
    const durations = screen.getAllByText(/~20 min/i);
    expect(durations.length).toBeGreaterThanOrEqual(2);
  });

  it("disables buttons when age is empty", () => {
    renderWithProviders(<StudyProfile />);
    const stressBtn = screen.getByText(/start stress session/i).closest("button");
    expect(stressBtn?.disabled).toBe(true);
  });

  it("enables buttons after valid age is entered", async () => {
    renderWithProviders(<StudyProfile />);
    const ageInput = screen.getByLabelText(/age/i);
    fireEvent.change(ageInput, { target: { value: "25" } });
    await waitFor(() => {
      const stressBtn = screen.getByText(/start stress session/i).closest("button");
      expect(stressBtn?.disabled).toBe(false);
    });
  });

  it("shows validation error for invalid age", async () => {
    renderWithProviders(<StudyProfile />);
    const ageInput = screen.getByLabelText(/age/i);
    fireEvent.change(ageInput, { target: { value: "5" } });
    await waitFor(() => {
      expect(screen.getByText(/must be between 18 and 99/i)).toBeTruthy();
    });
  });

  it("navigates to stress session on click", async () => {
    renderWithProviders(<StudyProfile />);
    fireEvent.change(screen.getByLabelText(/age/i), { target: { value: "25" } });
    await waitFor(() => {
      const btn = screen.getByText(/start stress session/i).closest("button");
      expect(btn?.disabled).toBe(false);
    });
    fireEvent.click(screen.getByText(/start stress session/i));
    await waitFor(() => {
      expect(navigateMock).toHaveBeenCalledWith("/study/session?code=P200&block=stress");
    });
  });
});
