import { describe, it, expect, vi, beforeAll, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import WorkoutPage from "@/pages/workout";

beforeAll(() => {
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

vi.mock("framer-motion", () => ({
  motion: {
    div: ({ children, ...rest }: any) => <div {...rest}>{children}</div>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

vi.mock("@/hooks/use-auth", () => ({
  useAuth: () => ({ user: { id: 1 } }),
}));

vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

describe("Workout page", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [],
    }) as unknown as typeof fetch;
  });

  it("renders the page heading", () => {
    renderWithProviders(<WorkoutPage />);
    expect(screen.getByText("Strength Builder")).toBeInTheDocument();
  });

  it("shows Start Workout card title and button", () => {
    renderWithProviders(<WorkoutPage />);
    // Both the card title and the action button contain "Start Workout"
    const matches = screen.getAllByText(/Start Workout/);
    expect(matches.length).toBeGreaterThanOrEqual(2);
  });

  it("shows the Start Workout action button", () => {
    renderWithProviders(<WorkoutPage />);
    const buttons = screen.getAllByRole("button", { name: /Start Workout/i });
    // At least the action button exists (there may also be the card title)
    expect(buttons.length).toBeGreaterThanOrEqual(1);
  });

  it("shows workout type selector buttons", () => {
    renderWithProviders(<WorkoutPage />);
    expect(screen.getByRole("button", { name: "Strength" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cardio" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "HIIT" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Flexibility" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Mixed" })).toBeInTheDocument();
  });

  it("shows Workout and History tabs", () => {
    renderWithProviders(<WorkoutPage />);
    expect(screen.getByText("Workout")).toBeInTheDocument();
    expect(screen.getByText("History")).toBeInTheDocument();
  });

  it("shows workout name input", () => {
    renderWithProviders(<WorkoutPage />);
    expect(screen.getByText("Workout Name (optional)")).toBeInTheDocument();
  });
});
