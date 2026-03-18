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

vi.mock("@/hooks/use-health-sync", () => ({
  useHealthSync: () => ({
    status: "idle",
    lastSyncAt: null,
    latestPayload: null,
    error: null,
    syncNow: vi.fn(),
    isAvailable: true,
  }),
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
    expect(screen.getByText("Workouts")).toBeInTheDocument();
  });

  it("shows Health Sync card", () => {
    renderWithProviders(<WorkoutPage />);
    expect(screen.getByText("Health Sync")).toBeInTheDocument();
  });

  it("shows Sync Workouts button", () => {
    renderWithProviders(<WorkoutPage />);
    expect(
      screen.getByRole("button", { name: /Sync Workouts/i })
    ).toBeInTheDocument();
  });

  it("shows empty state when no workouts imported", async () => {
    renderWithProviders(<WorkoutPage />);
    await waitFor(() => {
      expect(screen.getByText("No workouts yet")).toBeInTheDocument();
    });
  });

  it("shows auto-import message in empty state", () => {
    renderWithProviders(<WorkoutPage />);
    expect(
      screen.getByText(/automatically imported from Google Health or Apple Health/)
    ).toBeInTheDocument();
  });

  it("does not show manual workout creation", () => {
    renderWithProviders(<WorkoutPage />);
    expect(screen.queryByText("Start Workout")).not.toBeInTheDocument();
    expect(screen.queryByText("Workout Name (optional)")).not.toBeInTheDocument();
  });
});
