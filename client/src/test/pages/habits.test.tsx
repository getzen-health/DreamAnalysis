import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import Habits from "@/pages/habits";

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

describe("Habits page", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [],
    }) as unknown as typeof fetch;
  });

  it("renders the page heading", () => {
    renderWithProviders(<Habits />);
    expect(screen.getByText("Habits")).toBeInTheDocument();
  });

  it("shows the Add button", () => {
    renderWithProviders(<Habits />);
    expect(screen.getByRole("button", { name: /Add/i })).toBeInTheDocument();
  });

  it("shows subtitle text", () => {
    renderWithProviders(<Habits />);
    expect(
      screen.getByText("Track daily habits and build streaks")
    ).toBeInTheDocument();
  });

  it("shows empty state when no habits exist", async () => {
    renderWithProviders(<Habits />);
    await waitFor(() => {
      expect(screen.getByText("No habits yet")).toBeInTheDocument();
    });
  });

  it("shows instruction to add first habit in empty state", async () => {
    renderWithProviders(<Habits />);
    await waitFor(() => {
      expect(
        screen.getByText(/Tap "Add" to create your first habit/i)
      ).toBeInTheDocument();
    });
  });
});
