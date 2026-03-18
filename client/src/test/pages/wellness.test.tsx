import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import Wellness from "@/pages/wellness";

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
      screen.getByText("Cycle tracking and mood logging")
    ).toBeInTheDocument();
  });

  it("shows the Cycle tab", () => {
    renderWithProviders(<Wellness />);
    expect(screen.getByText("Cycle")).toBeInTheDocument();
  });

  it("shows the Mood tab", () => {
    renderWithProviders(<Wellness />);
    expect(screen.getByText("Mood")).toBeInTheDocument();
  });

  it("shows Log Today button in cycle tab by default", () => {
    renderWithProviders(<Wellness />);
    expect(screen.getByText("Log Today")).toBeInTheDocument();
  });

  it("shows day-of-week headers in the calendar", () => {
    renderWithProviders(<Wellness />);
    // The calendar shows S M T W T F S headers
    const dayHeaders = screen.getAllByText("S");
    expect(dayHeaders.length).toBeGreaterThanOrEqual(2); // Sunday and Saturday
  });

  it("shows flow level legend in cycle tab", () => {
    renderWithProviders(<Wellness />);
    expect(screen.getByText("Light")).toBeInTheDocument();
    expect(screen.getByText("Medium")).toBeInTheDocument();
    expect(screen.getByText("Heavy")).toBeInTheDocument();
  });
});
