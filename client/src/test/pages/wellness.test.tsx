import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
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
