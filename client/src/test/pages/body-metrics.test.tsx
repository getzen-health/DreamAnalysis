import { describe, it, expect, vi, beforeAll, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import BodyMetrics from "@/pages/body-metrics";

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
    circle: (props: any) => <circle {...props} />,
    rect: (props: any) => <rect {...props} />,
  },
  useMotionValue: (initial: number) => ({
    get: () => initial,
    set: vi.fn(),
  }),
  useTransform: (_mv: any, fn: (v: number) => number) => fn(0),
  animate: () => ({ stop: vi.fn() }),
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

vi.mock("@/hooks/use-auth", () => ({
  useAuth: () => ({ user: { id: 1 } }),
}));

vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

describe("BodyMetrics page", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [],
    }) as unknown as typeof fetch;
  });

  it("renders the page heading", () => {
    renderWithProviders(<BodyMetrics />);
    expect(screen.getByText("Body Metrics")).toBeInTheDocument();
  });

  it("shows the Log Weight section heading", () => {
    renderWithProviders(<BodyMetrics />);
    // The section label and the button both say "Log Weight"
    const matches = screen.getAllByText("Log Weight");
    expect(matches.length).toBeGreaterThanOrEqual(1);
  });

  it("shows weight input field", () => {
    renderWithProviders(<BodyMetrics />);
    expect(screen.getByText("Weight")).toBeInTheDocument();
  });

  it("shows body fat input field", () => {
    renderWithProviders(<BodyMetrics />);
    expect(screen.getByText("Body Fat % (optional)")).toBeInTheDocument();
  });

  it("shows the Log Weight button", () => {
    renderWithProviders(<BodyMetrics />);
    expect(
      screen.getByRole("button", { name: /Log Weight/i })
    ).toBeInTheDocument();
  });

  it("shows empty state when no history data", async () => {
    renderWithProviders(<BodyMetrics />);
    await waitFor(() => {
      expect(screen.getByText("No entries yet")).toBeInTheDocument();
    });
  });

  it("shows kg/lbs toggle button", () => {
    renderWithProviders(<BodyMetrics />);
    expect(
      screen.getByText(/kg.*tap to switch/i)
    ).toBeInTheDocument();
  });
});
