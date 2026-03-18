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

  it("shows the Synced from Health card", () => {
    renderWithProviders(<BodyMetrics />);
    expect(screen.getByText("Synced from Health")).toBeInTheDocument();
  });

  it("shows Sync Now button", () => {
    renderWithProviders(<BodyMetrics />);
    expect(
      screen.getByRole("button", { name: /Sync Now/i })
    ).toBeInTheDocument();
  });

  it("shows last synced indicator", () => {
    renderWithProviders(<BodyMetrics />);
    expect(screen.getByText(/Last synced:/)).toBeInTheDocument();
  });

  it("shows empty state when no data synced", async () => {
    renderWithProviders(<BodyMetrics />);
    await waitFor(() => {
      expect(screen.getByText("No body metrics yet")).toBeInTheDocument();
    });
  });

  it("shows prompt to sync when no metrics available", () => {
    renderWithProviders(<BodyMetrics />);
    expect(
      screen.getByText(/No body metrics synced yet/)
    ).toBeInTheDocument();
  });

  it("does not show any manual weight input", () => {
    renderWithProviders(<BodyMetrics />);
    expect(screen.queryByText("Log Weight")).not.toBeInTheDocument();
    expect(screen.queryByText("Body Fat % (optional)")).not.toBeInTheDocument();
  });
});
