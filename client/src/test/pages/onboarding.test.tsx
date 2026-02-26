import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import Onboarding from "@/pages/onboarding";

vi.mock("@/hooks/use-device", () => ({
  useDevice: () => ({
    latestFrame: null,
    state: "disconnected",
    deviceStatus: null,
  }),
}));

vi.mock("wouter", () => ({
  useLocation: () => ["/onboarding", vi.fn()],
  Link: (props: any) => <a href={props.href}>{props.children}</a>,
}));

vi.mock("@/lib/ml-api", () => ({
  getBaselineStatus: vi.fn().mockResolvedValue({ ready: false, n_frames: 0 }),
  addBaselineFrame: vi.fn().mockResolvedValue({}),
  simulateEEG: vi.fn().mockResolvedValue({ signals: [[]] }),
}));

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [],
    })
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Onboarding page — intro phase (device not connected)", () => {
  it("renders without crashing", () => {
    renderWithProviders(<Onboarding />);
  });

  it("shows the Brain Setup heading", () => {
    renderWithProviders(<Onboarding />);
    expect(screen.getByText("Brain Setup")).toBeInTheDocument();
  });

  it("shows the 2 minutes tagline", () => {
    renderWithProviders(<Onboarding />);
    expect(screen.getByText(/2 minutes · Done once/)).toBeInTheDocument();
  });

  it("shows the accuracy improvement claim", () => {
    renderWithProviders(<Onboarding />);
    expect(
      screen.getByText(/15–29% more accurate emotion readings/)
    ).toBeInTheDocument();
  });

  it("shows the no-headset simulation notice when device not connected", () => {
    renderWithProviders(<Onboarding />);
    expect(
      screen.getByText(/No headset detected — will use EEG simulation/)
    ).toBeInTheDocument();
  });

  it("shows the 'What you'll do' checklist items", () => {
    renderWithProviders(<Onboarding />);
    expect(screen.getByText("Sit still and close your eyes")).toBeInTheDocument();
    expect(screen.getByText("Breathe naturally for 2 minutes")).toBeInTheDocument();
    expect(
      screen.getByText("No headset required — simulation works too")
    ).toBeInTheDocument();
  });

  it("shows the Start calibration button", () => {
    renderWithProviders(<Onboarding />);
    expect(
      screen.getByRole("button", { name: /Start calibration/i })
    ).toBeInTheDocument();
  });

  it("shows the Skip link", () => {
    renderWithProviders(<Onboarding />);
    expect(
      screen.getByText(/Skip for now — readings will be less accurate/)
    ).toBeInTheDocument();
  });

  it("transitions to recording phase when Start calibration is clicked", () => {
    renderWithProviders(<Onboarding />);
    const startBtn = screen.getByRole("button", { name: /Start calibration/i });
    fireEvent.click(startBtn);
    expect(screen.getByText("Recording your baseline…")).toBeInTheDocument();
  });

  it("shows simulation mode notice in recording phase when no device", () => {
    renderWithProviders(<Onboarding />);
    const startBtn = screen.getByRole("button", { name: /Start calibration/i });
    fireEvent.click(startBtn);
    expect(
      screen.getByText(/Simulation mode — no headset needed/)
    ).toBeInTheDocument();
  });

  it("shows the frame progress counter in recording phase", () => {
    renderWithProviders(<Onboarding />);
    const startBtn = screen.getByRole("button", { name: /Start calibration/i });
    fireEvent.click(startBtn);
    // Shows "0 / 120 frames" initially before any frames are collected
    expect(screen.getByText(/0 \/ 120 frames/)).toBeInTheDocument();
  });
});
