import { describe, it, expect, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import CalibrationPage from "@/pages/calibration";

vi.mock("@/hooks/use-device", () => ({
  useDevice: () => ({
    latestFrame: null,
    state: "disconnected",
    deviceStatus: null,
  }),
}));

vi.mock("@/hooks/use-theme", () => ({
  useTheme: () => ({ theme: "dark", setTheme: vi.fn() }),
}));

vi.mock("wouter", () => ({
  useLocation: () => ["/calibration", vi.fn()],
  Link: (props: any) => <a href={props.href}>{props.children}</a>,
}));

vi.mock("@/lib/ml-api", () => ({
  addBaselineFrame: vi.fn().mockResolvedValue({ n_frames: 0, ready: false }),
  getBaselineStatus: vi.fn().mockResolvedValue({ n_frames: 0, ready: false }),
  resetBaselineCalibration: vi.fn().mockResolvedValue({}),
  simulateEEG: vi.fn().mockResolvedValue({ signals: [] }),
}));

describe("CalibrationPage", () => {
  it("renders without crashing", () => {
    renderWithProviders(<CalibrationPage />);
  });

  it("shows the main heading", () => {
    renderWithProviders(<CalibrationPage />);
    expect(screen.getByText("Baseline Calibration")).toBeInTheDocument();
  });

  it("shows the accuracy improvement description", () => {
    renderWithProviders(<CalibrationPage />);
    expect(
      screen.getByText(/improves emotion accuracy by up to 29%/)
    ).toBeInTheDocument();
  });

  it("shows mode selector with both options", () => {
    renderWithProviders(<CalibrationPage />);
    // Use exact button names to avoid ambiguity with other text on page
    expect(screen.getByRole("button", { name: /EEG headband/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Simulation \(no device\)/ })).toBeInTheDocument();
  });

  it("shows device not connected warning after switching to real-device mode", () => {
    renderWithProviders(<CalibrationPage />);
    // Page auto-enables simulation mode on mount (no device). Switch to real-device.
    const museBtn = screen.getByRole("button", { name: /EEG headband/ });
    fireEvent.click(museBtn);
    expect(screen.getByText("EEG headband not connected")).toBeInTheDocument();
  });

  it("shows 'Not started' ring label in idle state", () => {
    renderWithProviders(<CalibrationPage />);
    expect(screen.getByText("Not started")).toBeInTheDocument();
  });

  it("shows device warning in real-device mode when disconnected", () => {
    renderWithProviders(<CalibrationPage />);
    // Page starts in real-device mode — shows warning when no device connected
    expect(screen.getByText(/not connected/i)).toBeInTheDocument();
  });

  it("shows Connect device first button when not streaming", () => {
    renderWithProviders(<CalibrationPage />);
    // No device connected → button says "Connect device first"
    expect(
      screen.getByRole("button", { name: /Connect device first/ })
    ).toBeInTheDocument();
  });
});
