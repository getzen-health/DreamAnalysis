import { describe, it, expect, vi, beforeAll } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import Biofeedback from "@/pages/biofeedback";

beforeAll(() => {
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

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
  useLocation: () => ["/biofeedback", vi.fn()],
  Link: (props: any) => <a href={props.href}>{props.children}</a>,
}));

describe("Biofeedback page", () => {
  it("renders without crashing", () => {
    renderWithProviders(<Biofeedback />);
  });

  it("shows the main heading", () => {
    renderWithProviders(<Biofeedback />);
    expect(screen.getByText("Biofeedback")).toBeInTheDocument();
  });

  it("shows the subheading description", () => {
    renderWithProviders(<Biofeedback />);
    expect(
      screen.getByText("Watch your stress respond in real time as you breathe")
    ).toBeInTheDocument();
  });

  it("shows all four breathing exercises", () => {
    renderWithProviders(<Biofeedback />);
    expect(screen.getByText("Coherence")).toBeInTheDocument();
    expect(screen.getByText("4-7-8")).toBeInTheDocument();
    expect(screen.getByText("Box")).toBeInTheDocument();
    expect(screen.getByText("Deep Relax")).toBeInTheDocument();
  });

  it("shows no-device banner when Muse 2 not connected", () => {
    renderWithProviders(<Biofeedback />);
    expect(
      screen.getByText(/EEG is optional later for live stress tracking/)
    ).toBeInTheDocument();
  });

  it("shows a start button", () => {
    renderWithProviders(<Biofeedback />);
    const startButtons = screen.getAllByRole("button", { name: /start/i });
    expect(startButtons.length).toBeGreaterThan(0);
  });

  it("selecting a different exercise keeps it rendered", () => {
    renderWithProviders(<Biofeedback />);
    const boxExercise = screen.getByText("Box");
    fireEvent.click(boxExercise);
    expect(screen.getByText("Box")).toBeInTheDocument();
  });
});
