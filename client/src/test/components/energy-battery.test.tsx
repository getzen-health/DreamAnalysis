import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { EnergyBattery } from "@/components/energy-battery";

vi.mock("framer-motion", () => ({
  motion: {
    rect: (props: any) => <rect {...props} />,
    div: ({ children, ...rest }: any) => <div {...rest}>{children}</div>,
  },
  useMotionValue: (initial: number) => ({
    get: () => initial,
    set: vi.fn(),
  }),
  useTransform: (_mv: any, fn: (v: number) => number) => fn(0),
  animate: () => ({ stop: vi.fn() }),
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

describe("EnergyBattery", () => {
  it("renders the value when given a number", () => {
    renderWithProviders(<EnergyBattery value={75} />);
    expect(screen.getByText("75")).toBeInTheDocument();
  });

  it("renders em-dash when value is null", () => {
    renderWithProviders(<EnergyBattery value={null} />);
    expect(screen.getByText("\u2014")).toBeInTheDocument();
  });

  it("shows the Energy Bank label", () => {
    renderWithProviders(<EnergyBattery value={50} />);
    expect(screen.getByText("Energy Bank")).toBeInTheDocument();
  });

  it("has correct aria-label with value", () => {
    renderWithProviders(<EnergyBattery value={60} />);
    expect(screen.getByLabelText("Energy Bank: 60")).toBeInTheDocument();
  });

  it("has correct aria-label when null", () => {
    renderWithProviders(<EnergyBattery value={null} />);
    expect(screen.getByLabelText("Energy Bank: no data")).toBeInTheDocument();
  });

  it("shows percent sign when value is present", () => {
    renderWithProviders(<EnergyBattery value={80} />);
    expect(screen.getByText("%")).toBeInTheDocument();
  });

  it("does not show percent sign when null", () => {
    renderWithProviders(<EnergyBattery value={null} />);
    expect(screen.queryByText("%")).not.toBeInTheDocument();
  });

  it("rounds the displayed value", () => {
    renderWithProviders(<EnergyBattery value={82.6} />);
    expect(screen.getByText("83")).toBeInTheDocument();
  });
});
