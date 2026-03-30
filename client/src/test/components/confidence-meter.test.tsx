import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { ConfidenceMeter } from "@/components/confidence-meter";

describe("ConfidenceMeter", () => {
  // 1. Renders green for high confidence (>0.7)
  it("renders green bar and 'High confidence' label for confidence > 0.7", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.85} showLabel />);
    const meter = screen.getByTestId("confidence-meter");
    expect(meter).toBeInTheDocument();
    expect(screen.getByText("High confidence")).toBeInTheDocument();
    const bar = screen.getByTestId("confidence-bar-fill");
    expect(bar.className).toMatch(/emerald/);
  });

  // 2. Renders amber for moderate confidence (0.4-0.7)
  it("renders amber bar and 'Moderate' label for confidence 0.4-0.7", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.6} showLabel />);
    expect(screen.getByText("Moderate")).toBeInTheDocument();
    const bar = screen.getByTestId("confidence-bar-fill");
    expect(bar.className).toMatch(/amber/);
  });

  // 3. Renders red for low confidence (<0.4)
  it("renders red bar and 'Low' label for confidence < 0.4", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.2} showLabel />);
    const label = screen.getByTestId("confidence-label");
    expect(label.textContent).toContain("Low");
    expect(label.textContent).toContain("grain of salt");
    const bar = screen.getByTestId("confidence-bar-fill");
    expect(bar.className).toMatch(/red/);
  });

  // 4. Amber at boundary (0.4 exactly)
  it("renders amber at the 0.4 boundary", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.4} showLabel />);
    expect(screen.getByText("Moderate")).toBeInTheDocument();
    const bar = screen.getByTestId("confidence-bar-fill");
    expect(bar.className).toMatch(/amber/);
  });

  // 5. Respects size prop — sm (default)
  it("renders sm bar by default (h-1)", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.8} />);
    const meter = screen.getByTestId("confidence-meter");
    expect(meter.className).toMatch(/h-1/);
  });

  // 6. Respects size prop — md
  it("renders medium bar with md size", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.8} size="md" />);
    const meter = screen.getByTestId("confidence-meter");
    expect(meter.className).toMatch(/h-2/);
  });

  // 7. Hides label when showLabel is false
  it("does not render label text when showLabel is false", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.85} showLabel={false} />);
    expect(screen.queryByText("High confidence")).not.toBeInTheDocument();
  });

  // 8. Default showLabel is false
  it("defaults to hiding label when showLabel is not provided", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.85} />);
    expect(screen.queryByText("High confidence")).not.toBeInTheDocument();
  });

  // 9. Bar width reflects confidence percentage
  it("sets bar fill width proportional to confidence", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.65} />);
    const bar = screen.getByTestId("confidence-bar-fill");
    expect(bar.style.width).toBe("65%");
  });

  // 10. Shows warning text for low confidence when label is shown
  it("shows warning text for confidence below 0.4 when showLabel is true", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.25} showLabel />);
    const warning = screen.getByTestId("confidence-warning");
    expect(warning).toBeInTheDocument();
  });

  it("does not show warning for confidence above 0.4", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.75} showLabel />);
    expect(screen.queryByTestId("confidence-warning")).not.toBeInTheDocument();
  });
});
