import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { ConfidenceMeter } from "@/components/confidence-meter";

describe("ConfidenceMeter", () => {
  // 1. Renders green for high confidence
  it("renders green bar and 'High confidence' label for confidence > 0.7", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.85} showLabel />);
    const meter = screen.getByTestId("confidence-meter");
    expect(meter).toBeInTheDocument();
    expect(screen.getByText("High confidence")).toBeInTheDocument();
    const bar = screen.getByTestId("confidence-bar-fill");
    expect(bar.className).toMatch(/emerald/);
  });

  // 2. Renders amber for moderate confidence
  it("renders amber bar and 'Moderate confidence' label for confidence 0.5-0.7", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.6} showLabel />);
    expect(screen.getByText("Moderate confidence")).toBeInTheDocument();
    const bar = screen.getByTestId("confidence-bar-fill");
    expect(bar.className).toMatch(/amber/);
  });

  // 3. Renders red for low confidence
  it("renders red bar and 'Low confidence' label for confidence 0.3-0.5", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.4} showLabel />);
    expect(screen.getByText("Low confidence")).toBeInTheDocument();
    const bar = screen.getByTestId("confidence-bar-fill");
    expect(bar.className).toMatch(/red/);
  });

  // 4. Shows "Not enough data" below threshold
  it("shows 'Not enough data' when confidence is below 0.3", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.15} showLabel />);
    const label = screen.getByTestId("confidence-label");
    expect(label.textContent).toBe("Not enough data");
  });

  // 5. Respects size prop
  it("renders smaller bar with sm size", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.8} size="sm" />);
    const meter = screen.getByTestId("confidence-meter");
    expect(meter.className).toMatch(/h-1/);
  });

  it("renders medium bar with md size (default)", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.8} size="md" />);
    const meter = screen.getByTestId("confidence-meter");
    expect(meter.className).toMatch(/h-2/);
  });

  // 6. Hides label when showLabel is false
  it("does not render label text when showLabel is false", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.85} showLabel={false} />);
    expect(screen.queryByText("High confidence")).not.toBeInTheDocument();
  });

  // 7. Default showLabel is false
  it("defaults to hiding label when showLabel is not provided", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.85} />);
    expect(screen.queryByText("High confidence")).not.toBeInTheDocument();
  });

  // 8. Bar width reflects confidence percentage
  it("sets bar fill width proportional to confidence", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.65} />);
    const bar = screen.getByTestId("confidence-bar-fill");
    expect(bar.style.width).toBe("65%");
  });

  // 9. Shows warning text for low confidence when label is shown
  it("shows warning text for confidence below 0.5 when showLabel is true", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.35} showLabel />);
    const warning = screen.getByTestId("confidence-warning");
    expect(warning).toBeInTheDocument();
  });

  it("does not show warning for confidence above 0.5", () => {
    renderWithProviders(<ConfidenceMeter confidence={0.75} showLabel />);
    expect(screen.queryByTestId("confidence-warning")).not.toBeInTheDocument();
  });
});
