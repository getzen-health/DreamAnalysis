import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { EnsembleExplanation } from "@/components/ensemble-explanation";

describe("EnsembleExplanation", () => {
  // 1. Returns null when no contributions provided
  it("renders nothing when contributions are undefined", () => {
    const { container } = renderWithProviders(<EnsembleExplanation />);
    expect(container.innerHTML).toBe("");
  });

  // 2. Returns null when only one contribution provided
  it("renders nothing when only eegnetContribution is provided", () => {
    const { container } = renderWithProviders(
      <EnsembleExplanation eegnetContribution={0.6} />
    );
    expect(container.innerHTML).toBe("");
  });

  // 3. Renders explanation text with both contributions
  it("renders contribution percentages when both values are present", () => {
    renderWithProviders(
      <EnsembleExplanation
        eegnetContribution={0.63}
        heuristicContribution={0.37}
      />
    );
    const wrapper = screen.getByTestId("ensemble-explanation");
    expect(wrapper).toBeInTheDocument();
    expect(wrapper.textContent).toContain("63%");
    expect(wrapper.textContent).toContain("37%");
    expect(wrapper.textContent).toContain("brain wave patterns");
    expect(wrapper.textContent).toContain("neuroscience rules");
  });

  // 4. Renders stacked bar segments
  it("renders EEGNet and heuristic bar segments with correct widths", () => {
    renderWithProviders(
      <EnsembleExplanation
        eegnetContribution={0.7}
        heuristicContribution={0.3}
      />
    );
    const eegBar = screen.getByTestId("eegnet-bar");
    const heuBar = screen.getByTestId("heuristic-bar");
    expect(eegBar.style.width).toBe("70%");
    expect(heuBar.style.width).toBe("30%");
  });

  // 5. Shows quality warning for noisy signal
  it("shows quality note when epoch quality is poor", () => {
    renderWithProviders(
      <EnsembleExplanation
        eegnetContribution={0.3}
        heuristicContribution={0.7}
        epochQuality={0.2}
      />
    );
    const note = screen.getByTestId("quality-note");
    expect(note.textContent).toContain("noisy signal");
  });

  // 6. Shows moderate quality note
  it("shows moderate quality note for mid-range signal", () => {
    renderWithProviders(
      <EnsembleExplanation
        eegnetContribution={0.5}
        heuristicContribution={0.5}
        epochQuality={0.4}
      />
    );
    const note = screen.getByTestId("quality-note");
    expect(note.textContent).toContain("moderate signal quality");
  });

  // 7. No quality note when signal is good
  it("does not show quality note when signal quality is good", () => {
    renderWithProviders(
      <EnsembleExplanation
        eegnetContribution={0.6}
        heuristicContribution={0.4}
        epochQuality={0.8}
      />
    );
    expect(screen.queryByTestId("quality-note")).toBeNull();
  });

  // 8. Handles edge case: 100% EEGNet
  it("handles 100% EEGNet contribution", () => {
    renderWithProviders(
      <EnsembleExplanation
        eegnetContribution={1.0}
        heuristicContribution={0.0}
      />
    );
    const eegBar = screen.getByTestId("eegnet-bar");
    const heuBar = screen.getByTestId("heuristic-bar");
    expect(eegBar.style.width).toBe("100%");
    expect(heuBar.style.width).toBe("0%");
  });

  // 9. Rounds percentages correctly
  it("rounds decimal contributions to whole percentages", () => {
    renderWithProviders(
      <EnsembleExplanation
        eegnetContribution={0.6667}
        heuristicContribution={0.3333}
      />
    );
    const wrapper = screen.getByTestId("ensemble-explanation");
    expect(wrapper.textContent).toContain("67%");
    expect(wrapper.textContent).toContain("33%");
  });
});
