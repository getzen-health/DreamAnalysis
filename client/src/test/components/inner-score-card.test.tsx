import React from "react";
import { describe, it, expect } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { InnerScoreCard } from "@/components/inner-score-card";

describe("InnerScoreCard", () => {
  it("renders without crashing", () => {
    renderWithProviders(
      <InnerScoreCard score={null} tier="voice" factors={{}} narrative="" delta={null} trend={[]} />,
    );
    expect(screen.getByTestId("inner-score-card")).toBeInTheDocument();
  });

  it("shows building state when score is null", () => {
    renderWithProviders(
      <InnerScoreCard score={null} tier="voice" factors={{}} narrative="" delta={null} trend={[]} />,
    );
    expect(screen.getByTestId("inner-score-building")).toBeInTheDocument();
    expect(screen.getByText(/voice check-in/i)).toBeInTheDocument();
  });

  it("shows score number when score is provided", () => {
    renderWithProviders(
      <InnerScoreCard score={72} tier="health_voice" factors={{ sleep_quality: 85 }} narrative="Good sleep." delta={5} trend={[65, 68, 72]} />,
    );
    expect(screen.getByText("72")).toBeInTheDocument();
  });

  it("shows Inner Score label", () => {
    renderWithProviders(
      <InnerScoreCard score={72} tier="health_voice" factors={{}} narrative="" delta={null} trend={[]} />,
    );
    expect(screen.getByText("Inner Score")).toBeInTheDocument();
  });

  it("shows Good label for score 72", () => {
    renderWithProviders(
      <InnerScoreCard score={72} tier="health_voice" factors={{}} narrative="" delta={null} trend={[]} />,
    );
    expect(screen.getByText("Good")).toBeInTheDocument();
  });

  it("shows Thriving for score 85", () => {
    renderWithProviders(
      <InnerScoreCard score={85} tier="eeg_health_voice" factors={{}} narrative="" delta={null} trend={[]} />,
    );
    expect(screen.getByText("Thriving")).toBeInTheDocument();
  });

  it("shows Low for score 30", () => {
    renderWithProviders(
      <InnerScoreCard score={30} tier="voice" factors={{}} narrative="" delta={null} trend={[]} />,
    );
    expect(screen.getByText("Low")).toBeInTheDocument();
  });

  it("shows tier confidence label for health_voice", () => {
    renderWithProviders(
      <InnerScoreCard score={72} tier="health_voice" factors={{}} narrative="" delta={null} trend={[]} />,
    );
    expect(screen.getByText(/sleep, body, and mood/i)).toBeInTheDocument();
  });

  it("shows tier confidence label for voice", () => {
    renderWithProviders(
      <InnerScoreCard score={60} tier="voice" factors={{}} narrative="" delta={null} trend={[]} />,
    );
    expect(screen.getByText(/how you sound/i)).toBeInTheDocument();
  });

  it("shows delta when provided", () => {
    renderWithProviders(
      <InnerScoreCard score={72} tier="voice" factors={{}} narrative="" delta={5} trend={[]} />,
    );
    expect(screen.getByText("+5")).toBeInTheDocument();
  });

  it("shows negative delta", () => {
    renderWithProviders(
      <InnerScoreCard score={50} tier="voice" factors={{}} narrative="" delta={-3} trend={[]} />,
    );
    expect(screen.getByText("-3")).toBeInTheDocument();
  });

  it("renders SVG gauge", () => {
    renderWithProviders(
      <InnerScoreCard score={72} tier="voice" factors={{}} narrative="" delta={null} trend={[]} />,
    );
    expect(screen.getByTestId("inner-score-gauge")).toBeInTheDocument();
  });

  it("has accessible aria-label on gauge", () => {
    renderWithProviders(
      <InnerScoreCard score={72} tier="voice" factors={{}} narrative="" delta={null} trend={[]} />,
    );
    expect(screen.getByLabelText(/inner score: 72/i)).toBeInTheDocument();
  });

  it("shows factor bars and narrative after tap", () => {
    renderWithProviders(
      <InnerScoreCard
        score={72}
        tier="health_voice"
        factors={{ sleep_quality: 85, stress_inverse: 58 }}
        narrative="Good sleep is carrying you."
        delta={5}
        trend={[]}
      />,
    );
    fireEvent.click(screen.getByTestId("inner-score-card"));
    expect(screen.getByText("Good sleep is carrying you.")).toBeInTheDocument();
  });

  it("does not show building CTA when score exists", () => {
    renderWithProviders(
      <InnerScoreCard score={72} tier="voice" factors={{}} narrative="" delta={null} trend={[]} />,
    );
    expect(screen.queryByTestId("inner-score-building")).not.toBeInTheDocument();
  });
});
