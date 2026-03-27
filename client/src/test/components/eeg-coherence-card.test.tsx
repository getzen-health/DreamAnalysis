import React from "react";
import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { EEGCoherenceCard } from "@/components/eeg-coherence-card";

describe("EEGCoherenceCard", () => {
  it("renders without crashing", () => {
    renderWithProviders(<EEGCoherenceCard />);
    expect(screen.getByTestId("eeg-coherence-card")).toBeInTheDocument();
  });

  it("shows empty state when no PLV data", () => {
    renderWithProviders(<EEGCoherenceCard />);
    expect(screen.getByTestId("coherence-empty")).toBeInTheDocument();
  });

  it("shows streaming message when isStreaming=true and no data", () => {
    renderWithProviders(<EEGCoherenceCard isStreaming={true} />);
    expect(screen.getByText(/computing connectivity/i)).toBeInTheDocument();
  });

  it("shows connect headset message when not streaming and no data", () => {
    renderWithProviders(<EEGCoherenceCard isStreaming={false} />);
    expect(screen.getByText(/connect eeg headset/i)).toBeInTheDocument();
  });

  it("renders the SVG diagram", () => {
    renderWithProviders(<EEGCoherenceCard frontalPlv={0.5} />);
    expect(screen.getByTestId("coherence-svg")).toBeInTheDocument();
  });

  it("shows heading", () => {
    renderWithProviders(<EEGCoherenceCard frontalPlv={0.5} />);
    expect(screen.getByText("Brain Connectivity")).toBeInTheDocument();
  });

  it("shows PLV badge", () => {
    renderWithProviders(<EEGCoherenceCard frontalPlv={0.5} />);
    expect(screen.getByText("PLV")).toBeInTheDocument();
  });

  it("shows Frontal row when frontalPlv is provided", () => {
    renderWithProviders(<EEGCoherenceCard frontalPlv={0.5} />);
    expect(screen.getByText(/frontal/i)).toBeInTheDocument();
  });

  it("shows Temporal row when temporalPlv is provided", () => {
    renderWithProviders(<EEGCoherenceCard temporalPlv={0.4} />);
    expect(screen.getByText(/temporal/i)).toBeInTheDocument();
  });

  it("shows L. Fronto-temporal row when leftFrontotemporalPlv is provided", () => {
    renderWithProviders(<EEGCoherenceCard leftFrontotemporalPlv={0.6} />);
    expect(screen.getByText(/L\. Fronto-temporal/i)).toBeInTheDocument();
  });

  it("shows R. Fronto-temporal row when rightFrontotemporalPlv is provided", () => {
    renderWithProviders(<EEGCoherenceCard rightFrontotemporalPlv={0.3} />);
    expect(screen.getByText(/R\. Fronto-temporal/i)).toBeInTheDocument();
  });

  it("does not show empty state when any PLV data is present", () => {
    renderWithProviders(<EEGCoherenceCard temporalPlv={0.5} />);
    expect(screen.queryByTestId("coherence-empty")).not.toBeInTheDocument();
  });

  it("shows Moderate label for mid-range PLV", () => {
    renderWithProviders(<EEGCoherenceCard frontalPlv={0.5} />);
    expect(screen.getByText("Moderate")).toBeInTheDocument();
  });

  it("shows Strong label for high PLV", () => {
    renderWithProviders(<EEGCoherenceCard frontalPlv={0.75} />);
    expect(screen.getByText("Strong")).toBeInTheDocument();
  });

  it("shows Weak label for low PLV", () => {
    renderWithProviders(<EEGCoherenceCard frontalPlv={0.2} />);
    expect(screen.getByText("Weak")).toBeInTheDocument();
  });

  it("renders all four electrode labels in the SVG", () => {
    renderWithProviders(<EEGCoherenceCard frontalPlv={0.5} />);
    expect(screen.getByText("TP9")).toBeInTheDocument();
    expect(screen.getByText("AF7")).toBeInTheDocument();
    expect(screen.getByText("AF8")).toBeInTheDocument();
    expect(screen.getByText("TP10")).toBeInTheDocument();
  });

  it("renders all four electrodes even when no data", () => {
    renderWithProviders(<EEGCoherenceCard />);
    expect(screen.getByText("TP9")).toBeInTheDocument();
    expect(screen.getByText("TP10")).toBeInTheDocument();
  });
});
