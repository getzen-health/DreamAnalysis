import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { InterventionSuggestion } from "@/components/intervention-suggestion";
import { getSuggestion } from "@/components/intervention-suggestion";

describe("InterventionSuggestion", () => {
  // 1. Stressed -> breathing exercise
  it("suggests breathing exercise when stress is high", () => {
    renderWithProviders(
      <InterventionSuggestion emotion="neutral" stressIndex={0.7} />,
    );
    const card = screen.getByTestId("intervention-suggestion");
    expect(card).toBeInTheDocument();
    expect(screen.getByText(/breathing exercise/i)).toBeInTheDocument();
    const action = screen.getByTestId("intervention-action");
    expect(action.textContent).toMatch(/start breathing/i);
  });

  // 2. Sad -> AI companion
  it("suggests AI companion when emotion is sad", () => {
    renderWithProviders(
      <InterventionSuggestion emotion="sad" />,
    );
    expect(screen.getByText(/AI companion/i)).toBeInTheDocument();
  });

  // 3. Angry -> cognitive reappraisal (no navigation button)
  it("suggests cognitive reappraisal for angry emotion", () => {
    renderWithProviders(
      <InterventionSuggestion emotion="angry" />,
    );
    expect(screen.getByText(/cognitive reappraisal/i)).toBeInTheDocument();
    // Should not have a navigation action button
    expect(screen.queryByTestId("intervention-action")).not.toBeInTheDocument();
  });

  // 4. Happy -> dream journal
  it("suggests dream journal when emotion is happy", () => {
    renderWithProviders(
      <InterventionSuggestion emotion="happy" />,
    );
    expect(screen.getByText(/dream journal/i)).toBeInTheDocument();
  });

  // 5. Neutral -> focus session
  it("suggests focus session for neutral emotion", () => {
    renderWithProviders(
      <InterventionSuggestion emotion="neutral" />,
    );
    expect(screen.getByText(/focus session/i)).toBeInTheDocument();
  });

  // 6. Compact mode renders smaller card
  it("renders compact variant", () => {
    renderWithProviders(
      <InterventionSuggestion emotion="neutral" compact />,
    );
    const card = screen.getByTestId("intervention-suggestion");
    expect(card).toBeInTheDocument();
    // Compact card has p-3 instead of p-4
    expect(card.className).toMatch(/p-3/);
  });

  // 7. Fear triggers breathing exercise (same as stress)
  it("suggests breathing for fear emotion", () => {
    renderWithProviders(
      <InterventionSuggestion emotion="fear" />,
    );
    expect(screen.getByText(/breathing exercise/i)).toBeInTheDocument();
  });

  // 8. Low valence triggers AI companion
  it("suggests AI companion when valence is very negative", () => {
    renderWithProviders(
      <InterventionSuggestion emotion="neutral" valence={-0.5} />,
    );
    expect(screen.getByText(/AI companion/i)).toBeInTheDocument();
  });

  // 9. High positive valence triggers dream journal
  it("suggests dream journal when valence is very positive", () => {
    renderWithProviders(
      <InterventionSuggestion emotion="neutral" valence={0.6} />,
    );
    expect(screen.getByText(/dream journal/i)).toBeInTheDocument();
  });
});

describe("getSuggestion (unit)", () => {
  it("returns breathing for anxious emotion", () => {
    const s = getSuggestion("anxious", 0.3, 0);
    expect(s.route).toBe("/biofeedback");
  });

  it("returns companion for sad emotion", () => {
    const s = getSuggestion("sad", 0, 0);
    expect(s.route).toBe("/ai-companion");
  });

  it("returns null route for angry (inline exercise)", () => {
    const s = getSuggestion("angry", 0, 0);
    expect(s.route).toBeNull();
  });

  it("returns dream journal for happy emotion", () => {
    const s = getSuggestion("happy", 0, 0.5);
    expect(s.route).toBe("/dreams");
  });

  it("returns neurofeedback for balanced/neutral emotion", () => {
    const s = getSuggestion("neutral", 0.3, 0);
    expect(s.route).toBe("/neurofeedback");
  });

  it("stress takes priority over emotion label", () => {
    // High stress should override happy emotion
    const s = getSuggestion("happy", 0.8, 0.5);
    expect(s.route).toBe("/biofeedback");
  });
});
