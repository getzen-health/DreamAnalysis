import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { PredictiveAlertCard } from "@/components/predictive-alert-card";
import type { PredictiveAlert } from "@/lib/predictive-alerts";

const warningAlert: PredictiveAlert = {
  id: "sleep-debt-warning",
  type: "warning",
  headline: "Sleep debt building — tomorrow may be tough",
  body: "You've averaged 5.8 hours this week.",
  confidence: 72,
  factors: ["Today: less than 6 hours sleep", "7-day avg: 5.8h"],
  action: "Try to get 8+ hours tonight",
};

const positiveAlert: PredictiveAlert = {
  id: "great-day-ahead",
  type: "positive",
  headline: "Tomorrow looks strong — you're well-rested and relaxed",
  body: "Good sleep quality combined with low stress.",
  confidence: 68,
  factors: ["Sleep quality: 85%", "Stress: 15%", "Inner Score: 82"],
};

describe("PredictiveAlertCard", () => {
  // 1. Renders nothing when alert is null
  it("renders nothing when alert is null", () => {
    const { container } = renderWithProviders(<PredictiveAlertCard alert={null} />);
    expect(container.innerHTML).toBe("");
    expect(screen.queryByTestId("predictive-alert-card")).not.toBeInTheDocument();
  });

  // 2. Renders warning style with amber accent
  it("renders warning style with amber border for warning alerts", () => {
    renderWithProviders(<PredictiveAlertCard alert={warningAlert} />);
    const card = screen.getByTestId("predictive-alert-card");
    expect(card).toBeInTheDocument();
    expect(card.className).toContain("border-l-amber-500");
  });

  // 3. Renders positive style with emerald accent
  it("renders positive style with emerald border for positive alerts", () => {
    renderWithProviders(<PredictiveAlertCard alert={positiveAlert} />);
    const card = screen.getByTestId("predictive-alert-card");
    expect(card).toBeInTheDocument();
    expect(card.className).toContain("border-l-emerald-500");
  });

  // 4. Shows headline
  it("displays the headline text", () => {
    renderWithProviders(<PredictiveAlertCard alert={warningAlert} />);
    const headline = screen.getByTestId("predictive-headline");
    expect(headline.textContent).toBe(warningAlert.headline);
  });

  // 5. Shows body
  it("displays the body text", () => {
    renderWithProviders(<PredictiveAlertCard alert={warningAlert} />);
    const body = screen.getByTestId("predictive-body");
    expect(body.textContent).toBe(warningAlert.body);
  });

  // 6. Shows action pill when present
  it("displays action pill when action is present", () => {
    renderWithProviders(<PredictiveAlertCard alert={warningAlert} />);
    const action = screen.getByTestId("predictive-action");
    expect(action.textContent).toBe(warningAlert.action);
  });

  // 7. Does not show action when absent
  it("does not show action pill when action is undefined", () => {
    renderWithProviders(<PredictiveAlertCard alert={positiveAlert} />);
    expect(screen.queryByTestId("predictive-action")).not.toBeInTheDocument();
  });

  // 8. Shows factors
  it("displays factors as tags", () => {
    renderWithProviders(<PredictiveAlertCard alert={warningAlert} />);
    const factorsContainer = screen.getByTestId("predictive-factors");
    expect(factorsContainer).toBeInTheDocument();
    expect(factorsContainer.children.length).toBe(warningAlert.factors.length);
  });

  // 9. Shows "Tomorrow's Forecast" header
  it("displays 'Tomorrow\\'s Forecast' header", () => {
    renderWithProviders(<PredictiveAlertCard alert={warningAlert} />);
    expect(screen.getByText("Tomorrow's Forecast")).toBeInTheDocument();
  });
});
