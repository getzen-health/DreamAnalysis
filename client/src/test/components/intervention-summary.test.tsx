import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { InterventionSummary } from "@/components/intervention-summary";
import type { InterventionMetrics } from "@/components/intervention-summary";

// ── Clipboard mock ────────────────────────────────────────────────────────────

const writeTextMock = vi.fn().mockResolvedValue(undefined);

beforeEach(() => {
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText: writeTextMock },
  });
});

afterEach(() => {
  vi.clearAllMocks();
});

// ── Fixtures ──────────────────────────────────────────────────────────────────

const improved: { before: InterventionMetrics; after: InterventionMetrics } = {
  before: { stress: 0.70, focus: 0.40, hrv: 30 },
  after:  { stress: 0.44, focus: 0.65, hrv: 45 },
};

const worsened: { before: InterventionMetrics; after: InterventionMetrics } = {
  before: { stress: 0.30, focus: 0.80, hrv: 60 },
  after:  { stress: 0.55, focus: 0.50, hrv: 40 },
};

const noChange: { before: InterventionMetrics; after: InterventionMetrics } = {
  before: { stress: 0.50, focus: 0.50, hrv: 50 },
  after:  { stress: 0.50, focus: 0.50, hrv: 50 },
};

const noHrv: { before: InterventionMetrics; after: InterventionMetrics } = {
  before: { stress: 0.60, focus: 0.40, hrv: null },
  after:  { stress: 0.40, focus: 0.60, hrv: null },
};

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("InterventionSummary", () => {
  it("renders before and after metric values", () => {
    renderWithProviders(
      <InterventionSummary
        beforeMetrics={improved.before}
        afterMetrics={improved.after}
        duration={300}
        type="Coherence breathing"
      />
    );

    // Stress before: 0.70, after: 0.44
    expect(screen.getAllByText(/0\.70/)[0]).toBeInTheDocument();
    expect(screen.getAllByText(/0\.44/)[0]).toBeInTheDocument();

    // Focus before: 0.40, after: 0.65
    expect(screen.getAllByText(/0\.40/)[0]).toBeInTheDocument();
    expect(screen.getAllByText(/0\.65/)[0]).toBeInTheDocument();
  });

  it("shows the session type and formatted duration", () => {
    renderWithProviders(
      <InterventionSummary
        beforeMetrics={improved.before}
        afterMetrics={improved.after}
        duration={300}
        type="Box breathing"
      />
    );
    expect(screen.getByText(/Box breathing/)).toBeInTheDocument();
    expect(screen.getByText(/5 min/)).toBeInTheDocument();
  });

  it("shows improvement percentages for stress (lower is better)", () => {
    renderWithProviders(
      <InterventionSummary
        beforeMetrics={improved.before}
        afterMetrics={improved.after}
        duration={120}
        type="4-7-8 breathing"
      />
    );
    // stress went from 0.70 → 0.44: -37%
    expect(screen.getByText(/-37%/)).toBeInTheDocument();
  });

  it("shows improvement percentages for focus (higher is better)", () => {
    renderWithProviders(
      <InterventionSummary
        beforeMetrics={improved.before}
        afterMetrics={improved.after}
        duration={120}
        type="4-7-8 breathing"
      />
    );
    // focus went from 0.40 → 0.65: +63%
    expect(screen.getByText(/\+63%/)).toBeInTheDocument();
  });

  it("shows green (improved) icon for stress decrease", () => {
    renderWithProviders(
      <InterventionSummary
        beforeMetrics={improved.before}
        afterMetrics={improved.after}
        duration={180}
        type="Coherence breathing"
      />
    );
    // TrendingDown icon for stress (lowerIsBetter, pct < 0 = improved)
    expect(
      document.querySelector("[data-testid='icon-improved-stress']")
    ).toBeInTheDocument();
  });

  it("shows worsened icon for stress increase", () => {
    renderWithProviders(
      <InterventionSummary
        beforeMetrics={worsened.before}
        afterMetrics={worsened.after}
        duration={60}
        type="Power breathing"
      />
    );
    expect(
      document.querySelector("[data-testid='icon-worsened-stress']")
    ).toBeInTheDocument();
  });

  it("shows worsened icon for focus decrease", () => {
    renderWithProviders(
      <InterventionSummary
        beforeMetrics={worsened.before}
        afterMetrics={worsened.after}
        duration={60}
        type="Power breathing"
      />
    );
    expect(
      document.querySelector("[data-testid='icon-worsened-focus']")
    ).toBeInTheDocument();
  });

  it("shows no-change icon when metric is unchanged", () => {
    renderWithProviders(
      <InterventionSummary
        beforeMetrics={noChange.before}
        afterMetrics={noChange.after}
        duration={90}
        type="Coherence breathing"
      />
    );
    expect(
      document.querySelector("[data-testid='icon-unchanged-stress']")
    ).toBeInTheDocument();
  });

  it("shows 'no change' text label when metric is unchanged", () => {
    renderWithProviders(
      <InterventionSummary
        beforeMetrics={noChange.before}
        afterMetrics={noChange.after}
        duration={90}
        type="Coherence breathing"
      />
    );
    const noCahngeEls = screen.getAllByText(/no change/i);
    expect(noCahngeEls.length).toBeGreaterThanOrEqual(1);
  });

  it("shows HRV values when provided", () => {
    renderWithProviders(
      <InterventionSummary
        beforeMetrics={improved.before}
        afterMetrics={improved.after}
        duration={300}
        type="Coherence breathing"
      />
    );
    expect(screen.getByText(/30\.0 ms/)).toBeInTheDocument();
    expect(screen.getByText(/45\.0 ms/)).toBeInTheDocument();
  });

  it("renders dashes for HRV row when hrv is null", () => {
    renderWithProviders(
      <InterventionSummary
        beforeMetrics={noHrv.before}
        afterMetrics={noHrv.after}
        duration={120}
        type="Coherence breathing"
      />
    );
    // HRV row still renders — shows "—" placeholder
    const hrvRow = document.querySelector("[data-testid='metric-row-hrv']");
    expect(hrvRow).toBeInTheDocument();
    expect(hrvRow?.textContent).toContain("—");
  });

  it("calls navigator.clipboard.writeText when share button is clicked", async () => {
    renderWithProviders(
      <InterventionSummary
        beforeMetrics={improved.before}
        afterMetrics={improved.after}
        duration={300}
        type="Coherence breathing"
      />
    );

    const shareBtn = screen.getByTestId("share-button");
    fireEvent.click(shareBtn);

    expect(writeTextMock).toHaveBeenCalledOnce();
    const clipboardText: string = writeTextMock.mock.calls[0][0];
    expect(clipboardText).toContain("Coherence breathing");
    expect(clipboardText).toContain("Stress:");
    expect(clipboardText).toContain("Focus:");
  });

  it("clipboard text includes HRV line when hrv is available", async () => {
    renderWithProviders(
      <InterventionSummary
        beforeMetrics={improved.before}
        afterMetrics={improved.after}
        duration={300}
        type="Coherence breathing"
      />
    );

    const shareBtn = screen.getByTestId("share-button");
    fireEvent.click(shareBtn);

    const clipboardText: string = writeTextMock.mock.calls[0][0];
    expect(clipboardText).toContain("HRV:");
  });

  it("clipboard text omits HRV line when hrv is null", async () => {
    renderWithProviders(
      <InterventionSummary
        beforeMetrics={noHrv.before}
        afterMetrics={noHrv.after}
        duration={300}
        type="Coherence breathing"
      />
    );

    const shareBtn = screen.getByTestId("share-button");
    fireEvent.click(shareBtn);

    const clipboardText: string = writeTextMock.mock.calls[0][0];
    expect(clipboardText).not.toContain("HRV:");
  });
});
