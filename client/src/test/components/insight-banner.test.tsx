// client/src/test/components/insight-banner.test.tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { InsightBanner } from "@/components/insight-banner";

const mockEvent = {
  metric: "stress" as const,
  currentValue: 0.75,
  baselineMean: 0.40,
  zScore: 2.3,
  direction: "high" as const,
  durationMinutes: 8,
  baselineQuality: 0.5,
};

describe("InsightBanner", () => {
  it("renders nothing when events array is empty", () => {
    const { container } = render(<InsightBanner events={[]} onDismiss={vi.fn()} onCTA={vi.fn()} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders deviation context for stress event", () => {
    render(<InsightBanner events={[mockEvent]} onDismiss={vi.fn()} onCTA={vi.fn()} />);
    expect(screen.getByText(/stress/i)).toBeInTheDocument();
    expect(screen.getByText(/8 min/i)).toBeInTheDocument();
  });

  it("calls onDismiss when × button clicked", () => {
    const onDismiss = vi.fn();
    render(<InsightBanner events={[mockEvent]} onDismiss={onDismiss} onCTA={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));
    expect(onDismiss).toHaveBeenCalledOnce();
  });

  it("calls onCTA with intervention deeplink when CTA clicked", () => {
    const onCTA = vi.fn();
    render(<InsightBanner events={[mockEvent]} onDismiss={vi.fn()} onCTA={onCTA} />);
    const ctaBtn = screen.getByRole("button", { name: /breathing/i });
    fireEvent.click(ctaBtn);
    expect(onCTA).toHaveBeenCalledWith(expect.stringContaining("/biofeedback"));
  });
});
