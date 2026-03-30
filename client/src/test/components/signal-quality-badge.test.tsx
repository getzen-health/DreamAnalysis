import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { SignalQualityBadge } from "@/components/signal-quality-badge";

describe("SignalQualityBadge", () => {
  it("renders 'Good signal' when quality > 0.7", () => {
    render(<SignalQualityBadge quality={0.85} isStreaming={true} />);
    const badge = screen.getByTestId("signal-quality-badge");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveTextContent("Good signal");
    expect(badge.className).toContain("text-emerald-500");
  });

  it("renders 'Noisy' when quality is between 0.4 and 0.7", () => {
    render(<SignalQualityBadge quality={0.55} isStreaming={true} />);
    const badge = screen.getByTestId("signal-quality-badge");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveTextContent("Noisy");
    expect(badge.className).toContain("text-amber-500");
  });

  it("renders 'Too noisy' when quality <= 0.4", () => {
    render(<SignalQualityBadge quality={0.3} isStreaming={true} />);
    const badge = screen.getByTestId("signal-quality-badge");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveTextContent(/too noisy/i);
    expect(badge).toHaveTextContent(/relax face muscles/i);
    expect(badge.className).toContain("text-red-500");
  });

  it("renders nothing when not streaming", () => {
    const { container } = render(
      <SignalQualityBadge quality={0.85} isStreaming={false} />
    );
    expect(container.firstChild).toBeNull();
    expect(screen.queryByTestId("signal-quality-badge")).not.toBeInTheDocument();
  });

  it("treats quality at exactly 0.7 as moderate (not good)", () => {
    render(<SignalQualityBadge quality={0.7} isStreaming={true} />);
    const badge = screen.getByTestId("signal-quality-badge");
    expect(badge).toHaveTextContent("Noisy");
  });

  it("treats quality at exactly 0.4 as poor (not moderate)", () => {
    render(<SignalQualityBadge quality={0.4} isStreaming={true} />);
    const badge = screen.getByTestId("signal-quality-badge");
    expect(badge).toHaveTextContent(/too noisy/i);
  });

  it("treats quality of 0 as poor", () => {
    render(<SignalQualityBadge quality={0} isStreaming={true} />);
    const badge = screen.getByTestId("signal-quality-badge");
    expect(badge).toHaveTextContent(/too noisy/i);
  });

  it("shows green dot for good signal", () => {
    render(<SignalQualityBadge quality={0.9} isStreaming={true} />);
    const badge = screen.getByTestId("signal-quality-badge");
    const dot = badge.querySelector("span.bg-emerald-500");
    expect(dot).toBeInTheDocument();
  });

  it("shows amber dot for moderate signal", () => {
    render(<SignalQualityBadge quality={0.5} isStreaming={true} />);
    const badge = screen.getByTestId("signal-quality-badge");
    const dot = badge.querySelector("span.bg-amber-500");
    expect(dot).toBeInTheDocument();
  });

  it("shows red dot for poor signal", () => {
    render(<SignalQualityBadge quality={0.2} isStreaming={true} />);
    const badge = screen.getByTestId("signal-quality-badge");
    const dot = badge.querySelector("span.bg-red-500");
    expect(dot).toBeInTheDocument();
  });
});
