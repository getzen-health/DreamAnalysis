import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import DeviceTierBadge from "@/components/device-tier-badge";
import type { DeviceTier } from "@/lib/dream-pipeline";

describe("DeviceTierBadge", () => {
  it("renders with data-testid", () => {
    render(<DeviceTierBadge tier="eeg_full" />);
    expect(screen.getByTestId("device-tier-badge")).toBeInTheDocument();
  });

  it.each<[DeviceTier, string]>([
    ["eeg_full", "Full EEG"],
    ["eeg_basic", "Basic EEG"],
    ["phone_only", "Phone Sensors"],
    ["none", "Journal Only"],
  ])("shows correct label for tier %s", (tier, expectedLabel) => {
    render(<DeviceTierBadge tier={tier} />);
    expect(screen.getByTestId("device-tier-badge")).toHaveTextContent(expectedLabel);
  });

  it("applies green styling for eeg_full", () => {
    render(<DeviceTierBadge tier="eeg_full" />);
    const badge = screen.getByTestId("device-tier-badge");
    expect(badge.className).toContain("emerald");
  });

  it("applies amber styling for eeg_basic", () => {
    render(<DeviceTierBadge tier="eeg_basic" />);
    const badge = screen.getByTestId("device-tier-badge");
    expect(badge.className).toContain("amber");
  });

  it("applies blue styling for phone_only", () => {
    render(<DeviceTierBadge tier="phone_only" />);
    const badge = screen.getByTestId("device-tier-badge");
    expect(badge.className).toContain("blue");
  });

  it("applies muted styling for none", () => {
    render(<DeviceTierBadge tier="none" />);
    const badge = screen.getByTestId("device-tier-badge");
    expect(badge.className).toContain("muted");
  });

  it("accepts additional className", () => {
    render(<DeviceTierBadge tier="eeg_full" className="ml-2" />);
    const badge = screen.getByTestId("device-tier-badge");
    expect(badge.className).toContain("ml-2");
  });
});
