// client/src/test/components/emotion-picker.test.tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/react";
import { EmotionPicker } from "@/components/emotion-picker";

describe("EmotionPicker", () => {
  it("renders 4 quadrant tabs", () => {
    render(<EmotionPicker valence={0.6} arousal={0.7} onSelect={vi.fn()} />);
    expect(screen.getByRole("tab", { name: /high energy positive/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /high energy negative/i })).toBeInTheDocument();
  });

  it("auto-selects the quadrant matching valence+arousal", () => {
    render(<EmotionPicker valence={0.3} arousal={0.8} onSelect={vi.fn()} />);
    // ha_neg: high arousal, negative valence
    expect(screen.getByRole("tab", { name: /high energy negative/i }))
      .toHaveAttribute("aria-selected", "true");
  });

  it("calls onSelect with label when emotion chip clicked", () => {
    const onSelect = vi.fn();
    render(<EmotionPicker valence={0.6} arousal={0.7} onSelect={onSelect} />);
    fireEvent.click(screen.getByRole("button", { name: "excited" }));
    expect(onSelect).toHaveBeenCalledWith("excited");
  });

  it("renders personal fingerprints in the correct quadrant tab", () => {
    render(
      <EmotionPicker
        valence={0.3}
        arousal={0.8}
        onSelect={vi.fn()}
        personalFingerprints={[{ label: "my-custom-word", quadrant: "ha_neg" }]}
      />
    );
    const yourWordsSection = screen.getByText("Your words").closest("div")!;
    expect(yourWordsSection).toBeInTheDocument();
    expect(within(yourWordsSection).getByRole("button", { name: "my-custom-word" })).toBeInTheDocument();
  });

  it("toggles chip selection: second click deselects", () => {
    const onSelect = vi.fn();
    render(<EmotionPicker valence={0.6} arousal={0.7} onSelect={onSelect} />);
    const chip = screen.getByRole("button", { name: "excited" });
    fireEvent.click(chip); // select
    expect(chip).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(chip); // deselect
    expect(chip).toHaveAttribute("aria-pressed", "false");
  });

  it("renders custom label input and calls onSelect with custom label", () => {
    const onSelect = vi.fn();
    render(<EmotionPicker valence={0.5} arousal={0.5} onSelect={onSelect} />);
    const input = screen.getByPlaceholderText(/type your own/i);
    fireEvent.change(input, { target: { value: "wired but tired" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onSelect).toHaveBeenCalledWith("wired but tired");
  });
});
