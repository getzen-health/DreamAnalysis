// client/src/test/components/emotion-picker.test.tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { EmotionPicker } from "@/components/emotion-picker";

describe("EmotionPicker", () => {
  it("renders 4 quadrant tabs", () => {
    render(<EmotionPicker valence={0.6} arousal={0.7} onSelect={vi.fn()} />);
    expect(screen.getByRole("tab", { name: /high energy positive/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /high energy negative/i })).toBeInTheDocument();
  });

  it("auto-selects the quadrant matching valence+arousal", () => {
    render(<EmotionPicker valence={0.3} arousal={0.8} onSelect={vi.fn()} />);
    // ha_neg quadrant: high arousal, negative valence
    expect(screen.getByText("anxious")).toBeInTheDocument();
    expect(screen.getByText("scattered")).toBeInTheDocument();
  });

  it("calls onSelect with label when emotion chip clicked", () => {
    const onSelect = vi.fn();
    render(<EmotionPicker valence={0.6} arousal={0.7} onSelect={onSelect} />);
    fireEvent.click(screen.getByRole("button", { name: "excited" }));
    expect(onSelect).toHaveBeenCalledWith("excited");
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
