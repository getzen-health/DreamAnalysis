import { describe, it, expect, vi, beforeAll } from "vitest";
import { render, screen, act } from "@testing-library/react";
import { AnimatedNumber } from "@/components/animated-number";

beforeAll(() => {
  // Mock requestAnimationFrame for deterministic tests
  vi.spyOn(window, "requestAnimationFrame").mockImplementation((cb) => {
    cb(performance.now() + 500); // simulate enough time for animation to complete
    return 0;
  });
});

describe("AnimatedNumber", () => {
  it("renders the initial value", () => {
    render(<AnimatedNumber value={42} />);
    expect(screen.getByText("42")).toBeInTheDocument();
  });

  it("renders with a suffix", () => {
    render(<AnimatedNumber value={75} suffix="%" />);
    expect(screen.getByText("75%")).toBeInTheDocument();
  });

  it("applies custom className", () => {
    render(<AnimatedNumber value={10} className="text-primary" />);
    const el = screen.getByText("10");
    expect(el.className).toContain("text-primary");
  });

  it("renders as a span element", () => {
    const { container } = render(<AnimatedNumber value={50} />);
    expect(container.querySelector("span")).toBeInTheDocument();
  });
});
