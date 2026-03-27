import React from "react";
import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { SleepHypnogram } from "@/components/sleep-hypnogram";
import type { StageEvent } from "@/components/sleep-hypnogram";

const SAMPLE_HISTORY: StageEvent[] = [
  { stage: "N1",  t: 0    },
  { stage: "N2",  t: 480  },  // 8 min
  { stage: "N3",  t: 960  },  // 16 min
  { stage: "REM", t: 1800 },  // 30 min
  { stage: "N2",  t: 2280 },  // 38 min
  { stage: "REM", t: 2760 },  // 46 min
];
const TOTAL_SEC = 3600; // 1 hour

describe("SleepHypnogram", () => {
  it("renders empty state when no history", () => {
    renderWithProviders(<SleepHypnogram stageHistory={[]} totalSeconds={0} />);
    expect(screen.getByTestId("sleep-hypnogram-empty")).toBeInTheDocument();
  });

  it("renders empty state when totalSeconds is 0", () => {
    renderWithProviders(
      <SleepHypnogram stageHistory={SAMPLE_HISTORY} totalSeconds={0} />
    );
    expect(screen.getByTestId("sleep-hypnogram-empty")).toBeInTheDocument();
  });

  it("renders hypnogram SVG when data is present", () => {
    renderWithProviders(
      <SleepHypnogram stageHistory={SAMPLE_HISTORY} totalSeconds={TOTAL_SEC} />
    );
    expect(screen.getByTestId("sleep-hypnogram")).toBeInTheDocument();
  });

  it("renders SVG with aria-label", () => {
    renderWithProviders(
      <SleepHypnogram stageHistory={SAMPLE_HISTORY} totalSeconds={TOTAL_SEC} />
    );
    expect(screen.getByLabelText("Sleep stage hypnogram")).toBeInTheDocument();
  });

  it("renders stage labels in the SVG", () => {
    renderWithProviders(
      <SleepHypnogram stageHistory={SAMPLE_HISTORY} totalSeconds={TOTAL_SEC} />
    );
    // SVG text elements should contain the stage labels
    const svg = screen.getByLabelText("Sleep stage hypnogram");
    expect(svg.textContent).toContain("Wake");
    expect(svg.textContent).toContain("N1");
    expect(svg.textContent).toContain("N2");
    expect(svg.textContent).toContain("REM");
  });

  it("renders with a single stage event without crashing", () => {
    renderWithProviders(
      <SleepHypnogram
        stageHistory={[{ stage: "N2", t: 0 }]}
        totalSeconds={600}
      />
    );
    expect(screen.getByTestId("sleep-hypnogram")).toBeInTheDocument();
  });

  it("applies custom height", () => {
    const { container } = renderWithProviders(
      <SleepHypnogram stageHistory={SAMPLE_HISTORY} totalSeconds={TOTAL_SEC} height={120} />
    );
    const svg = container.querySelector("svg");
    expect(svg).not.toBeNull();
    expect(svg!.getAttribute("height")).toBe("120");
  });
});
