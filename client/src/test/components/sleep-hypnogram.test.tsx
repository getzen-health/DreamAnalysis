import React from "react";
import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { SleepHypnogram } from "@/components/sleep-hypnogram";
import type { StageEvent, DreamEpisode } from "@/components/sleep-hypnogram";

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

  // ── Dream Episode Overlay Tests ────────────────────────────────────────────

  it("renders without dreamEpisodes (backward compatible)", () => {
    renderWithProviders(
      <SleepHypnogram stageHistory={SAMPLE_HISTORY} totalSeconds={TOTAL_SEC} />
    );
    expect(screen.getByTestId("sleep-hypnogram")).toBeInTheDocument();
    expect(screen.queryByTestId("dream-overlay-0")).not.toBeInTheDocument();
  });

  it("renders dream overlay rectangles when episodes provided", () => {
    const episodes: DreamEpisode[] = [
      { startT: 1800, endT: 2100 },
      { startT: 2760, endT: 3000, intensity: 0.8 },
    ];
    renderWithProviders(
      <SleepHypnogram
        stageHistory={SAMPLE_HISTORY}
        totalSeconds={TOTAL_SEC}
        dreamEpisodes={episodes}
      />
    );
    expect(screen.getByTestId("dream-overlay-0")).toBeInTheDocument();
    expect(screen.getByTestId("dream-overlay-1")).toBeInTheDocument();
    expect(screen.getByTestId("dream-rect-0")).toBeInTheDocument();
    expect(screen.getByTestId("dream-rect-1")).toBeInTheDocument();
  });

  it("dream overlay positioned correctly based on time", () => {
    const episodes: DreamEpisode[] = [
      { startT: 900, endT: 1800 }, // quarter to half of the session
    ];
    const { container } = renderWithProviders(
      <SleepHypnogram
        stageHistory={SAMPLE_HISTORY}
        totalSeconds={TOTAL_SEC}
        dreamEpisodes={episodes}
      />
    );
    const rect = screen.getByTestId("dream-rect-0");
    const x = parseFloat(rect.getAttribute("x") ?? "0");
    const width = parseFloat(rect.getAttribute("width") ?? "0");
    // LABEL_W=32, CHART_W=68; startT=900/3600=0.25 => x=32+0.25*68=49
    // endT=1800/3600=0.5 => x2=32+0.5*68=66; width=66-49=17
    expect(x).toBeCloseTo(49, 0);
    expect(width).toBeCloseTo(17, 0);
    // Rect should span the full chart height
    expect(rect.getAttribute("y")).toBe("0");
  });

  it("renders no dream overlays when empty array provided", () => {
    renderWithProviders(
      <SleepHypnogram
        stageHistory={SAMPLE_HISTORY}
        totalSeconds={TOTAL_SEC}
        dreamEpisodes={[]}
      />
    );
    expect(screen.getByTestId("sleep-hypnogram")).toBeInTheDocument();
    expect(screen.queryByTestId("dream-overlay-0")).not.toBeInTheDocument();
  });
});
