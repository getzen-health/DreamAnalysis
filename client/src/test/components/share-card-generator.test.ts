import { describe, it, expect, beforeAll, afterAll } from "vitest";
import {
  generateShareCard,
  type ShareTemplate,
  type ShareFormat,
  type ShareData,
} from "@/components/share-card-generator";

// jsdom doesn't implement Canvas 2D. We need to patch HTMLCanvasElement
// prototype to return a mock context and mock toDataURL.

function createMockCanvasContext(): Partial<CanvasRenderingContext2D> {
  return {
    fillRect: () => {},
    fillText: () => {},
    strokeText: () => {},
    measureText: () => ({ width: 100 } as TextMetrics),
    beginPath: () => {},
    closePath: () => {},
    moveTo: () => {},
    lineTo: () => {},
    arc: () => {},
    quadraticCurveTo: () => {},
    fill: () => {},
    stroke: () => {},
    save: () => {},
    restore: () => {},
    scale: () => {},
    createLinearGradient: () => ({
      addColorStop: () => {},
    } as any),
    createRadialGradient: () => ({
      addColorStop: () => {},
    } as any),
    roundRect: () => {},
    set fillStyle(_v: any) {},
    set strokeStyle(_v: any) {},
    set lineWidth(_v: any) {},
    set lineCap(_v: any) {},
    set font(_v: any) {},
    set textAlign(_v: any) {},
    set textBaseline(_v: any) {},
    set letterSpacing(_v: any) {},
    set shadowColor(_v: any) {},
    set shadowBlur(_v: any) {},
  };
}

// Patch prototype once
const origGetContext = HTMLCanvasElement.prototype.getContext;
const origToDataURL = HTMLCanvasElement.prototype.toDataURL;

beforeAll(() => {
  HTMLCanvasElement.prototype.getContext = function (contextId: string) {
    if (contextId === "2d") return createMockCanvasContext() as any;
    return origGetContext.call(this, contextId as any) as any;
  } as any;

  HTMLCanvasElement.prototype.toDataURL = function () {
    return "data:image/png;base64,mockPngData";
  };
});

afterAll(() => {
  HTMLCanvasElement.prototype.getContext = origGetContext;
  HTMLCanvasElement.prototype.toDataURL = origToDataURL;
});

const BASE_DATA: ShareData = {
  recoveryScore: 78,
  sleepScore: 65,
  strainScore: 42,
  stressScore: 30,
  nutritionScore: 55,
  energyBank: 72,
};

const FULL_DATA: ShareData = {
  ...BASE_DATA,
  workoutName: "Morning Run",
  exercises: [{ name: "Push-ups", sets: 3, reps: 10 }],
  durationMin: 45,
  caloriesBurned: 320,
  weeklyScores: [
    { day: "Mon", recovery: 70, sleep: 65, strain: 40, stress: 30 },
    { day: "Tue", recovery: 75, sleep: 70, strain: 50, stress: 25 },
    { day: "Wed", recovery: 80, sleep: 72, strain: 35, stress: 20 },
  ],
  dateRange: "Mar 20 - Mar 26",
  bestMetric: "Recovery +8%",
  emotions: [
    { label: "Happy", value: 0.4 },
    { label: "Neutral", value: 0.3 },
    { label: "Sad", value: 0.1 },
  ],
  focusHours: 3.5,
  brainAge: 28,
  streakDays: 14,
  completionRate: 87,
  last30Days: Array(30)
    .fill(true)
    .map((_, i) => i % 5 !== 0),
};

describe("generateShareCard", () => {
  const templates: ShareTemplate[] = [
    "daily-overview",
    "workout-summary",
    "weekly-summary",
    "brain-report",
    "habit-streak",
  ];
  const formats: ShareFormat[] = ["stories", "square"];

  for (const template of templates) {
    for (const format of formats) {
      it(`generates ${template} in ${format} format without throwing`, async () => {
        const result = await generateShareCard(template, FULL_DATA, format);
        expect(result).toBeDefined();
        expect(result).toContain("data:image/png");
      });
    }
  }

  it("handles null/missing score data gracefully", async () => {
    const sparseData: ShareData = {
      recoveryScore: null,
      sleepScore: null,
      strainScore: null,
      stressScore: null,
    };

    const result = await generateShareCard("daily-overview", sparseData, "stories");
    expect(result).toContain("data:image/png");
  });

  it("handles empty exercise list for workout template", async () => {
    const data: ShareData = {
      workoutName: "Test",
      exercises: [],
    };

    const result = await generateShareCard("workout-summary", data, "square");
    expect(result).toContain("data:image/png");
  });

  it("handles empty weekly scores", async () => {
    const data: ShareData = {
      weeklyScores: [],
    };

    const result = await generateShareCard("weekly-summary", data, "stories");
    expect(result).toContain("data:image/png");
  });

  it("handles zero streak days", async () => {
    const data: ShareData = {
      streakDays: 0,
      completionRate: 0,
      last30Days: [],
    };

    const result = await generateShareCard("habit-streak", data, "stories");
    expect(result).toContain("data:image/png");
  });

  it("defaults to stories format", async () => {
    const result = await generateShareCard("daily-overview", BASE_DATA);
    expect(result).toContain("data:image/png");
  });
});
