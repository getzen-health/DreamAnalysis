import { describe, it, expect, vi, beforeEach } from "vitest";
import type { DreamShareData } from "@/lib/dream-share-card";

// ── Canvas mock ──────────────────────────────────────────────────────────────

function createMockContext() {
  const calls: { method: string; args: unknown[] }[] = [];

  const track = (method: string) =>
    vi.fn((...args: unknown[]) => {
      calls.push({ method, args });
    });

  const ctx: Record<string, unknown> = {
    // Drawing
    fillRect: track("fillRect"),
    fillText: track("fillText"),
    strokeRect: track("strokeRect"),
    clearRect: track("clearRect"),

    // Paths
    beginPath: track("beginPath"),
    closePath: track("closePath"),
    moveTo: track("moveTo"),
    lineTo: track("lineTo"),
    arc: track("arc"),
    quadraticCurveTo: track("quadraticCurveTo"),
    fill: track("fill"),
    stroke: track("stroke"),

    // Transforms
    scale: track("scale"),
    save: track("save"),
    restore: track("restore"),

    // Gradients
    createLinearGradient: vi.fn(() => ({
      addColorStop: vi.fn(),
    })),
    createRadialGradient: vi.fn(() => ({
      addColorStop: vi.fn(),
    })),

    // Text
    measureText: vi.fn((text: string) => ({
      width: text.length * 10,
    })),

    // Properties
    fillStyle: "",
    strokeStyle: "",
    font: "",
    textAlign: "start",
    textBaseline: "alphabetic",
    lineWidth: 1,
    lineCap: "butt",
    globalCompositeOperation: "source-over",
    letterSpacing: "0px",
  };

  return { ctx: ctx as unknown as CanvasRenderingContext2D, calls };
}

function createMockCanvas(mockCtx: CanvasRenderingContext2D) {
  const canvas = {
    width: 0,
    height: 0,
    getContext: vi.fn(() => mockCtx),
    toBlob: vi.fn((callback: (blob: Blob | null) => void, type?: string) => {
      callback(new Blob(["fake-png"], { type: type ?? "image/png" }));
    }),
  };
  return canvas as unknown as HTMLCanvasElement;
}

// ── Test data ────────────────────────────────────────────────────────────────

const SAMPLE_DATA: DreamShareData = {
  dreamSummary: "I was flying over a vast ocean of shimmering light, feeling completely free and at peace.",
  emotionalTone: "peaceful",
  sleepDuration: "7h 23m",
  remPercentage: 24,
  dreamCount: 3,
  date: "March 30, 2026",
};

// ── Tests ────────────────────────────────────────────────────────────────────

describe("dream-share-card", () => {
  let renderDreamShareCard: typeof import("@/lib/dream-share-card").renderDreamShareCard;
  let mockCtx: CanvasRenderingContext2D;
  let calls: { method: string; args: unknown[] }[];
  let canvas: HTMLCanvasElement;

  beforeEach(async () => {
    const mock = createMockContext();
    mockCtx = mock.ctx;
    calls = mock.calls;
    canvas = createMockCanvas(mockCtx);

    // Dynamic import to avoid module-level document access issues
    const mod = await import("@/lib/dream-share-card");
    renderDreamShareCard = mod.renderDreamShareCard;
  });

  it("sets canvas dimensions to 2x DPI (2160x3840)", async () => {
    await renderDreamShareCard(canvas, SAMPLE_DATA);
    expect(canvas.width).toBe(2160);
    expect(canvas.height).toBe(3840);
  });

  it("scales context to 2x DPI", async () => {
    await renderDreamShareCard(canvas, SAMPLE_DATA);
    const scaleCalls = calls.filter((c) => c.method === "scale");
    expect(scaleCalls.length).toBeGreaterThanOrEqual(1);
    expect(scaleCalls[0].args).toEqual([2, 2]);
  });

  it("renders the date text", async () => {
    await renderDreamShareCard(canvas, SAMPLE_DATA);
    const textCalls = calls.filter((c) => c.method === "fillText");
    const dateCall = textCalls.find((c) => c.args[0] === "March 30, 2026");
    expect(dateCall).toBeDefined();
  });

  it("renders the title text", async () => {
    await renderDreamShareCard(canvas, SAMPLE_DATA);
    const textCalls = calls.filter((c) => c.method === "fillText");
    const titleCall = textCalls.find((c) => c.args[0] === "LAST NIGHT'S DREAM");
    expect(titleCall).toBeDefined();
  });

  it("renders the emotional tone badge text", async () => {
    await renderDreamShareCard(canvas, SAMPLE_DATA);
    const textCalls = calls.filter((c) => c.method === "fillText");
    const toneCall = textCalls.find((c) => c.args[0] === "Peaceful");
    expect(toneCall).toBeDefined();
  });

  it("renders dream summary text (at least first line)", async () => {
    await renderDreamShareCard(canvas, SAMPLE_DATA);
    const textCalls = calls.filter((c) => c.method === "fillText");
    // The summary gets word-wrapped, so check that at least part of it appears
    const summaryCall = textCalls.find(
      (c) => typeof c.args[0] === "string" && (c.args[0] as string).includes("flying"),
    );
    expect(summaryCall).toBeDefined();
  });

  it("renders sleep duration stat", async () => {
    await renderDreamShareCard(canvas, SAMPLE_DATA);
    const textCalls = calls.filter((c) => c.method === "fillText");
    const durCall = textCalls.find((c) => c.args[0] === "7h 23m");
    expect(durCall).toBeDefined();
  });

  it("renders REM percentage stat", async () => {
    await renderDreamShareCard(canvas, SAMPLE_DATA);
    const textCalls = calls.filter((c) => c.method === "fillText");
    const remCall = textCalls.find((c) => c.args[0] === "24%");
    expect(remCall).toBeDefined();
  });

  it("renders dream count stat", async () => {
    await renderDreamShareCard(canvas, SAMPLE_DATA);
    const textCalls = calls.filter((c) => c.method === "fillText");
    const countCall = textCalls.find((c) => c.args[0] === "3");
    expect(countCall).toBeDefined();
  });

  it("renders 'Dreams' label (plural) for count > 1", async () => {
    await renderDreamShareCard(canvas, SAMPLE_DATA);
    const textCalls = calls.filter((c) => c.method === "fillText");
    const label = textCalls.find((c) => c.args[0] === "Dreams");
    expect(label).toBeDefined();
  });

  it("renders 'Dream' label (singular) for count === 1", async () => {
    const singleDream = { ...SAMPLE_DATA, dreamCount: 1 };
    await renderDreamShareCard(canvas, singleDream);
    const textCalls = calls.filter((c) => c.method === "fillText");
    const label = textCalls.find((c) => c.args[0] === "Dream");
    expect(label).toBeDefined();
  });

  it("renders app branding", async () => {
    await renderDreamShareCard(canvas, SAMPLE_DATA);
    const textCalls = calls.filter((c) => c.method === "fillText");
    const brandCall = textCalls.find((c) =>
      typeof c.args[0] === "string" && (c.args[0] as string).includes("ANTARAI"),
    );
    expect(brandCall).toBeDefined();
  });

  it("draws stars (arc calls for decorative dots)", async () => {
    await renderDreamShareCard(canvas, SAMPLE_DATA);
    const arcCalls = calls.filter((c) => c.method === "arc");
    // Moon glow + moon body + moon cut-out = 3 arcs, plus 40 star arcs = 43+
    expect(arcCalls.length).toBeGreaterThanOrEqual(40);
  });

  it("draws the moon (uses save/restore for compositing)", async () => {
    await renderDreamShareCard(canvas, SAMPLE_DATA);
    const saveCalls = calls.filter((c) => c.method === "save");
    const restoreCalls = calls.filter((c) => c.method === "restore");
    expect(saveCalls.length).toBeGreaterThanOrEqual(1);
    expect(restoreCalls.length).toBeGreaterThanOrEqual(1);
  });

  it("draws gradient background (fillRect covering canvas)", async () => {
    await renderDreamShareCard(canvas, SAMPLE_DATA);
    const fillRects = calls.filter((c) => c.method === "fillRect");
    // First fillRect should be the full background (0, 0, 1080, 1920)
    const bgRect = fillRects.find(
      (c) => c.args[0] === 0 && c.args[1] === 0 && c.args[2] === 1080 && c.args[3] === 1920,
    );
    expect(bgRect).toBeDefined();
  });

  it("returns a Blob of type image/png", async () => {
    const result = await renderDreamShareCard(canvas, SAMPLE_DATA);
    expect(result).toBeInstanceOf(Blob);
    expect(result.type).toBe("image/png");
  });

  it("throws if canvas context is null", async () => {
    const badCanvas = {
      width: 0,
      height: 0,
      getContext: vi.fn(() => null),
      toBlob: vi.fn(),
    } as unknown as HTMLCanvasElement;

    await expect(renderDreamShareCard(badCanvas, SAMPLE_DATA)).rejects.toThrow(
      "Cannot create 2D canvas context",
    );
  });

  it("handles unknown emotional tone gracefully (falls back to neutral colors)", async () => {
    const unknownTone = { ...SAMPLE_DATA, emotionalTone: "whimsical" };
    // Should not throw
    await renderDreamShareCard(canvas, unknownTone);
    const textCalls = calls.filter((c) => c.method === "fillText");
    const toneCall = textCalls.find((c) => c.args[0] === "Whimsical");
    expect(toneCall).toBeDefined();
  });

  it("handles very long dream summary without crashing", async () => {
    const longSummary = {
      ...SAMPLE_DATA,
      dreamSummary: "I was walking through an endless forest of crystalline trees that " +
        "reflected every color of the rainbow, and each step I took created ripples in " +
        "the ground like water, spreading out in concentric circles of light that " +
        "illuminated hidden pathways leading to ancient temples made of pure starlight " +
        "and humming with an otherworldly resonance that filled my entire being with " +
        "a profound sense of connection to the universe itself.",
    };
    // Should not throw, should truncate to ~6 lines
    await renderDreamShareCard(canvas, longSummary);
    const textCalls = calls.filter((c) => c.method === "fillText");
    expect(textCalls.length).toBeGreaterThan(0);
  });

  it("creates linear gradients for background and dividers", async () => {
    await renderDreamShareCard(canvas, SAMPLE_DATA);
    expect(mockCtx.createLinearGradient).toHaveBeenCalled();
  });

  it("creates radial gradient for moon glow", async () => {
    await renderDreamShareCard(canvas, SAMPLE_DATA);
    expect(mockCtx.createRadialGradient).toHaveBeenCalled();
  });
});
