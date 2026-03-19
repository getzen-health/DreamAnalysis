/**
 * EFS Share Card — Canvas-based PNG export of the Emotional Fitness Score.
 *
 * Creates an 800x500 landscape card (good for Instagram stories) with:
 * - Large EFS score
 * - 5 horizontal bars for each vital
 * - Daily insight text
 * - NeuralDreamWorkshop watermark
 *
 * Follows the same Canvas 2D pattern used in weekly-brain-summary.tsx.
 */

import type { EFSData } from "@/lib/ml-api";

// ── Colors ──────────────────────────────────────────────────────────────────

const BG = "#0f172a";
const CARD_BG = "#1e293b";
const GREEN = "#0891b2";
const AMBER = "#d4a017";
const RED = "#e879a8";
const TEXT_PRIMARY = "#f8fafc";
const TEXT_MUTED = "#94a3b8";
const TEXT_DIM = "#64748b";
const BORDER = "#334155";

function scoreColor(color: "green" | "amber" | "red" | null): string {
  if (color === "green") return GREEN;
  if (color === "amber") return AMBER;
  if (color === "red") return RED;
  return TEXT_DIM;
}

function vitalBarColor(score: number): string {
  if (score >= 70) return GREEN;
  if (score >= 40) return AMBER;
  return RED;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  if (typeof ctx.roundRect === "function") {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
    ctx.fill();
  } else {
    // Fallback for browsers without roundRect
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
    ctx.fill();
  }
}

// ── Export function ─────────────────────────────────────────────────────────

const VITAL_LABELS: Record<string, string> = {
  resilience: "Resilience",
  regulation: "Regulation",
  awareness: "Awareness",
  range: "Range",
  stability: "Stability",
};

const VITAL_ORDER = ["resilience", "regulation", "awareness", "range", "stability"];

export function exportEFSCard(data: EFSData): void {
  const W = 800;
  const H = 500;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // ── Background ──────────────────────────────────────────────────────────
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, W, H);

  // Subtle grid pattern
  ctx.strokeStyle = "#1e293b40";
  ctx.lineWidth = 1;
  for (let x = 0; x < W; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
  }
  for (let y = 0; y < H; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  }

  // ── Title ───────────────────────────────────────────────────────────────
  ctx.fillStyle = TEXT_PRIMARY;
  ctx.font = "bold 24px system-ui, -apple-system, sans-serif";
  ctx.fillText("Emotional Fitness", 40, 50);

  // Date
  ctx.fillStyle = TEXT_DIM;
  ctx.font = "13px system-ui, sans-serif";
  const dateStr = new Date().toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });
  ctx.fillText(dateStr, 40, 72);

  // ── Accent line ─────────────────────────────────────────────────────────
  const grad = ctx.createLinearGradient(0, 88, W * 0.5, 88);
  grad.addColorStop(0, GREEN);
  grad.addColorStop(1, `${GREEN}00`);
  ctx.fillStyle = grad;
  ctx.fillRect(40, 88, W - 80, 2);

  // ── Score display (left side) ───────────────────────────────────────────
  const displayScore = data.score ?? 0;
  const color = scoreColor(data.color);

  // Score card background
  ctx.fillStyle = CARD_BG;
  roundRect(ctx, 40, 110, 200, 160, 12);

  // Score number
  ctx.fillStyle = color;
  ctx.font = "bold 64px system-ui, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(String(displayScore), 140, 195);

  // "/ 100" subtext
  ctx.fillStyle = TEXT_DIM;
  ctx.font = "14px system-ui, sans-serif";
  ctx.fillText("/ 100", 140, 218);

  // Label
  if (data.label) {
    ctx.fillStyle = color;
    ctx.font = "bold 14px system-ui, sans-serif";
    ctx.fillText(data.label, 140, 248);
  }

  ctx.textAlign = "left";

  // ── Vitals bars (right side) ────────────────────────────────────────────
  const barsX = 280;
  const barsY = 110;
  const barW = 460;
  const barH = 22;
  const barGap = 10;

  VITAL_ORDER.forEach((key, i) => {
    const vital = data.vitals[key];
    if (!vital) return;

    const y = barsY + i * (barH + barGap);
    const score = vital.score ?? 0;
    const barColor = vital.status === "available" ? vitalBarColor(score) : TEXT_DIM;

    // Label
    ctx.fillStyle = TEXT_MUTED;
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText(VITAL_LABELS[key] ?? key, barsX, y + 14);

    // Score text
    ctx.fillStyle = vital.status === "available" ? barColor : TEXT_DIM;
    ctx.font = "bold 12px system-ui, sans-serif";
    const scoreText = vital.status === "available" ? String(score) : "--";
    const scoreTw = ctx.measureText(scoreText).width;
    ctx.fillText(scoreText, barsX + barW - scoreTw, y + 14);

    // Bar background
    const barStartX = barsX + 80;
    const fillW = barW - 80 - scoreTw - 12;

    ctx.fillStyle = BORDER;
    roundRect(ctx, barStartX, y + 4, fillW, 6, 3);

    // Bar fill
    if (vital.status === "available" && score > 0) {
      ctx.fillStyle = barColor;
      roundRect(ctx, barStartX, y + 4, Math.max(4, (score / 100) * fillW), 6, 3);
    }
  });

  // ── Daily insight (below vitals) ────────────────────────────────────────
  if (data.dailyInsight) {
    const insightY = barsY + VITAL_ORDER.length * (barH + barGap) + 20;

    // Insight card background
    ctx.fillStyle = CARD_BG;
    roundRect(ctx, 40, insightY, W - 80, 80, 10);

    // Left accent bar
    ctx.fillStyle = GREEN;
    ctx.fillRect(40, insightY, 3, 80);

    // Insight text — wrap to fit
    ctx.fillStyle = TEXT_PRIMARY;
    ctx.font = "13px system-ui, sans-serif";
    const maxTextW = W - 80 - 40;
    const words = data.dailyInsight.text.split(" ");
    let line = "";
    let lineY = insightY + 24;
    const lineHeight = 18;
    let lineCount = 0;

    for (const word of words) {
      const test = line ? `${line} ${word}` : word;
      if (ctx.measureText(test).width > maxTextW && line) {
        ctx.fillText(line, 56, lineY);
        line = word;
        lineY += lineHeight;
        lineCount++;
        if (lineCount >= 2) break; // max 3 lines
      } else {
        line = test;
      }
    }
    if (line) ctx.fillText(line, 56, lineY);

    // Action nudge
    if (data.dailyInsight.actionNudge) {
      ctx.fillStyle = TEXT_DIM;
      ctx.font = "11px system-ui, sans-serif";
      const nudgeText =
        data.dailyInsight.actionNudge.length > 80
          ? data.dailyInsight.actionNudge.slice(0, 77) + "..."
          : data.dailyInsight.actionNudge;
      ctx.fillText(nudgeText, 56, insightY + 66);
    }
  }

  // ── Footer ──────────────────────────────────────────────────────────────
  ctx.fillStyle = BORDER;
  ctx.fillRect(40, H - 45, W - 80, 1);

  ctx.fillStyle = TEXT_DIM;
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText("NeuralDreamWorkshop", 40, H - 18);

  // Confidence label
  if (data.confidence === "early_estimate") {
    ctx.fillStyle = AMBER;
    ctx.font = "11px system-ui, sans-serif";
    const earlyLabel = "Early estimate";
    const ew = ctx.measureText(earlyLabel).width;
    ctx.fillText(earlyLabel, W - 40 - ew, H - 18);
  }

  // ── Download ────────────────────────────────────────────────────────────
  const link = document.createElement("a");
  link.download = "emotional-fitness.png";
  link.href = canvas.toDataURL("image/png");
  link.click();
}
