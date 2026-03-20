/**
 * EFS Share Card — Canvas-based PNG export of the Emotional Fitness Score.
 *
 * Creates a 1080x1920 portrait card (Instagram Stories 9:16) with:
 * - Rich gradient background with radial glow
 * - Massive EFS score with glow effect
 * - 5 premium gradient-filled vital bars
 * - Daily insight text
 * - Aspirational tagline + NeuralDreamWorkshop branding
 *
 * Follows the same Canvas 2D pattern used in weekly-brain-summary.tsx.
 */

import type { EFSData } from "@/lib/ml-api";

// ── Colors ──────────────────────────────────────────────────────────────────

const GREEN = "#0891b2";
const GREEN_BRIGHT = "#22d3ee";
const AMBER = "#fbbf24";
const AMBER_BRIGHT = "#fde68a";
const RED = "#f472b6";
const RED_BRIGHT = "#fda4af";
const TEXT_PRIMARY = "#f8fafc";
const TEXT_MUTED = "#94a3b8";
const TEXT_DIM = "#64748b";

function scoreColor(color: "green" | "amber" | "red" | null): string {
  if (color === "green") return GREEN_BRIGHT;
  if (color === "amber") return AMBER_BRIGHT;
  if (color === "red") return RED_BRIGHT;
  return TEXT_DIM;
}

function scoreGlowColor(color: "green" | "amber" | "red" | null): string {
  if (color === "green") return GREEN;
  if (color === "amber") return AMBER;
  if (color === "red") return RED;
  return TEXT_DIM;
}

function vitalBarColors(score: number): [string, string] {
  if (score >= 70) return [GREEN, GREEN_BRIGHT];
  if (score >= 40) return [AMBER, AMBER_BRIGHT];
  return [RED, RED_BRIGHT];
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
  ctx.beginPath();
  if (typeof ctx.roundRect === "function") {
    ctx.roundRect(x, y, w, h, r);
  } else {
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
  }
}

function fillRoundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  roundRect(ctx, x, y, w, h, r);
  ctx.fill();
}

function strokeRoundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  roundRect(ctx, x, y, w, h, r);
  ctx.stroke();
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
  const W = 1080;
  const H = 1920;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const PAD = 80;

  // ── Background gradient ─────────────────────────────────────────────────
  // Diagonal gradient: deep purple → dark navy → dark teal
  const bgGrad = ctx.createLinearGradient(0, 0, W, H);
  bgGrad.addColorStop(0, "#1a0533");
  bgGrad.addColorStop(0.35, "#0f172a");
  bgGrad.addColorStop(0.7, "#0c1929");
  bgGrad.addColorStop(1, "#0a1a2a");
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, W, H);

  // Soft radial glow behind score area (upper center)
  const glowColor = scoreGlowColor(data.color);
  const glow = ctx.createRadialGradient(W / 2, 520, 0, W / 2, 520, 500);
  glow.addColorStop(0, glowColor + "30");
  glow.addColorStop(0.4, glowColor + "12");
  glow.addColorStop(1, "transparent");
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, W, H);

  // Second subtle glow at bottom for depth
  const glow2 = ctx.createRadialGradient(W / 2, H - 200, 0, W / 2, H - 200, 400);
  glow2.addColorStop(0, "#6366f110");
  glow2.addColorStop(1, "transparent");
  ctx.fillStyle = glow2;
  ctx.fillRect(0, 0, W, H);

  // Subtle noise-like dot pattern for texture
  ctx.fillStyle = "#ffffff03";
  for (let px = 0; px < W; px += 6) {
    for (let py = 0; py < H; py += 6) {
      if (Math.sin(px * 0.7 + py * 1.3) > 0.85) {
        ctx.fillRect(px, py, 1, 1);
      }
    }
  }

  // ── Top section: tagline + date ──────────────────────────────────────────
  let y = 140;

  // "Your Emotional DNA" tagline
  ctx.fillStyle = TEXT_DIM;
  ctx.font = "500 28px system-ui, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.letterSpacing = "6px";
  ctx.fillText("YOUR EMOTIONAL DNA", W / 2, y);
  ctx.letterSpacing = "0px";

  // Date
  y += 50;
  ctx.fillStyle = TEXT_DIM;
  ctx.font = "400 24px system-ui, -apple-system, sans-serif";
  const dateStr = new Date().toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });
  ctx.fillText(dateStr, W / 2, y);

  // Thin gradient divider line
  y += 40;
  const divGrad = ctx.createLinearGradient(PAD + 100, y, W - PAD - 100, y);
  divGrad.addColorStop(0, "transparent");
  divGrad.addColorStop(0.3, glowColor + "60");
  divGrad.addColorStop(0.5, glowColor + "90");
  divGrad.addColorStop(0.7, glowColor + "60");
  divGrad.addColorStop(1, "transparent");
  ctx.fillStyle = divGrad;
  ctx.fillRect(PAD + 100, y, W - 2 * PAD - 200, 2);

  // ── Score section (centered, large) ──────────────────────────────────────
  const displayScore = data.score ?? 0;
  const color = scoreColor(data.color);

  // Score circle background — frosted glass effect
  const scoreY = 580;
  const circleR = 180;

  // Outer ring glow
  ctx.strokeStyle = glowColor + "40";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(W / 2, scoreY, circleR + 20, 0, Math.PI * 2);
  ctx.stroke();

  // Score arc (progress ring) — shows score as fraction of circle
  const arcStart = -Math.PI / 2;
  const arcEnd = arcStart + (displayScore / 100) * Math.PI * 2;
  ctx.strokeStyle = color;
  ctx.lineWidth = 6;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.arc(W / 2, scoreY, circleR + 20, arcStart, arcEnd);
  ctx.stroke();
  ctx.lineCap = "butt";

  // Inner frosted circle
  const circBg = ctx.createRadialGradient(W / 2, scoreY, 0, W / 2, scoreY, circleR);
  circBg.addColorStop(0, "#1e293b80");
  circBg.addColorStop(1, "#0f172a60");
  ctx.fillStyle = circBg;
  ctx.beginPath();
  ctx.arc(W / 2, scoreY, circleR, 0, Math.PI * 2);
  ctx.fill();

  // Subtle border on circle
  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(W / 2, scoreY, circleR, 0, Math.PI * 2);
  ctx.stroke();

  // Score number — massive
  ctx.fillStyle = color;
  ctx.font = "bold 140px system-ui, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  // Text glow effect (draw blurred text behind)
  ctx.save();
  ctx.shadowColor = glowColor + "80";
  ctx.shadowBlur = 40;
  ctx.fillText(String(displayScore), W / 2, scoreY - 10);
  ctx.restore();

  // Crisp text on top
  ctx.fillText(String(displayScore), W / 2, scoreY - 10);

  // "/ 100" subtext
  ctx.fillStyle = TEXT_DIM;
  ctx.font = "400 28px system-ui, -apple-system, sans-serif";
  ctx.fillText("/ 100", W / 2, scoreY + 70);

  // Label (e.g. "Thriving", "Recovering")
  if (data.label) {
    ctx.fillStyle = color;
    ctx.font = "600 32px system-ui, -apple-system, sans-serif";
    ctx.letterSpacing = "4px";
    ctx.fillText(data.label.toUpperCase(), W / 2, scoreY + 130);
    ctx.letterSpacing = "0px";
  }

  ctx.textBaseline = "alphabetic";

  // ── Vital bars section ────────────────────────────────────────────────────
  const barsStartY = scoreY + 210;
  const barH = 16;
  const barGap = 70;
  const barPad = PAD + 20;
  const barW = W - 2 * barPad;

  // Section label
  ctx.fillStyle = TEXT_DIM;
  ctx.font = "500 22px system-ui, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.letterSpacing = "4px";
  ctx.fillText("VITALS", W / 2, barsStartY - 30);
  ctx.letterSpacing = "0px";

  ctx.textAlign = "left";

  VITAL_ORDER.forEach((key, i) => {
    const vital = data.vitals[key];
    if (!vital) return;

    const vy = barsStartY + i * barGap;
    const score = vital.score ?? 0;
    const [barStart, barEnd] = vital.status === "available"
      ? vitalBarColors(score)
      : [TEXT_DIM, TEXT_DIM];

    // Label
    ctx.fillStyle = TEXT_MUTED;
    ctx.font = "500 26px system-ui, -apple-system, sans-serif";
    ctx.fillText(VITAL_LABELS[key] ?? key, barPad, vy);

    // Score text on right
    ctx.fillStyle = vital.status === "available" ? barEnd : TEXT_DIM;
    ctx.font = "bold 26px system-ui, -apple-system, sans-serif";
    const scoreText = vital.status === "available" ? String(score) : "--";
    const scoreTw = ctx.measureText(scoreText).width;
    ctx.fillText(scoreText, barPad + barW - scoreTw, vy);

    // Bar background (track)
    const trackY = vy + 14;
    ctx.fillStyle = "#1e293b";
    fillRoundRect(ctx, barPad, trackY, barW, barH, barH / 2);

    // Subtle track border
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1;
    strokeRoundRect(ctx, barPad, trackY, barW, barH, barH / 2);

    // Bar fill with gradient
    if (vital.status === "available" && score > 0) {
      const fillW = Math.max(barH, (score / 100) * barW);
      const barGradient = ctx.createLinearGradient(barPad, 0, barPad + fillW, 0);
      barGradient.addColorStop(0, barStart);
      barGradient.addColorStop(1, barEnd);
      ctx.fillStyle = barGradient;
      fillRoundRect(ctx, barPad, trackY, fillW, barH, barH / 2);
    }
  });

  // ── Daily insight card ────────────────────────────────────────────────────
  if (data.dailyInsight) {
    const insightY = barsStartY + VITAL_ORDER.length * barGap + 30;
    const insightH = 160;

    // Card background — frosted glass
    const insightBg = ctx.createLinearGradient(PAD, insightY, PAD, insightY + insightH);
    insightBg.addColorStop(0, "#1e293b60");
    insightBg.addColorStop(1, "#0f172a40");
    ctx.fillStyle = insightBg;
    fillRoundRect(ctx, PAD, insightY, W - 2 * PAD, insightH, 16);

    // Border
    ctx.strokeStyle = "#33415540";
    ctx.lineWidth = 1;
    strokeRoundRect(ctx, PAD, insightY, W - 2 * PAD, insightH, 16);

    // Left accent bar with gradient
    const accentGrad = ctx.createLinearGradient(PAD, insightY, PAD, insightY + insightH);
    accentGrad.addColorStop(0, GREEN_BRIGHT);
    accentGrad.addColorStop(1, GREEN);
    ctx.fillStyle = accentGrad;
    fillRoundRect(ctx, PAD, insightY, 4, insightH, 2);

    // Insight text — word wrap
    ctx.fillStyle = TEXT_PRIMARY;
    ctx.font = "400 24px system-ui, -apple-system, sans-serif";
    ctx.textAlign = "left";
    const maxTextW = W - 2 * PAD - 60;
    const words = data.dailyInsight.text.split(" ");
    let line = "";
    let lineY = insightY + 42;
    const lineHeight = 34;
    let lineCount = 0;

    for (const word of words) {
      const test = line ? `${line} ${word}` : word;
      if (ctx.measureText(test).width > maxTextW && line) {
        ctx.fillText(line, PAD + 28, lineY);
        line = word;
        lineY += lineHeight;
        lineCount++;
        if (lineCount >= 2) break;
      } else {
        line = test;
      }
    }
    if (line) ctx.fillText(line, PAD + 28, lineY);

    // Action nudge
    if (data.dailyInsight.actionNudge) {
      ctx.fillStyle = TEXT_DIM;
      ctx.font = "400 20px system-ui, -apple-system, sans-serif";
      const nudgeText =
        data.dailyInsight.actionNudge.length > 70
          ? data.dailyInsight.actionNudge.slice(0, 67) + "..."
          : data.dailyInsight.actionNudge;
      ctx.fillText(nudgeText, PAD + 28, insightY + insightH - 24);
    }
  }

  // ── Footer / branding ─────────────────────────────────────────────────────
  // Gradient divider
  const footDivY = H - 140;
  const footGrad = ctx.createLinearGradient(PAD + 100, footDivY, W - PAD - 100, footDivY);
  footGrad.addColorStop(0, "transparent");
  footGrad.addColorStop(0.5, "#33415560");
  footGrad.addColorStop(1, "transparent");
  ctx.fillStyle = footGrad;
  ctx.fillRect(PAD + 100, footDivY, W - 2 * PAD - 200, 1);

  // Brand name
  ctx.fillStyle = TEXT_DIM;
  ctx.font = "500 24px system-ui, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.letterSpacing = "3px";
  ctx.fillText("NEURALDREAMWORKSHOP", W / 2, H - 90);
  ctx.letterSpacing = "0px";

  // Confidence label
  if (data.confidence === "early_estimate") {
    ctx.fillStyle = AMBER + "90";
    ctx.font = "400 20px system-ui, -apple-system, sans-serif";
    ctx.fillText("Early estimate", W / 2, H - 55);
  }

  // ── Download ──────────────────────────────────────────────────────────────
  const link = document.createElement("a");
  link.download = "emotional-fitness.png";
  link.href = canvas.toDataURL("image/png");
  link.click();
}
