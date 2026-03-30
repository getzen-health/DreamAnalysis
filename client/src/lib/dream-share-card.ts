/**
 * Dream Share Card — Canvas 2D renderer for Instagram Stories sharing.
 *
 * Renders a 1080x1920 (2x DPI = 2160x3840) card with:
 *   - Dark gradient background (deep blue to purple)
 *   - Moon/star decorations
 *   - Dream summary text
 *   - Emotional tone badge
 *   - Sleep stats row (duration, REM%, dream count)
 *   - App branding at bottom
 *   - Date in corner
 */

// ── Types ──────────────────────────────────────────────────────────────────

export interface DreamShareData {
  dreamSummary: string;       // 1-2 sentence dream summary
  emotionalTone: string;      // "peaceful", "intense", etc.
  sleepDuration: string;      // "7h 23m"
  remPercentage: number;
  dreamCount: number;
  date: string;               // "March 30, 2026"
}

// ── Colors ─────────────────────────────────────────────────────────────────

const TEXT_PRIMARY = "#f8fafc";
const TEXT_MUTED = "#94a3b8";
const TEXT_DIM = "#64748b";
const ACCENT_PURPLE = "#a78bfa";
const ACCENT_INDIGO = "#818cf8";
const ACCENT_CYAN = "#22d3ee";

// ── Tone color mapping ────────────────────────────────────────────────────

const TONE_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  peaceful:    { bg: "#818cf820", text: "#a5b4fc", border: "#818cf840" },
  intense:     { bg: "#f4727220", text: "#fca5a5", border: "#f4727240" },
  joyful:      { bg: "#fbbf2420", text: "#fde68a", border: "#fbbf2440" },
  anxious:     { bg: "#f9731620", text: "#fdba74", border: "#f9731640" },
  mysterious:  { bg: "#a78bfa20", text: "#c4b5fd", border: "#a78bfa40" },
  melancholic: { bg: "#60a5fa20", text: "#93c5fd", border: "#60a5fa40" },
  surreal:     { bg: "#c084fc20", text: "#d8b4fe", border: "#c084fc40" },
  neutral:     { bg: "#64748b20", text: "#94a3b8", border: "#64748b40" },
};

function getToneColors(tone: string): { bg: string; text: string; border: string } {
  const key = tone.toLowerCase().trim();
  return TONE_COLORS[key] ?? TONE_COLORS.neutral;
}

// ── Canvas helpers ─────────────────────────────────────────────────────────

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

function drawText(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  opts: {
    color?: string;
    size?: number;
    weight?: string;
    align?: CanvasTextAlign;
    letterSpacing?: string;
    maxWidth?: number;
  } = {},
) {
  const {
    color = TEXT_PRIMARY,
    size = 28,
    weight = "400",
    align = "center",
    letterSpacing,
    maxWidth,
  } = opts;
  ctx.fillStyle = color;
  ctx.font = `${weight} ${size}px system-ui, -apple-system, sans-serif`;
  ctx.textAlign = align;
  if (letterSpacing) ctx.letterSpacing = letterSpacing;
  if (maxWidth) {
    ctx.fillText(text, x, y, maxWidth);
  } else {
    ctx.fillText(text, x, y);
  }
  if (letterSpacing) ctx.letterSpacing = "0px";
}

/**
 * Word-wrap text to fit within maxWidth, returning an array of lines.
 */
function wrapText(
  ctx: CanvasRenderingContext2D,
  text: string,
  maxWidth: number,
  fontSize: number,
  fontWeight: string = "400",
): string[] {
  ctx.font = `${fontWeight} ${fontSize}px system-ui, -apple-system, sans-serif`;
  const words = text.split(" ");
  const lines: string[] = [];
  let currentLine = "";

  for (const word of words) {
    const testLine = currentLine ? `${currentLine} ${word}` : word;
    const metrics = ctx.measureText(testLine);
    if (metrics.width > maxWidth && currentLine) {
      lines.push(currentLine);
      currentLine = word;
    } else {
      currentLine = testLine;
    }
  }
  if (currentLine) lines.push(currentLine);
  return lines;
}

// ── Background ─────────────────────────────────────────────────────────────

function drawBackground(ctx: CanvasRenderingContext2D, w: number, h: number) {
  // Deep blue to purple gradient (diagonal)
  const grad = ctx.createLinearGradient(0, 0, w * 0.3, h);
  grad.addColorStop(0, "#0c0a2a");
  grad.addColorStop(0.35, "#1a103d");
  grad.addColorStop(0.7, "#120b3b");
  grad.addColorStop(1, "#0a0e2e");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  // Subtle dot texture (matches share-card-generator pattern)
  ctx.fillStyle = "#ffffff03";
  for (let px = 0; px < w; px += 8) {
    for (let py = 0; py < h; py += 8) {
      if (Math.sin(px * 0.7 + py * 1.3) > 0.85) {
        ctx.fillRect(px, py, 1, 1);
      }
    }
  }
}

// ── Stars ──────────────────────────────────────────────────────────────────

function drawStars(ctx: CanvasRenderingContext2D, w: number, h: number) {
  // Deterministic pseudo-random stars using sine hash
  const starCount = 40;
  for (let i = 0; i < starCount; i++) {
    const sx = ((Math.sin(i * 127.1 + 311.7) * 0.5 + 0.5) * w);
    const sy = ((Math.sin(i * 269.5 + 183.3) * 0.5 + 0.5) * h * 0.6);
    const size = (Math.sin(i * 43.7) * 0.5 + 0.5) * 2 + 0.5;
    const alpha = (Math.sin(i * 77.3) * 0.5 + 0.5) * 0.4 + 0.1;

    ctx.beginPath();
    ctx.arc(sx, sy, size, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
    ctx.fill();
  }
}

// ── Moon icon ──────────────────────────────────────────────────────────────

function drawMoon(ctx: CanvasRenderingContext2D, cx: number, cy: number, radius: number) {
  // Crescent moon: outer circle minus offset inner circle
  ctx.save();

  // Glow
  const glowGrad = ctx.createRadialGradient(cx, cy, radius * 0.5, cx, cy, radius * 2.5);
  glowGrad.addColorStop(0, "#a78bfa20");
  glowGrad.addColorStop(1, "#a78bfa00");
  ctx.fillStyle = glowGrad;
  ctx.beginPath();
  ctx.arc(cx, cy, radius * 2.5, 0, Math.PI * 2);
  ctx.fill();

  // Moon body (gradient fill)
  const moonGrad = ctx.createLinearGradient(cx - radius, cy - radius, cx + radius, cy + radius);
  moonGrad.addColorStop(0, "#e2e8f0");
  moonGrad.addColorStop(1, "#cbd5e1");
  ctx.fillStyle = moonGrad;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fill();

  // Cut out crescent by drawing a circle offset to the right
  ctx.globalCompositeOperation = "destination-out";
  ctx.beginPath();
  ctx.arc(cx + radius * 0.55, cy - radius * 0.1, radius * 0.85, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

// ── Emotional tone badge ───────────────────────────────────────────────────

function drawToneBadge(
  ctx: CanvasRenderingContext2D,
  cx: number,
  y: number,
  tone: string,
) {
  const colors = getToneColors(tone);
  const label = tone.charAt(0).toUpperCase() + tone.slice(1).toLowerCase();

  ctx.font = "500 22px system-ui, -apple-system, sans-serif";
  const textWidth = ctx.measureText(label).width;
  const paddingX = 28;
  const badgeW = textWidth + paddingX * 2;
  const badgeH = 42;
  const bx = cx - badgeW / 2;

  // Badge background
  roundRect(ctx, bx, y, badgeW, badgeH, badgeH / 2);
  ctx.fillStyle = colors.bg;
  ctx.fill();

  // Badge border
  roundRect(ctx, bx, y, badgeW, badgeH, badgeH / 2);
  ctx.strokeStyle = colors.border;
  ctx.lineWidth = 1;
  ctx.stroke();

  // Badge text
  drawText(ctx, label, cx, y + 29, {
    color: colors.text,
    size: 22,
    weight: "500",
  });
}

// ── Sleep stats row ────────────────────────────────────────────────────────

function drawStatBox(
  ctx: CanvasRenderingContext2D,
  cx: number,
  y: number,
  value: string,
  label: string,
  color: string,
) {
  // Value
  drawText(ctx, value, cx, y, {
    color,
    size: 36,
    weight: "bold",
  });

  // Label
  drawText(ctx, label, cx, y + 30, {
    color: TEXT_DIM,
    size: 18,
    weight: "400",
  });
}

function drawStatsRow(
  ctx: CanvasRenderingContext2D,
  w: number,
  y: number,
  data: DreamShareData,
) {
  const PAD = 80;

  // Divider line above stats
  const divGrad = ctx.createLinearGradient(PAD, y, w - PAD, y);
  divGrad.addColorStop(0, "transparent");
  divGrad.addColorStop(0.5, "#a78bfa30");
  divGrad.addColorStop(1, "transparent");
  ctx.fillStyle = divGrad;
  ctx.fillRect(PAD, y, w - 2 * PAD, 1);

  const statsY = y + 60;
  const colW = (w - 2 * PAD) / 3;

  drawStatBox(ctx, PAD + colW * 0.5, statsY, data.sleepDuration, "Sleep", ACCENT_INDIGO);
  drawStatBox(ctx, PAD + colW * 1.5, statsY, `${data.remPercentage}%`, "REM", ACCENT_CYAN);
  drawStatBox(ctx, PAD + colW * 2.5, statsY, String(data.dreamCount), data.dreamCount === 1 ? "Dream" : "Dreams", ACCENT_PURPLE);

  // Divider line below stats
  const divY2 = statsY + 50;
  const divGrad2 = ctx.createLinearGradient(PAD, divY2, w - PAD, divY2);
  divGrad2.addColorStop(0, "transparent");
  divGrad2.addColorStop(0.5, "#a78bfa30");
  divGrad2.addColorStop(1, "transparent");
  ctx.fillStyle = divGrad2;
  ctx.fillRect(PAD, divY2, w - 2 * PAD, 1);
}

// ── Branding ───────────────────────────────────────────────────────────────

function drawBranding(ctx: CanvasRenderingContext2D, w: number, h: number) {
  // Divider
  const divY = h - 140;
  const grad = ctx.createLinearGradient(w * 0.2, divY, w * 0.8, divY);
  grad.addColorStop(0, "transparent");
  grad.addColorStop(0.5, "#33415560");
  grad.addColorStop(1, "transparent");
  ctx.fillStyle = grad;
  ctx.fillRect(w * 0.2, divY, w * 0.6, 1);

  // App name
  drawText(ctx, "Tracked with ANTARAI", w / 2, h - 85, {
    color: TEXT_DIM,
    size: 22,
    weight: "400",
  });

  // Subtitle
  drawText(ctx, "EEG-Powered Dream Analysis", w / 2, h - 52, {
    color: TEXT_DIM + "80",
    size: 16,
  });
}

// ── Main renderer ──────────────────────────────────────────────────────────

export async function renderDreamShareCard(
  canvas: HTMLCanvasElement,
  data: DreamShareData,
): Promise<Blob> {
  const DPI = 2;
  const logicalW = 1080;
  const logicalH = 1920;
  const W = logicalW * DPI;
  const H = logicalH * DPI;

  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Cannot create 2D canvas context");

  // Scale all drawing ops to 2x for Retina
  ctx.scale(DPI, DPI);

  // 1. Background gradient
  drawBackground(ctx, logicalW, logicalH);

  // 2. Stars scattered across upper portion
  drawStars(ctx, logicalW, logicalH);

  // 3. Moon icon at top center
  drawMoon(ctx, logicalW / 2, 220, 50);

  // 4. Date below moon
  drawText(ctx, data.date, logicalW / 2, 340, {
    color: TEXT_DIM,
    size: 22,
    weight: "400",
  });

  // 5. Title
  drawText(ctx, "LAST NIGHT'S DREAM", logicalW / 2, 400, {
    color: TEXT_MUTED,
    size: 26,
    weight: "600",
    letterSpacing: "5px",
  });

  // 6. Emotional tone badge
  drawToneBadge(ctx, logicalW / 2, 440, data.emotionalTone);

  // 7. Dream summary text (word-wrapped)
  const PAD = 100;
  const maxTextWidth = logicalW - 2 * PAD;
  const summaryLines = wrapText(ctx, data.dreamSummary, maxTextWidth, 28, "400");
  const lineHeight = 40;
  const summaryStartY = 550;

  // Decorative quote marks
  drawText(ctx, "\u201C", logicalW / 2, summaryStartY - 20, {
    color: ACCENT_PURPLE + "60",
    size: 64,
    weight: "300",
  });

  for (let i = 0; i < Math.min(summaryLines.length, 6); i++) {
    drawText(ctx, summaryLines[i], logicalW / 2, summaryStartY + i * lineHeight, {
      color: TEXT_PRIMARY,
      size: 28,
      weight: "400",
    });
  }

  // Closing quote mark
  const lastLineY = summaryStartY + (Math.min(summaryLines.length, 6) - 1) * lineHeight;
  drawText(ctx, "\u201D", logicalW / 2, lastLineY + 50, {
    color: ACCENT_PURPLE + "60",
    size: 64,
    weight: "300",
  });

  // 8. Sleep stats row — positioned below summary text
  const statsRowY = lastLineY + 120;
  drawStatsRow(ctx, logicalW, statsRowY, data);

  // 9. Branding at bottom
  drawBranding(ctx, logicalW, logicalH);

  // Export as Blob
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) resolve(blob);
        else reject(new Error("canvas.toBlob returned null"));
      },
      "image/png",
    );
  });
}
