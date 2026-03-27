/**
 * Share Card Generator — Canvas 2D PNG export with 5 template styles.
 *
 * All rendering is pure Canvas 2D — no DOM, no React. Generates a data URL
 * suitable for sharing via share-utils.ts.
 *
 * Templates render at 2x DPI for Retina displays:
 *   stories = 1080x1920 logical → 2160x3840 canvas
 *   square  = 1080x1080 logical → 2160x2160 canvas
 */

// ── Types ──────────────────────────────────────────────────────────────────

export type ShareTemplate =
  | "daily-overview"
  | "workout-summary"
  | "weekly-summary"
  | "brain-report"
  | "habit-streak";

export type ShareFormat = "stories" | "square";

export interface ShareData {
  // Daily overview
  recoveryScore?: number | null;
  sleepScore?: number | null;
  strainScore?: number | null;
  stressScore?: number | null;
  nutritionScore?: number | null;
  energyBank?: number | null;

  // Workout
  workoutName?: string;
  exercises?: { name: string; sets?: number; reps?: number }[];
  totalVolume?: number;
  durationMin?: number;
  caloriesBurned?: number;

  // Weekly
  weeklyScores?: {
    day: string;
    recovery?: number;
    sleep?: number;
    strain?: number;
    stress?: number;
  }[];
  dateRange?: string;
  bestMetric?: string;

  // Brain report
  emotions?: { label: string; value: number }[];
  focusHours?: number;
  brainAge?: number | null;

  // Habit streak
  streakDays?: number;
  completionRate?: number;
  last30Days?: boolean[]; // true = completed
}

// ── Colors ─────────────────────────────────────────────────────────────────

const CYAN = "#22d3ee";
const AMBER = "#fde68a";
const ROSE = "#fda4af";
const TEXT_PRIMARY = "#f8fafc";
const TEXT_MUTED = "#94a3b8";
const TEXT_DIM = "#64748b";
const TRACK_BG = "#1e293b";

function scoreColor(score: number | null | undefined): string {
  if (score == null) return TEXT_DIM;
  if (score >= 70) return CYAN;
  if (score >= 40) return AMBER;
  return ROSE;
}

function scoreGlow(score: number | null | undefined): string {
  if (score == null) return TEXT_DIM;
  if (score >= 70) return "#0891b2";
  if (score >= 40) return "#fbbf24";
  return "#f472b6";
}

// ── Gradient backgrounds ───────────────────────────────────────────────────

type GradientDef = { stops: [number, string][] };

const GRADIENTS: Record<ShareTemplate, GradientDef> = {
  "daily-overview": {
    stops: [
      [0, "#1a0533"],
      [0.4, "#0f172a"],
      [1, "#0c1929"],
    ],
  },
  "workout-summary": {
    stops: [
      [0, "#042f2e"],
      [0.4, "#0c1929"],
      [1, "#0f172a"],
    ],
  },
  "weekly-summary": {
    stops: [
      [0, "#1e1b4b"],
      [0.4, "#1a0533"],
      [1, "#0f172a"],
    ],
  },
  "brain-report": {
    stops: [
      [0, "#1a0533"],
      [0.5, "#2d0a4e"],
      [1, "#500724"],
    ],
  },
  "habit-streak": {
    stops: [
      [0, "#022c22"],
      [0.4, "#042f2e"],
      [1, "#0c1929"],
    ],
  },
};

// ── Canvas helpers ─────────────────────────────────────────────────────────

function drawGradientBg(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  def: GradientDef,
) {
  const grad = ctx.createLinearGradient(0, 0, w, h);
  for (const [stop, color] of def.stops) {
    grad.addColorStop(stop, color);
  }
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  // Subtle dot texture
  ctx.fillStyle = "#ffffff03";
  for (let px = 0; px < w; px += 8) {
    for (let py = 0; py < h; py += 8) {
      if (Math.sin(px * 0.7 + py * 1.3) > 0.85) {
        ctx.fillRect(px, py, 1, 1);
      }
    }
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

function drawCircularGauge(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  radius: number,
  value: number | null | undefined,
  label: string,
  color: string,
) {
  const score = value ?? 0;

  // Track
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.strokeStyle = TRACK_BG;
  ctx.lineWidth = 8;
  ctx.stroke();

  // Fill arc
  if (score > 0) {
    const start = -Math.PI / 2;
    const end = start + (score / 100) * Math.PI * 2;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, start, end);
    ctx.strokeStyle = color;
    ctx.lineWidth = 8;
    ctx.lineCap = "round";
    ctx.stroke();
    ctx.lineCap = "butt";
  }

  // Score text
  ctx.fillStyle = value != null ? color : TEXT_DIM;
  ctx.font = "bold 48px system-ui, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(value != null ? String(score) : "--", cx, cy - 4);

  // Label
  ctx.fillStyle = TEXT_MUTED;
  ctx.font = "500 20px system-ui, -apple-system, sans-serif";
  ctx.textBaseline = "alphabetic";
  ctx.fillText(label, cx, cy + radius + 28);
}

function drawBar(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  barWidth: number,
  height: number,
  value: number,
  maxValue: number,
  color: string,
) {
  // Track
  roundRect(ctx, x, y, barWidth, height, height / 2);
  ctx.fillStyle = TRACK_BG;
  ctx.fill();

  // Fill
  if (value > 0) {
    const fillW = Math.max(height, (value / maxValue) * barWidth);
    const grad = ctx.createLinearGradient(x, 0, x + fillW, 0);
    grad.addColorStop(0, color + "80");
    grad.addColorStop(1, color);
    ctx.fillStyle = grad;
    roundRect(ctx, x, y, fillW, height, height / 2);
    ctx.fill();
  }
}

function drawBranding(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  subtitle?: string,
) {
  // Divider
  const divY = h - 120;
  const grad = ctx.createLinearGradient(w * 0.2, divY, w * 0.8, divY);
  grad.addColorStop(0, "transparent");
  grad.addColorStop(0.5, "#33415560");
  grad.addColorStop(1, "transparent");
  ctx.fillStyle = grad;
  ctx.fillRect(w * 0.2, divY, w * 0.6, 1);

  // Brand
  drawText(ctx, "ANTARAI", w / 2, h - 70, {
    color: TEXT_DIM,
    size: 24,
    weight: "500",
    letterSpacing: "3px",
  });

  if (subtitle) {
    drawText(ctx, subtitle, w / 2, h - 38, {
      color: TEXT_DIM + "80",
      size: 18,
    });
  }
}

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

function formatDate(): string {
  return new Date().toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });
}

// ── Template renderers ─────────────────────────────────────────────────────

function renderDailyOverview(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  data: ShareData,
) {
  drawGradientBg(ctx, w, h, GRADIENTS["daily-overview"]);

  // Date header
  drawText(ctx, formatDate(), w / 2, 100, { color: TEXT_DIM, size: 24 });

  // Title
  drawText(ctx, "DAILY OVERVIEW", w / 2, 160, {
    color: TEXT_MUTED,
    size: 28,
    weight: "600",
    letterSpacing: "6px",
  });

  // 4 score gauges in 2x2 grid
  const scores = [
    { label: "Recovery", value: data.recoveryScore },
    { label: "Sleep", value: data.sleepScore },
    { label: "Strain", value: data.strainScore },
    { label: "Stress", value: data.stressScore },
  ];

  const isStories = h > w;
  const gaugeR = isStories ? 80 : 70;
  const colSpacing = w / 3;
  const row1Y = isStories ? 360 : 340;
  const row2Y = isStories ? 600 : 560;

  scores.forEach((s, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const cx = colSpacing + col * colSpacing;
    const cy = row === 0 ? row1Y : row2Y;
    drawCircularGauge(ctx, cx, cy, gaugeR, s.value, s.label, scoreColor(s.value));
  });

  // Energy Bank at bottom center
  if (data.energyBank != null) {
    const ebY = isStories ? h - 300 : h - 220;
    drawText(ctx, "ENERGY BANK", w / 2, ebY, {
      color: TEXT_DIM,
      size: 20,
      weight: "500",
      letterSpacing: "4px",
    });
    drawText(ctx, `${data.energyBank}%`, w / 2, ebY + 56, {
      color: scoreColor(data.energyBank),
      size: 56,
      weight: "bold",
    });
  }

  drawBranding(ctx, w, h);
}

function renderWorkoutSummary(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  data: ShareData,
) {
  drawGradientBg(ctx, w, h, GRADIENTS["workout-summary"]);

  // Header
  drawText(ctx, formatDate(), w / 2, 100, { color: TEXT_DIM, size: 24 });
  drawText(ctx, "WORKOUT COMPLETE", w / 2, 160, {
    color: "#5eead4",
    size: 32,
    weight: "700",
    letterSpacing: "4px",
  });

  // Workout name
  if (data.workoutName) {
    drawText(ctx, data.workoutName, w / 2, 220, {
      color: TEXT_PRIMARY,
      size: 36,
      weight: "600",
      maxWidth: w - 160,
    });
  }

  const PAD = 80;
  const isStories = h > w;
  let y = 300;

  // Exercise list
  if (data.exercises && data.exercises.length > 0) {
    const maxShow = isStories ? 8 : 5;
    const shown = data.exercises.slice(0, maxShow);
    ctx.textAlign = "left";

    for (const ex of shown) {
      drawText(ctx, ex.name, PAD, y, {
        color: TEXT_PRIMARY,
        size: 26,
        weight: "500",
        align: "left",
      });
      const detail =
        ex.sets && ex.reps ? `${ex.sets}x${ex.reps}` : "";
      if (detail) {
        drawText(ctx, detail, w - PAD, y, {
          color: TEXT_MUTED,
          size: 24,
          align: "right",
        });
      }
      y += 50;
    }

    if (data.exercises.length > maxShow) {
      drawText(ctx, `+${data.exercises.length - maxShow} more`, PAD, y, {
        color: TEXT_DIM,
        size: 22,
        align: "left",
      });
      y += 50;
    }
  }

  // Stats row
  y += 30;
  const divGrad = ctx.createLinearGradient(PAD, y, w - PAD, y);
  divGrad.addColorStop(0, "transparent");
  divGrad.addColorStop(0.5, "#5eead430");
  divGrad.addColorStop(1, "transparent");
  ctx.fillStyle = divGrad;
  ctx.fillRect(PAD, y, w - 2 * PAD, 1);
  y += 40;

  const stats: { label: string; value: string }[] = [];
  if (data.durationMin) stats.push({ label: "Duration", value: `${data.durationMin} min` });
  if (data.totalVolume) stats.push({ label: "Volume", value: `${data.totalVolume.toLocaleString()} kg` });
  if (data.caloriesBurned) stats.push({ label: "Calories", value: `${data.caloriesBurned} kcal` });

  for (const stat of stats) {
    drawText(ctx, stat.label, PAD, y, {
      color: TEXT_MUTED,
      size: 24,
      align: "left",
    });
    drawText(ctx, stat.value, w - PAD, y, {
      color: "#5eead4",
      size: 26,
      weight: "600",
      align: "right",
    });
    y += 48;
  }

  drawBranding(ctx, w, h);
}

function renderWeeklySummary(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  data: ShareData,
) {
  drawGradientBg(ctx, w, h, GRADIENTS["weekly-summary"]);

  // Header
  if (data.dateRange) {
    drawText(ctx, data.dateRange, w / 2, 100, { color: TEXT_DIM, size: 22 });
  }
  drawText(ctx, "WEEK IN REVIEW", w / 2, 155, {
    color: "#a78bfa",
    size: 32,
    weight: "700",
    letterSpacing: "4px",
  });

  const PAD = 80;
  const days = data.weeklyScores ?? [];
  const isStories = h > w;

  // 7-day mini bar charts for each score type
  const metrics: {
    label: string;
    key: keyof typeof days[number];
    color: string;
  }[] = [
    { label: "Recovery", key: "recovery", color: "#22d3ee" },
    { label: "Sleep", key: "sleep", color: "#818cf8" },
    { label: "Strain", key: "strain", color: "#f97316" },
    { label: "Stress", key: "stress", color: "#f472b6" },
  ];

  const sectionH = isStories ? 160 : 130;
  let startY = 220;

  for (const metric of metrics) {
    drawText(ctx, metric.label, PAD, startY, {
      color: TEXT_MUTED,
      size: 22,
      weight: "500",
      align: "left",
    });

    // Mini bar chart
    const barAreaX = PAD;
    const barAreaW = w - 2 * PAD;
    const barAreaY = startY + 16;
    const barMaxH = sectionH - 60;
    const barGap = 8;
    const barW = (barAreaW - barGap * (days.length - 1)) / Math.max(days.length, 1);

    for (let i = 0; i < days.length; i++) {
      const val = (days[i][metric.key] as number | undefined) ?? 0;
      const barH = Math.max(4, (val / 100) * barMaxH);
      const bx = barAreaX + i * (barW + barGap);
      const by = barAreaY + barMaxH - barH;

      roundRect(ctx, bx, by, barW, barH, 4);
      ctx.fillStyle = val > 0 ? metric.color : TRACK_BG;
      ctx.fill();

      // Day label
      drawText(ctx, days[i].day.slice(0, 1), bx + barW / 2, barAreaY + barMaxH + 18, {
        color: TEXT_DIM,
        size: 16,
      });
    }

    startY += sectionH;
  }

  // Best metric highlight
  if (data.bestMetric) {
    drawText(ctx, `Best: ${data.bestMetric}`, w / 2, startY + 20, {
      color: "#a78bfa",
      size: 24,
      weight: "600",
    });
  }

  drawBranding(ctx, w, h);
}

function renderBrainReport(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  data: ShareData,
) {
  drawGradientBg(ctx, w, h, GRADIENTS["brain-report"]);

  // Header
  drawText(ctx, formatDate(), w / 2, 100, { color: TEXT_DIM, size: 24 });
  drawText(ctx, "BRAIN REPORT", w / 2, 160, {
    color: "#c084fc",
    size: 32,
    weight: "700",
    letterSpacing: "4px",
  });

  // EEG-powered badge
  const badgeW = 180;
  const badgeH = 36;
  const badgeX = w / 2 - badgeW / 2;
  const badgeY = 190;
  roundRect(ctx, badgeX, badgeY, badgeW, badgeH, badgeH / 2);
  ctx.fillStyle = "#7c3aed30";
  ctx.fill();
  ctx.strokeStyle = "#7c3aed60";
  ctx.lineWidth = 1;
  roundRect(ctx, badgeX, badgeY, badgeW, badgeH, badgeH / 2);
  ctx.stroke();
  drawText(ctx, "EEG-powered", w / 2, badgeY + 24, {
    color: "#c084fc",
    size: 18,
    weight: "500",
  });

  const PAD = 80;
  let y = 280;
  const isStories = h > w;

  // Emotion distribution as horizontal bars
  if (data.emotions && data.emotions.length > 0) {
    drawText(ctx, "EMOTION DISTRIBUTION", PAD, y, {
      color: TEXT_DIM,
      size: 20,
      weight: "500",
      align: "left",
      letterSpacing: "3px",
    });
    y += 30;

    const barW = w - 2 * PAD - 100;
    const barH = 16;
    const maxVal = Math.max(...data.emotions.map((e) => e.value), 1);

    const emotionColors: Record<string, string> = {
      happy: "#22d3ee",
      sad: "#818cf8",
      angry: "#f97316",
      fear: "#f472b6",
      surprise: "#fbbf24",
      neutral: "#94a3b8",
    };

    for (const emo of data.emotions) {
      drawText(ctx, emo.label, PAD, y + 13, {
        color: TEXT_MUTED,
        size: 22,
        align: "left",
      });

      const color = emotionColors[emo.label.toLowerCase()] ?? "#94a3b8";
      drawBar(ctx, PAD + 120, y, barW, barH, emo.value, maxVal, color);

      // Percentage
      const pct = maxVal > 0 ? Math.round((emo.value / maxVal) * 100) : 0;
      drawText(ctx, `${pct}%`, w - PAD, y + 13, {
        color: TEXT_MUTED,
        size: 20,
        align: "right",
      });

      y += isStories ? 52 : 44;
    }
  }

  // Focus hours
  y += 20;
  if (data.focusHours != null) {
    drawText(ctx, "Focus Time", PAD, y, {
      color: TEXT_MUTED,
      size: 24,
      align: "left",
    });
    drawText(ctx, `${data.focusHours.toFixed(1)}h`, w - PAD, y, {
      color: "#c084fc",
      size: 28,
      weight: "bold",
      align: "right",
    });
    y += 50;
  }

  // Brain age
  if (data.brainAge != null) {
    drawText(ctx, "Brain Age", PAD, y, {
      color: TEXT_MUTED,
      size: 24,
      align: "left",
    });
    drawText(ctx, `${data.brainAge}`, w - PAD, y, {
      color: "#c084fc",
      size: 28,
      weight: "bold",
      align: "right",
    });
  }

  drawBranding(ctx, w, h, "Powered by EEG");
}

function renderHabitStreak(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  data: ShareData,
) {
  drawGradientBg(ctx, w, h, GRADIENTS["habit-streak"]);

  // Header
  drawText(ctx, formatDate(), w / 2, 100, { color: TEXT_DIM, size: 24 });
  drawText(ctx, "STREAK REPORT", w / 2, 160, {
    color: "#34d399",
    size: 32,
    weight: "700",
    letterSpacing: "4px",
  });

  const isStories = h > w;

  // Large streak number with fire emoji
  const streakY = isStories ? 380 : 340;
  drawText(ctx, String(data.streakDays ?? 0), w / 2, streakY, {
    color: "#34d399",
    size: 140,
    weight: "bold",
  });

  // Fire emoji below
  drawText(ctx, "days", w / 2, streakY + 60, {
    color: TEXT_MUTED,
    size: 32,
    weight: "500",
  });

  // Completion rate
  if (data.completionRate != null) {
    const rateY = streakY + 130;
    drawText(ctx, `${data.completionRate}% completion`, w / 2, rateY, {
      color: TEXT_PRIMARY,
      size: 28,
      weight: "500",
    });
  }

  // 30-day mini grid
  if (data.last30Days && data.last30Days.length > 0) {
    const gridY = isStories ? h - 400 : h - 280;
    drawText(ctx, "LAST 30 DAYS", w / 2, gridY - 20, {
      color: TEXT_DIM,
      size: 20,
      weight: "500",
      letterSpacing: "3px",
    });

    const cols = 10;
    const rows = 3;
    const cellSize = 32;
    const gap = 8;
    const gridW = cols * (cellSize + gap) - gap;
    const startX = (w - gridW) / 2;

    for (let i = 0; i < Math.min(data.last30Days.length, 30); i++) {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const cx = startX + col * (cellSize + gap);
      const cy = gridY + row * (cellSize + gap);

      roundRect(ctx, cx, cy, cellSize, cellSize, 6);
      ctx.fillStyle = data.last30Days[i] ? "#34d399" : TRACK_BG;
      ctx.fill();
    }
  }

  drawBranding(ctx, w, h);
}

// ── Main export ────────────────────────────────────────────────────────────

const RENDERERS: Record<
  ShareTemplate,
  (ctx: CanvasRenderingContext2D, w: number, h: number, data: ShareData) => void
> = {
  "daily-overview": renderDailyOverview,
  "workout-summary": renderWorkoutSummary,
  "weekly-summary": renderWeeklySummary,
  "brain-report": renderBrainReport,
  "habit-streak": renderHabitStreak,
};

/**
 * Generate a share card PNG as a data URL.
 *
 * All rendering is at 2x DPI for Retina; the logical dimensions are:
 *   stories: 1080 x 1920
 *   square:  1080 x 1080
 */
export async function generateShareCard(
  template: ShareTemplate,
  data: ShareData,
  format: ShareFormat = "stories",
): Promise<string> {
  const DPI = 2;
  const logicalW = 1080;
  const logicalH = format === "stories" ? 1920 : 1080;
  const W = logicalW * DPI;
  const H = logicalH * DPI;

  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Cannot create 2D canvas context");

  // Scale all drawing ops to 2x
  ctx.scale(DPI, DPI);

  const renderer = RENDERERS[template];
  renderer(ctx, logicalW, logicalH, data);

  return canvas.toDataURL("image/png");
}
