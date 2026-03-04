#!/usr/bin/env python3
"""
Generate Neural Dream Workshop app icon (1024×1024 PNG).

Design:
  - Deep indigo gradient background
  - Two-hemisphere brain made from overlapping ellipses + fold lines
  - Teal EEG wave below
  - Star accents
"""

import math
import pathlib
from PIL import Image, ImageDraw

SIZE = 1024

OUT_PATHS = [
    pathlib.Path(__file__).parent.parent
    / "ios/App/App/Assets.xcassets/AppIcon.appiconset/AppIcon-512@2x.png",
]


# ── helpers ──────────────────────────────────────────────────────────────────

def lerp(a, b, t):
    return a + (b - a) * t

def lerp_color(c1, c2, t):
    return tuple(round(lerp(a, b, t)) for a, b in zip(c1, c2))


def radial_gradient(draw, cx, cy, r_max, c_inner, c_outer, steps=100):
    for i in range(steps, -1, -1):
        t = i / steps
        r = r_max * t
        c = lerp_color(c_inner, c_outer, t)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=c)


def draw_wave(draw, cx, y_center, width, amplitude, cycles, n_lines=3,
              color_start=(90, 220, 170), color_end=(50, 180, 130), line_w=8):
    """Draw EEG-style sine wave as a series of short line segments."""
    for ln in range(n_lines):
        y_off = (ln - n_lines // 2) * (line_w + 2)
        t_col = ln / max(n_lines - 1, 1)
        col = lerp_color(color_start, color_end, t_col) + (220 - ln * 30,)
        pts = []
        n = 300
        for i in range(n + 1):
            x = cx - width / 2 + i / n * width
            phase = i / n * math.pi * 2 * cycles
            envelope = math.sin(i / n * math.pi) ** 0.6
            y = y_center + y_off + math.sin(phase) * amplitude * envelope
            pts.append((x, y))
        for j in range(len(pts) - 1):
            draw.line([pts[j], pts[j + 1]], fill=col, width=line_w)


def draw_star(draw, x, y, outer_r, inner_r, n=5, color=(200, 240, 220, 180)):
    pts = []
    for k in range(n * 2):
        angle = math.radians(-90 + k * 180 / n)
        r = outer_r if k % 2 == 0 else inner_r
        pts.append((x + r * math.cos(angle), y + r * math.sin(angle)))
    draw.polygon(pts, fill=color)


# ── build ─────────────────────────────────────────────────────────────────────

img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 255))
draw = ImageDraw.Draw(img)

# 1. Background
radial_gradient(draw, SIZE / 2, SIZE / 2.2, SIZE * 0.72,
                (28, 32, 75), (6, 8, 22))

# 2. Subtle inner glow (teal haze behind brain)
glow_layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
gd = ImageDraw.Draw(glow_layer)
for r in range(260, 0, -6):
    a = int(55 * (1 - r / 260) ** 1.5)
    gd.ellipse(
        (SIZE // 2 - r, int(SIZE * 0.40) - r,
         SIZE // 2 + r, int(SIZE * 0.40) + r),
        fill=(40, 200, 140, a),
    )
img = Image.alpha_composite(img, glow_layer)
draw = ImageDraw.Draw(img)

# 3. Brain — two offset ellipses for left/right hemispheres
BX = SIZE // 2
BY = int(SIZE * 0.38)

# Shared ellipse parameters
LW, LH = int(SIZE * 0.21), int(SIZE * 0.27)   # left hemisphere half-axes
RW, RH = int(SIZE * 0.21), int(SIZE * 0.27)   # right hemisphere half-axes
OFFSET = int(SIZE * 0.13)                      # centre-to-centre

LEFT_CX  = BX - OFFSET // 2
RIGHT_CX = BX + OFFSET // 2

BRAIN_GREEN  = (35, 185, 120, 230)
BRAIN_GREEN2 = (22, 145, 90, 240)
FOLD_COLOR   = (15, 100, 65, 200)
FOLD_COLOR2  = (255, 255, 255, 130)

# Draw right hemisphere first (behind)
draw.ellipse(
    (RIGHT_CX - RW, BY - RH, RIGHT_CX + RW, BY + RH),
    fill=BRAIN_GREEN,
)
# Left hemisphere on top
draw.ellipse(
    (LEFT_CX - LW, BY - LH, LEFT_CX + LW, BY + LH),
    fill=BRAIN_GREEN,
)

# Corpus callosum dividing line
for dw in range(3, -1, -1):
    alpha = 180 - dw * 40
    draw.line(
        [(BX, BY - int(LH * 0.75)), (BX, BY + int(LH * 0.70))],
        fill=(10, 80, 50, alpha),
        width=int(SIZE * 0.014) + dw * 2,
    )

# Surface fold arcs on left hemisphere
folds_left = [
    # (arc bounding box relative to LEFT_CX/BY, start_angle, end_angle, width_frac)
    ((-LW * 0.9, -LH * 0.85, LW * 0.05, -LH * 0.30), 200, 340, 0.012),
    ((-LW * 0.8, -LH * 0.45,  LW * 0.10, LH * 0.10), 195, 345, 0.011),
    ((-LW * 0.7,  LH * 0.00,  LW * 0.08, LH * 0.55), 205, 350, 0.010),
]
for (x0, y0, x1, y1), sa, ea, wf in folds_left:
    draw.arc(
        (LEFT_CX + x0, BY + y0, LEFT_CX + x1, BY + y1),
        start=sa, end=ea,
        fill=FOLD_COLOR2,
        width=int(SIZE * wf),
    )

# Surface fold arcs on right hemisphere
folds_right = [
    ((-RW * 0.05, -RH * 0.85, RW * 0.90, -RH * 0.30), 200, 340, 0.012),
    ((-RW * 0.08, -RH * 0.45, RW * 0.80,  RH * 0.10), 195, 345, 0.011),
    ((-RW * 0.08,  RH * 0.00, RW * 0.72,  RH * 0.55), 205, 350, 0.010),
]
for (x0, y0, x1, y1), sa, ea, wf in folds_right:
    draw.arc(
        (RIGHT_CX + x0, BY + y0, RIGHT_CX + x1, BY + y1),
        start=sa, end=ea,
        fill=FOLD_COLOR2,
        width=int(SIZE * wf),
    )

# 4. EEG wave
draw_wave(
    draw,
    cx=SIZE // 2,
    y_center=int(SIZE * 0.72),
    width=int(SIZE * 0.68),
    amplitude=int(SIZE * 0.055),
    cycles=3,
    n_lines=3,
    color_start=(80, 220, 160),
    color_end=(40, 170, 120),
    line_w=int(SIZE * 0.013),
)

# 5. Stars
stars = [
    (0.18, 0.18, 14),
    (0.80, 0.14, 11),
    (0.10, 0.65, 10),
    (0.86, 0.60, 13),
    (0.72, 0.86, 9),
    (0.28, 0.88, 8),
]
for sx, sy, sr in stars:
    draw_star(draw, sx * SIZE, sy * SIZE, sr, sr * 0.42, color=(200, 240, 225, 190))

# 6. Save
for path in OUT_PATHS:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(str(path), "PNG", optimize=True)
    print(f"Saved {path}  ({path.stat().st_size // 1024} KB)")

print("Done.")
