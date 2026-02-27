"""
Generate app icons for Svapnastra (NeuralDreamWorkshop).
Sizes: 1024, 512, 192, 180 (Apple touch), maskable 512
Run: python3 scripts/generate_icons.py
"""

from PIL import Image, ImageDraw, ImageFilter
import math
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "client", "public")

# Brand colors
BG_TOP    = (11, 13, 20)      # deep navy
BG_BTM    = (16, 20, 35)      # slightly lighter navy
GREEN     = (45, 212, 160)    # hsl(152, 60%, 50%)
GOLD      = (245, 166, 35)    # hsl(38, 92%, 55%)
WHITE     = (255, 255, 255)


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def make_bg(size):
    """Vertical gradient background."""
    import numpy as np
    arr = []
    for y in range(size):
        t = y / max(size - 1, 1)
        c = lerp_color(BG_TOP, BG_BTM, t)
        arr.append([list(c + (255,))] * size)
    import numpy as np
    return Image.fromarray(__import__('numpy').array(arr, dtype='uint8'), 'RGBA')


def draw_icon(size, maskable=False):
    """
    Draw Svapnastra icon:
    - Dark gradient background (rounded square for regular, full square for maskable)
    - Central glowing orb (green→gold radial gradient)
    - 6 neural node dots orbiting the orb
    - Crescent moon arc above (dream layer)
    - Subtle radiating lines
    """
    img = make_bg(size)
    draw = ImageDraw.Draw(img, 'RGBA')

    cx, cy = size // 2, size // 2
    r_outer = size * 0.28          # outer glow radius
    r_core  = size * 0.14          # bright core radius

    # --- Rounded square mask for non-maskable ---
    if not maskable:
        corner = size * 0.22
        mask = Image.new('L', (size, size), 0)
        mdraw = ImageDraw.Draw(mask)
        mdraw.rounded_rectangle([(0, 0), (size - 1, size - 1)], radius=corner, fill=255)
        img.putalpha(mask)

    # --- Radial glow (layered circles, outside-in) ---
    layers = 32
    for i in range(layers, 0, -1):
        t = i / layers
        rad = int(r_outer * t)
        # Color: outer = gold at low alpha, inner = green at higher alpha
        c = lerp_color(GOLD, GREEN, 1 - t)
        alpha = int(30 + 120 * (1 - t) ** 1.5)
        bbox = [cx - rad, cy - rad, cx + rad, cy + rad]
        draw.ellipse(bbox, fill=c + (alpha,))

    # --- Bright core ---
    core_r = int(r_core)
    draw.ellipse(
        [cx - core_r, cy - core_r, cx + core_r, cy + core_r],
        fill=GREEN + (220,)
    )
    # Specular highlight on core
    hl_r = int(core_r * 0.45)
    hl_off = int(core_r * 0.22)
    draw.ellipse(
        [cx - hl_r - hl_off, cy - hl_r - hl_off,
         cx + hl_r - hl_off, cy + hl_r - hl_off],
        fill=(255, 255, 255, 90)
    )

    # --- 6 neural nodes orbiting ---
    node_orbit = size * 0.38
    node_r = int(size * 0.035)
    for i in range(6):
        angle = math.pi / 2 + i * (2 * math.pi / 6)   # start from top
        nx = int(cx + node_orbit * math.cos(angle))
        ny = int(cy - node_orbit * math.sin(angle))
        t_node = i / 6
        nc = lerp_color(GREEN, GOLD, t_node)
        # Line from core to node
        draw.line(
            [(cx, cy), (nx, ny)],
            fill=nc + (55,),
            width=max(1, int(size * 0.006))
        )
        # Node dot
        draw.ellipse(
            [nx - node_r, ny - node_r, nx + node_r, ny + node_r],
            fill=nc + (200,)
        )
        # Small inner highlight
        hi_r = max(1, node_r // 3)
        draw.ellipse(
            [nx - hi_r, ny - hi_r, nx + hi_r, ny + hi_r],
            fill=(255, 255, 255, 120)
        )

    # --- Crescent moon (dream symbol) above the orb ---
    moon_cx = cx
    moon_cy = int(cy - size * 0.18)
    moon_r  = int(size * 0.075)
    moon_alpha = 200
    # Full moon circle
    draw.ellipse(
        [moon_cx - moon_r, moon_cy - moon_r,
         moon_cx + moon_r, moon_cy + moon_r],
        fill=GOLD + (moon_alpha,)
    )
    # Bite-out circle to create crescent (offset right + down)
    bite_off = int(moon_r * 0.52)
    bite_r   = int(moon_r * 0.82)
    draw.ellipse(
        [moon_cx + bite_off - bite_r, moon_cy - bite_off - bite_r,
         moon_cx + bite_off + bite_r, moon_cy - bite_off + bite_r],
        fill=BG_TOP + (255,)
    )

    # --- Subtle ring halo around entire composition ---
    ring_r = int(size * 0.44)
    ring_w = max(1, int(size * 0.008))
    draw.arc(
        [cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r],
        start=0, end=360,
        fill=GREEN + (35,),
        width=ring_w
    )

    # Slight Gaussian blur for the glow (creates bloom effect)
    img = img.filter(ImageFilter.GaussianBlur(radius=size * 0.008))

    # Re-apply the rounded-square mask after blur (blur bleeds edges)
    if not maskable:
        mask2 = Image.new('L', (size, size), 0)
        mdraw2 = ImageDraw.Draw(mask2)
        mdraw2.rounded_rectangle([(0, 0), (size - 1, size - 1)], radius=size * 0.22, fill=255)
        img.putalpha(mask2)

    return img


def main():
    os.makedirs(OUT, exist_ok=True)

    specs = [
        ("icon-1024.png",       1024, False),
        ("icon-512.png",         512, False),
        ("icon-192.png",         192, False),
        ("icon-180.png",         180, False),   # Apple touch icon
        ("icon-maskable-512.png", 512, True),
        ("icon-maskable-192.png", 192, True),
    ]

    for filename, size, maskable in specs:
        path = os.path.join(OUT, filename)
        icon = draw_icon(size, maskable=maskable)
        # Save with alpha for non-maskable; flatten onto solid bg for maskable
        if maskable:
            bg = Image.new('RGBA', (size, size), BG_TOP + (255,))
            bg.paste(icon, mask=icon.split()[3] if icon.mode == 'RGBA' else None)
            bg.convert('RGB').save(path, 'PNG', optimize=True)
        else:
            icon.save(path, 'PNG', optimize=True)
        print(f"  {filename}  ({size}x{size})")

    # Also copy 512 as the legacy favicon.png
    import shutil
    shutil.copy(os.path.join(OUT, "icon-512.png"), os.path.join(OUT, "favicon.png"))
    print("  favicon.png  (copied from icon-512)")

    print("\nAll icons written to client/public/")


if __name__ == "__main__":
    main()
