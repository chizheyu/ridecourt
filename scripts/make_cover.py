"""Generate docs/cover.png and docs/cover-380x216.png.

Composition (top to bottom):
  - Courthouse facade (pediment + columns) centered, with a gavel overlay
  - A thin separator band with the label "DISPUTE RESOLUTION SYSTEM"
  - The big title "RideCourt"
  - A tagline

The 16:9 cover is 1520x864. The 380x216 thumbnail is a downscaled copy.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DOCS = Path(__file__).resolve().parent.parent / "docs"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
NAVY = (26, 35, 50)           # deep background
NAVY_LIGHT = (40, 52, 74)
INK = (15, 23, 42)
GOLD = (212, 160, 74)          # courthouse + gavel accent
GOLD_LIGHT = (244, 196, 112)
GOLD_DARK = (150, 110, 40)
WHITE = (248, 249, 252)
SLATE = (107, 114, 128)
SAND = (232, 226, 210)
SHADOW = (10, 14, 22)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Avenir Next.ttc",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_courthouse_facade(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int) -> None:
    """Draw a courthouse (pediment + 6 columns + base) in the given rect."""
    w = x1 - x0
    h = y1 - y0
    cx = (x0 + x1) // 2

    # Base / steps
    draw.rectangle([x0, y1 - 24, x1, y1], fill=SHADOW)
    draw.rectangle([x0 + 6, y1 - 24, x1 - 6, y1 - 6], fill=NAVY_LIGHT)
    draw.rectangle([x0 + 18, y1 - 6, x1 - 18, y1], fill=INK)

    # Columns
    n_cols = 6
    col_w = max(14, w // 22)
    col_h = int(h * 0.55)
    col_top = y1 - 24 - col_h
    spacing = (w - 40 - n_cols * col_w) / (n_cols - 1)
    for i in range(n_cols):
        cx_col = int(x0 + 20 + i * (col_w + spacing))
        # shaft
        draw.rectangle([cx_col, col_top, cx_col + col_w, y1 - 24], fill=SAND)
        # capital
        draw.rectangle([cx_col - 4, col_top - 8, cx_col + col_w + 4, col_top], fill=GOLD_LIGHT)
        # base
        draw.rectangle([cx_col - 4, y1 - 32, cx_col + col_w + 4, y1 - 24], fill=GOLD_DARK)
        # fluting (thin vertical lines)
        for fx in (cx_col + 4, cx_col + col_w - 5):
            draw.line([(fx, col_top + 2), (fx, y1 - 28)], fill=(190, 184, 165), width=1)

    # Architrave (beam above columns)
    draw.rectangle([x0 + 6, col_top - 16, x1 - 6, col_top - 8], fill=GOLD)
    # Pediment (triangle)
    ped_h = int(h * 0.22)
    ped_top = col_top - 16 - ped_h
    draw.polygon(
        [(x0 + 6, col_top - 16), (cx, ped_top), (x1 - 6, col_top - 16)],
        fill=GOLD_DARK,
    )
    # Pediment inner trim
    draw.polygon(
        [(x0 + 22, col_top - 18), (cx, ped_top + 14), (x1 - 22, col_top - 18)],
        outline=GOLD_LIGHT,
        width=2,
    )
    # Small roundel in pediment
    r = max(6, ped_h // 6)
    draw.ellipse([cx - r, ped_top + ped_h // 2 - r, cx + r, ped_top + ped_h // 2 + r], fill=GOLD_LIGHT)


def _draw_gavel(draw: ImageDraw.ImageDraw, cx: int, cy: int, length: int, head_w: int, head_h: int) -> None:
    """Draw a gavel centered on (cx, cy), rotated ~25 degrees."""
    # We compose the gavel from a handle (rectangle) and a head (rectangle with caps),
    # drawn rotated around (cx, cy).
    angle = math.radians(28)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def rotate(px: float, py: float) -> tuple[int, int]:
        return int(cx + px * cos_a - py * sin_a), int(cy + px * sin_a + py * cos_a)

    # Handle: long thin rectangle from (0,0) to (length, 0), thickness ~ length/8
    t = max(6, length // 9)
    handle_poly = [
        (-length // 2, -t // 2),
        (length // 2, -t // 2),
        (length // 2, t // 2),
        (-length // 2, t // 2),
    ]
    draw.polygon([rotate(x, y) for x, y in handle_poly], fill=GOLD_LIGHT)

    # Head: centered near the far end of the handle
    head_cx = length // 2 - head_w // 2
    head_poly = [
        (head_cx - head_w // 2, -head_h // 2),
        (head_cx + head_w // 2, -head_h // 2),
        (head_cx + head_w // 2, head_h // 2),
        (head_cx - head_w // 2, head_h // 2),
    ]
    draw.polygon([rotate(x, y) for x, y in head_poly], fill=GOLD)
    # Head end caps (darker)
    cap_w = max(3, head_w // 6)
    cap_poly = [
        (head_cx + head_w // 2 - cap_w, -head_h // 2),
        (head_cx + head_w // 2, -head_h // 2),
        (head_cx + head_w // 2, head_h // 2),
        (head_cx + head_w // 2 - cap_w, head_h // 2),
    ]
    draw.polygon([rotate(x, y) for x, y in cap_poly], fill=GOLD_DARK)

    # Strike plate (a small rounded block under the head)
    plate_poly = [
        (head_cx - head_w // 2 - 6, head_h // 2),
        (head_cx + head_w // 2 + 6, head_h // 2),
        (head_cx + head_w // 2 + 4, head_h // 2 + 8),
        (head_cx - head_w // 2 - 4, head_h // 2 + 8),
    ]
    draw.polygon([rotate(x, y) for x, y in plate_poly], fill=GOLD_DARK)


def _text_center(draw: ImageDraw.ImageDraw, text: str, cx: int, y: int, font, fill) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    draw.text((cx - w // 2, y), text, font=font, fill=fill)
    return w


def render_cover(width: int = 1520, height: int = 864) -> Image.Image:
    img = Image.new("RGB", (width, height), NAVY)
    draw = ImageDraw.Draw(img)

    # Subtle vignette gradient at the top
    for i in range(height // 3):
        shade = NAVY[0] + (NAVY_LIGHT[0] - NAVY[0]) * i // (height // 3)
        draw.line([(0, i), (width, i)], fill=(shade, shade + 2, shade + 6))

    # ---- Courthouse facade (top ~55% of canvas) ----
    pad = width // 10
    cx = width // 2
    facade_top = int(height * 0.06)
    facade_bottom = int(height * 0.50)
    _draw_courthouse_facade(draw, pad, facade_top, width - pad, facade_bottom)

    # ---- Gavel, sitting on the courthouse base (above the label) ----
    gavel_cx = cx
    gavel_cy = facade_bottom - int(height * 0.12)
    _draw_gavel(
        draw,
        cx=gavel_cx,
        cy=gavel_cy,
        length=int(width * 0.20),
        head_w=int(width * 0.05),
        head_h=int(height * 0.06),
    )

    # ---- Label band ----
    band_y = int(height * 0.55)
    band_h = int(height * 0.06)
    draw.rectangle([0, band_y, width, band_y + band_h], fill=INK)
    # Thin gold rules above/below the band
    draw.rectangle([0, band_y, width, band_y + 2], fill=GOLD)
    draw.rectangle([0, band_y + band_h - 2, width, band_y + band_h], fill=GOLD)

    label_font = _font(max(10, int(height * 0.035)), bold=True)
    label = "DISPUTE RESOLUTION SYSTEM"
    # Render with letter spacing for a tracked, "official" look.
    spacing_px = max(2, int(height * 0.004))
    # Compute total width with tracking.
    char_widths = [draw.textbbox((0, 0), ch, font=label_font)[2] for ch in label]
    total_w = sum(char_widths) + spacing_px * (len(label) - 1)
    # Draw each char centered as a group.
    x = (width - total_w) // 2
    char_h = draw.textbbox((0, 0), label, font=label_font)[3]
    y_label = band_y + (band_h - char_h) // 2 - 2
    for i, ch in enumerate(label):
        draw.text((x, y_label), ch, font=label_font, fill=GOLD_LIGHT)
        x += char_widths[i] + spacing_px

    # ---- Title ----
    title_font = _font(max(20, int(height * 0.13)), bold=True)
    title_y = int(height * 0.64)
    _text_center(draw, "RideCourt", cx, title_y, title_font, WHITE)

    # ---- Tagline ----
    tag_font = _font(max(9, int(height * 0.028)))
    tag_y = int(height * 0.82)
    _text_center(
        draw,
        "Agents argue.  Code computes.  Fair rulings in seconds.",
        cx, tag_y, tag_font, SLATE,
    )

    # ---- Footer line ----
    foot_font = _font(max(8, int(height * 0.022)))
    _text_center(
        draw,
        "Tencent Cloud AI CAN DO IT Hackathon  ·  Singapore 2026  ·  Ryde",
        cx, int(height * 0.90), foot_font, (90, 98, 116),
    )

    return img


def main() -> None:
    big = render_cover(1520, 864)
    big.save(DOCS / "cover.png", optimize=True)

    # Thumbnail: re-render at the small size for crisp text, then also save.
    small = render_cover(380, 216)
    small.save(DOCS / "cover-380x216.png", optimize=True)

    print("Wrote docs/cover.png (1520x864) and docs/cover-380x216.png (380x216)")


if __name__ == "__main__":
    main()
