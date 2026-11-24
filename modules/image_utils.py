"""Image quality checks, bundled-font poster rendering, and local export helpers."""

from __future__ import annotations

import io
import csv
import math
import os
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFilter, ImageFont

import config
from modules.prompt_processor import PosterCopy, get_campaign_preset


# ------------------------------------------------------------------
# Conversions & File Helpers
# ------------------------------------------------------------------
def ensure_logo() -> Path:
    config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if not config.LOGO_PATH.exists():
        im = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.ellipse((4, 4, 124, 124), fill=config.COLORS["primary"])
        draw_health_cross(d, 38, 38, 52, "#FFFFFF")
        im.save(config.LOGO_PATH, "PNG")
    return config.LOGO_PATH


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def pil_to_pdf_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    rgb = image.convert("RGB")
    rgb.save(buffer, format="PDF", resolution=150.0)
    return buffer.getvalue()


def png_bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def detect_stray_text(image: Image.Image) -> Tuple[bool, float, str]:
    """Run OCR when available, otherwise use edge-density as a conservative fallback."""
    try:
        import pytesseract
        data = pytesseract.image_to_data(image.convert("RGB"), output_type=pytesseract.Output.DICT)
        recognized = [
            text.strip()
            for text, confidence in zip(data.get("text", []), data.get("conf", []))
            if text.strip() and float(confidence) >= 35
        ]
        return bool(recognized), float(len(recognized)), "OCR"
    except (ImportError, OSError, RuntimeError, ValueError):
        pass

    gray = image.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    histogram = edges.histogram()
    edge_pixels = sum(histogram[72:])
    density = edge_pixels / max(1, gray.width * gray.height)
    return density > 0.24, density, "edge-density fallback"


def assess_face_quality(image: Image.Image) -> Dict[str, object]:
    """Check that a person prompt has a sufficiently large, non-overlapping face."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {
            "available": False,
            "passed": False,
            "score": 0.0,
            "reason": "face detector unavailable",
        }

    detector_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(detector_path))
    if detector.empty():
        return {
            "available": False,
            "passed": False,
            "score": 0.0,
            "reason": "face detector data unavailable",
        }

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(max(24, int(image.width * 0.08)), 24)
    )
    face_boxes = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
    widths = [w / image.width for _, _, w, _ in face_boxes]
    large_faces = [box for box, ratio in zip(face_boxes, widths) if ratio >= 0.08]
    overlaps = False
    for index, (x1, y1, w1, h1) in enumerate(large_faces):
        for x2, y2, w2, h2 in large_faces[index + 1:]:
            intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(
                0, min(y1 + h1, y2 + h2) - max(y1, y2)
            )
            if intersection / max(1, min(w1 * h1, w2 * h2)) > 0.15:
                overlaps = True
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    largest_face = max(widths, default=0.0)
    score = largest_face + min(sharpness, 500.0) / 5000.0
    passed = bool(large_faces) and not overlaps
    reason = (
        f"passed ({len(large_faces)} face(s), largest {largest_face:.0%} image width)"
        if passed else "no sufficiently large face or faces overlap"
    )
    return {
        "available": True,
        "passed": passed,
        "score": score,
        "face_count": len(large_faces),
        "sharpness": sharpness,
        "reason": reason,
    }


def slugify(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text).strip().lower()
    return re.sub(r"[\s-]+", "-", text)[:max_len].strip("-") or "poster"


def make_filename(category: str, seed: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"wellaware_{slugify(category)}_{stamp}_{seed}.png"


def save_image(
    image: Image.Image,
    filename: str,
    output_dir: Optional[Path] = None,
) -> Path:
    target_dir = output_dir or config.OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / filename
    image.save(out_path, format="PNG", optimize=True)
    return out_path


def make_zip(entries: Iterable[dict]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        lines = ["WellAware AI - Campaign Poster Export Log", "=" * 45, ""]
        for idx, item in enumerate(entries, start=1):
            name = item.get("filename") or f"poster_{idx}.png"
            data = item.get("png_bytes")
            if data:
                zf.writestr(name, data)
                lines.append(f"#{idx}: {name}")
                lines.append(f"  Category: {item.get('category', 'Unknown')}")
                lines.append(f"  Headline: {item.get('poster_title', '')}")
                lines.append(f"  Prompt:   {item.get('user_prompt', '')}")
                lines.append(f"  Layout:   {item.get('layout', 'Editorial')}")
                lines.append("")
        zf.writestr("export_summary.txt", "\n".join(lines).encode("utf-8"))
    return buf.getvalue()


build_zip = make_zip


def append_audit_record(record: Dict[str, object]) -> None:
    """Append one completed generation record to the local audit CSV."""
    config.AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "timestamp", "category", "user_prompt", "final_prompt", "negative_prompt",
        "seed", "model", "text_gate_result", "face_gate_result",
    )
    write_header = not config.AUDIT_LOG_PATH.exists()
    with config.AUDIT_LOG_PATH.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def audit_log_bytes() -> bytes:
    if not config.AUDIT_LOG_PATH.exists():
        return b"timestamp,category,user_prompt,final_prompt,negative_prompt,seed,model,text_gate_result,face_gate_result\n"
    return config.AUDIT_LOG_PATH.read_bytes()


def pil_to_jpeg_bytes(image: Image.Image, quality: int = 92) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


# ------------------------------------------------------------------
# Bundled Font Loading
# ------------------------------------------------------------------
def _load_font(size: int, bold: bool = False, script: bool = False) -> ImageFont.ImageFont:
    """Load OFL-licensed workspace fonts; fail explicitly if the assets are missing."""
    filename = (
        "Poppins-700.ttf" if script else "Inter-700.ttf" if bold else "Inter-400.ttf"
    )
    path = config.ASSETS_DIR / "fonts" / filename
    try:
        return ImageFont.truetype(str(path), size)
    except OSError as exc:
        raise FileNotFoundError(
            "Bundled fonts are missing. Run `python scripts/download_fonts.py` before starting the app."
        ) from exc


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    lines = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            candidate_box = draw.textbbox((0, 0), candidate, font=font)
            if current and candidate_box[2] - candidate_box[0] > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines


def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_lines: int,
    max_size: int,
    min_size: int,
    bold: bool = False,
    script: bool = False,
) -> Tuple[ImageFont.ImageFont, List[str]]:
    for size in range(max_size, max(7, min_size) - 1, -1):
        font = _load_font(size, bold=bold, script=script)
        lines = _wrap_text(draw, text, font, max_width)
        if len(lines) <= max_lines or size == 7:
            return font, lines

    font = _load_font(7, bold=bold, script=script)
    lines = _wrap_text(draw, text, font, max_width)
    return font, lines


def _readable_text_color(background: str, preferred: Optional[str] = None) -> str:
    def luminance(color: str) -> float:
        channels = ImageColor.getrgb(color)[:3]
        linear = []
        for value in channels:
            channel = value / 255
            linear.append(channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4)
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    bg_luminance = luminance(background)

    def contrast(color: str) -> float:
        text_luminance = luminance(color)
        lighter, darker = max(bg_luminance, text_luminance), min(bg_luminance, text_luminance)
        return (lighter + 0.05) / (darker + 0.05)

    if preferred and contrast(preferred) >= 4.5:
        return preferred
    alternatives = ("#FFFFFF", "#0B1D30")
    return max(alternatives, key=contrast)


def _draw_wrapped_bullets(
    draw: ImageDraw.ImageDraw,
    bullet_points: List[str],
    x: int,
    y: int,
    text_width: int,
    fill: str,
    accent: str,
    category: str,
    font_size: int = 20,
    max_items: int = 3,
) -> int:
    for tip in bullet_points[:max_items]:
        font, lines = _fit_text(draw, tip, text_width, 2, font_size, max(14, font_size - 4))
        draw_theme_bullet_icon(draw, x, y + 3, 16, accent, category)
        line_height = max(22, int(font.size * 1.25))
        for line in lines:
            draw.text((x + 28, y), line, fill=fill, font=font)
            y += line_height
        y += 6
    return y


def _draw_quote_block(
    draw: ImageDraw.ImageDraw,
    quote: str,
    y: int,
    width: int,
    fill: str,
    max_size: int = 20,
) -> int:
    if not quote.strip():
        return y
    font, lines = _fit_text(draw, f'“{quote.strip()}”', width - 160, 2, max_size, 14, bold=True)
    for line in lines:
        line_width = draw.textlength(line, font=font)
        draw.text(((width - line_width) // 2, y), line, fill=fill, font=font)
        y += int(font.size * 1.25)
    return y


def _resize_artwork(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    scale = max(size[0] / image.width, size[1] / image.height)
    resized = image.resize(size, Image.Resampling.LANCZOS)
    if scale > 1.15:
        resized = resized.filter(ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=3))
    return resized


# ------------------------------------------------------------------
# Vector Graphic Helpers (Anti-aliased and Mathematical)
# ------------------------------------------------------------------
def draw_sunburst(draw: ImageDraw.ImageDraw, center: Tuple[int, int], radius: int,
                  color1: str, color2: str, num_rays: int = 32) -> None:
    """Draw radiating sunburst rays from center (like Reference 1)."""
    cx, cy = center
    step = 360.0 / num_rays
    for i in range(num_rays):
        if i % 2 == 0:
            continue
        a1 = math.radians(i * step)
        a2 = math.radians((i + 1) * step)
        x1 = cx + radius * math.cos(a1)
        y1 = cy + radius * math.sin(a1)
        x2 = cx + radius * math.cos(a2)
        y2 = cy + radius * math.sin(a2)
        draw.polygon([(cx, cy), (int(x1), int(y1)), (int(x2), int(y2))], fill=color2)


def draw_cloud(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, color: str = "#FFFFFF") -> None:
    """Draw friendly soft clouds (like Reference 1 & 2)."""
    r = h // 2
    draw.ellipse((x, y, x + 2 * r, y + 2 * r), fill=color)
    draw.ellipse((x + w - 2 * r, y, x + w, y + 2 * r), fill=color)
    draw.ellipse((x + r // 2, y - r // 2, x + w - r // 2, y + 3 * r // 2), fill=color)
    draw.rectangle((x + r, y + r // 2, x + w - r, y + 2 * r), fill=color)


def draw_health_cross(draw: ImageDraw.ImageDraw, x: int, y: int, size: int, color: str = "#E63946") -> None:
    """Draw a clean healthcare cross marker (like Reference 2)."""
    t = max(3, size // 3)
    c = x + size // 2
    m = y + size // 2
    draw.rectangle((c - t // 2, y, c + t // 2, y + size), fill=color)
    draw.rectangle((x, m - t // 2, x + size, m + t // 2), fill=color)


def draw_heart_icon(draw: ImageDraw.ImageDraw, x: int, y: int, size: int, color: str = "#F5A623") -> None:
    """Draw a small heart icon for health bullet points."""
    s = size / 24.0
    pts = [
        (12 * s, 21 * s), (5 * s, 14 * s), (3 * s, 10 * s), (3 * s, 6 * s),
        (6 * s, 3 * s), (10 * s, 3 * s), (12 * s, 6 * s), (14 * s, 3 * s),
        (18 * s, 3 * s), (21 * s, 6 * s), (21 * s, 10 * s), (19 * s, 14 * s),
    ]
    poly = [(int(x + px), int(y + py)) for px, py in pts]
    draw.polygon(poly, fill=color)


def draw_theme_bullet_icon(
    draw: ImageDraw.ImageDraw, x: int, y: int, size: int, color: str,
    category: str = "General Wellness",
) -> None:
    """Draw a category-tailored bullet icon so every campaign has distinctive, relevant markers."""
    cat_low = (category or "").lower()
    cx, cy = x + size // 2, y + size // 2
    r = size // 2

    if "heart" in cat_low or "cardio" in cat_low or "lung" in cat_low:
        # Heart icon
        draw_heart_icon(draw, x, y, size, color)
    elif "blood" in cat_low or "donor" in cat_low:
        # Sleek teardrop / blood drop bullet
        draw.polygon([(cx, y), (cx + r, cy + 2), (cx, y + size), (cx - r, cy + 2)], fill=color)
        draw.ellipse((cx - r, cy, cx + r, y + size), fill=color)
    elif "cancer" in cat_low:
        # Awareness ribbon / solidarity loop bullet
        lw = max(2, size // 5)
        draw.arc((cx - r // 2, y, cx + r // 2, cy), 180, 0, fill=color, width=lw)
        draw.line((cx - r // 2, cy - 2, cx + r // 2, y + size), fill=color, width=lw)
        draw.line((cx + r // 2, cy - 2, cx - r // 2, y + size), fill=color, width=lw)
    elif "eat" in cat_low or "food" in cat_low or "nutrition" in cat_low:
        # Organic fresh leaf bullet
        draw.pieslice((cx - r, cy - r, cx + r, cy + r), 0, 90, fill=color)
        draw.pieslice((cx - r, cy - r, cx + r, cy + r), 180, 270, fill=color)
        draw.line((x + 2, y + size - 2, x + size - 2, y + 2), fill="#FFFFFF", width=1)
    elif "diabet" in cat_low:
        # Official blue circle / health ring bullet
        lw = max(2, size // 4)
        draw.ellipse((x, y, x + size, y + size), outline=color, width=lw)
    elif "fitness" in cat_low or "sport" in cat_low or "move" in cat_low:
        # Active energy chevron / forward arrow
        draw.polygon([
            (x, y + 2), (cx, cy), (x, y + size - 2),
            (x + r // 2, y + size - 2), (x + size, cy), (x + r // 2, y + 2)
        ], fill=color)
    elif "women" in cat_low:
        # 4-petal blooming flower bullet
        fr = max(2, size // 3)
        draw.ellipse((cx - fr // 2, y, cx + fr // 2, y + fr), fill=color)
        draw.ellipse((cx - fr // 2, y + size - fr, cx + fr // 2, y + size), fill=color)
        draw.ellipse((x, cy - fr // 2, x + fr, cy + fr // 2), fill=color)
        draw.ellipse((x + size - fr, cy - fr // 2, x + size, cy + fr // 2), fill=color)
        draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill="#FEF08A")
    elif "mental" in cat_low or "mind" in cat_low or "brain" in cat_low:
        # Spark / neuro-star node bullet
        pts = [
            (cx, y), (cx + r // 3, cy - r // 3), (cx + r, cy),
            (cx + r // 3, cy + r // 3), (cx, y + size), (cx - r // 3, cy + r // 3),
            (cx - r, cy), (cx - r // 3, cy - r // 3)
        ]
        draw.polygon(pts, fill=color)
    elif "world" in cat_low:
        # Healthcare plus marker
        t = max(2, size // 3)
        draw.rectangle((cx - t // 2, y, cx + t // 2, y + size), fill=color)
        draw.rectangle((x, cy - t // 2, x + size, cy + t // 2), fill=color)
    else:
        # Clean geometric wellness diamond
        draw.polygon([(cx, y), (x + size, cy), (cx, y + size), (x, cy)], fill=color)


# ------------------------------------------------------------------
# Dynamic Color Extraction from User Prompt
# ------------------------------------------------------------------
PROMPT_COLOR_MAP: Dict[str, Tuple[str, str, str, str]] = {
    "blue": ("#1D4ED8", "#60A5FA", "#EFF6FF", "#1E3A8A"),
    "navy": ("#0F172A", "#38BDF8", "#F8FAFC", "#0F172A"),
    "cyan": ("#0E7490", "#22D3EE", "#ECFEFF", "#164E63"),
    "teal": ("#0F766E", "#14B8A6", "#F0FDFA", "#115E59"),
    "green": ("#15803D", "#22C55E", "#F0FDF4", "#14532D"),
    "emerald": ("#047857", "#34D399", "#ECFDF5", "#064E3B"),
    "lime": ("#4D7C0F", "#84CC16", "#F7FEE7", "#365314"),
    "yellow": ("#B45309", "#F59E0B", "#FFFBEB", "#78350F"),
    "gold": ("#B45309", "#FBBF24", "#FEFCE8", "#78350F"),
    "amber": ("#B45309", "#F59E0B", "#FFFBEB", "#78350F"),
    "orange": ("#C2410C", "#FB923C", "#FFF7ED", "#7C2D12"),
    "red": ("#B91C1C", "#EF4444", "#FEF2F2", "#7F1D1D"),
    "crimson": ("#991B1B", "#F87171", "#FEF2F2", "#7F1D1D"),
    "pink": ("#BE185D", "#F472B6", "#FDF2F8", "#831843"),
    "rose": ("#BE123C", "#FB7185", "#FFF1F2", "#881337"),
    "purple": ("#6B21A8", "#A855F7", "#FAF5FF", "#3B0764"),
    "violet": ("#5B21B6", "#8B5CF6", "#F5F3FF", "#4C1D95"),
    "lavender": ("#6B21A8", "#C084FC", "#FAF5FF", "#4C1D95"),
}


def extract_palette_from_prompt(
    prompt: str,
    base_palette: Dict[str, str],
    theme_choice: Optional[str] = None,
    custom_palette: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Dynamically adapt poster color palette from custom user colors, theme selection, or prompt keywords."""
    if custom_palette and isinstance(custom_palette, dict) and "primary" in custom_palette:
        return dict(custom_palette)

    if theme_choice and theme_choice in getattr(config, "THEME_PALETTES", {}) and config.THEME_PALETTES[theme_choice]:
        return dict(config.THEME_PALETTES[theme_choice])

    low = (prompt or "").lower()
    found = []
    for c in PROMPT_COLOR_MAP:
        if re.search(r"\b" + c + r"\b", low):
            found.append(c)
    if not found:
        return dict(base_palette)
    p = dict(base_palette)
    primary_c = PROMPT_COLOR_MAP[found[0]]
    p["primary"] = primary_c[0]
    p["bg_light"] = primary_c[2]
    p["bg_dark"] = primary_c[3]
    if len(found) > 1:
        accent_c = PROMPT_COLOR_MAP[found[1]]
        p["accent"] = accent_c[1]
    else:
        p["accent"] = primary_c[1]
    return p


def draw_people_and_scene_art(draw: ImageDraw.ImageDraw, cx: int, cy: int,
                              palette: Dict[str, str], prompt: str = "") -> None:
    """Draw vibrant healthcare heroes & diverse community figures when user requests people."""
    # Vitality backdrop
    draw.ellipse((cx - 180, cy - 180, cx + 180, cy + 180), fill=palette.get("bg_light", "#F0FDFA"))
    draw.ellipse((cx - 150, cy - 150, cx + 150, cy + 150), outline=palette.get("accent", "#14B8A6"), width=4)

    # Sunburst / energy aura
    for angle in range(0, 360, 30):
        rad = math.radians(angle)
        x1 = cx + 140 * math.cos(rad)
        y1 = cy + 140 * math.sin(rad)
        x2 = cx + 175 * math.cos(rad)
        y2 = cy + 175 * math.sin(rad)
        draw.line([(int(x1), int(y1)), (int(x2), int(y2))], fill=palette.get("accent", "#F5A623"), width=4)

    # Center Figure (Doctor / Healthcare Professional in Scrubs)
    draw.ellipse((cx - 30, cy - 150, cx + 30, cy - 90), fill="#2D3748")
    draw.rounded_rectangle((cx - 20, cy - 115, cx + 20, cy - 95), radius=4, fill="#FFFFFF")
    draw.polygon([(cx - 45, cy - 85), (cx + 45, cy - 85), (cx + 55, cy + 90), (cx - 55, cy + 90)],
                 fill=palette.get("primary", "#0F766E"))
    # Stethoscope
    draw.arc((cx - 28, cy - 80, cx + 28, cy - 20), 0, 180, fill="#E2E8F0", width=4)
    draw.ellipse((cx + 12, cy - 22, cx + 24, cy - 10), fill="#94A3B8")

    # Left Figure (Nurse / Patient / Partner)
    draw.ellipse((cx - 130, cy - 120, cx - 80, cy - 70), fill="#4A5568")
    draw.polygon([(cx - 140, cy - 65), (cx - 70, cy - 65), (cx - 60, cy + 100), (cx - 150, cy + 100)],
                 fill=palette.get("accent", "#3B82F6"))

    # Right Figure (Community Member / Child / Elder)
    draw.ellipse((cx + 80, cy - 120, cx + 130, cy - 70), fill="#334155")
    draw.polygon([(cx + 70, cy - 65), (cx + 140, cy - 65), (cx + 150, cy + 100), (cx + 60, cy + 100)],
                 fill=palette.get("accent2", "#F59E0B"))

    # Connecting / supportive arms
    draw.line([(cx - 70, cy - 20), (cx - 45, cy - 10)], fill="#E2E8F0", width=8)
    draw.line([(cx + 45, cy - 10), (cx + 70, cy - 20)], fill="#E2E8F0", width=8)

    # Healthcare cross badge in center
    draw_health_cross(draw, cx - 18, cy + 120, 36, "#EF4444")


def draw_social_icons(draw: ImageDraw.ImageDraw, x: int, y: int, size: int, color: str) -> None:
    """Draw miniature modern social icons (IG, FB, Web)."""
    # Instagram icon
    draw.rounded_rectangle((x, y, x + size, y + size), radius=size // 4, outline=color, width=2)
    draw.ellipse((x + size // 4, y + size // 4, x + 3 * size // 4, y + 3 * size // 4), outline=color, width=2)
    draw.ellipse((x + size * 3 // 4 - 2, y + size // 4, x + size * 3 // 4 + 2, y + size // 4 + 4), fill=color)

    # Facebook 'f'
    fx = x + size + 16
    draw.rounded_rectangle((fx, y, fx + size, y + size), radius=size // 4, outline=color, width=2)
    draw.rectangle((fx + size // 2, y + 4, fx + size // 2 + 3, y + size - 4), fill=color)
    draw.rectangle((fx + size // 3, y + size // 3, fx + size * 2 // 3 + 2, y + size // 3 + 3), fill=color)

    # Web globe
    gx = fx + size + 16
    draw.ellipse((gx, y, gx + size, y + size), outline=color, width=2)
    draw.ellipse((gx + size // 4, y, gx + 3 * size // 4, y + size), outline=color, width=1)
    draw.line((gx, y + size // 2, gx + size, y + size // 2), fill=color, width=1)


# ------------------------------------------------------------------
# Procedural Vector Art Generator (Distinctive High-Quality Artwork for all 10 Campaigns)
# ------------------------------------------------------------------
def draw_heart_and_lungs_art(draw: ImageDraw.ImageDraw, cx: int, cy: int, palette: Dict[str, str]) -> None:
    """Cardiovascular & Respiratory Health: anatomical lungs and heart with ECG rhythm."""
    # Trachea
    draw.rectangle((cx - 14, cy - 180, cx + 14, cy - 80), fill="#D9534F")
    for ry in range(cy - 170, cy - 80, 15):
        draw.line((cx - 12, ry, cx + 12, ry), fill="#FFFFFF", width=2)
    draw.polygon([(cx - 14, cy - 80), (cx - 50, cy - 30), (cx - 30, cy - 20), (cx, cy - 65)], fill="#D9534F")
    draw.polygon([(cx + 14, cy - 80), (cx + 50, cy - 30), (cx + 30, cy - 20), (cx, cy - 65)], fill="#D9534F")
    # Left lung
    draw.polygon([
        (cx - 30, cy - 50), (cx - 150, cy - 10), (cx - 190, cy + 90),
        (cx - 160, cy + 180), (cx - 60, cy + 170), (cx - 30, cy + 70)
    ], fill="#E63946")
    # Right lung
    draw.polygon([
        (cx + 30, cy - 50), (cx + 150, cy - 10), (cx + 190, cy + 90),
        (cx + 160, cy + 180), (cx + 60, cy + 170), (cx + 30, cy + 70)
    ], fill="#E63946")
    # Lung bronchial tree lines
    for offset in [-1, 1]:
        bx = cx + offset * 90
        draw.line((bx, cy - 10, bx - offset * 40, cy + 80), fill="#FF8FA3", width=6)
        draw.line((bx, cy + 30, bx + offset * 40, cy + 120), fill="#FF8FA3", width=4)
    # Center heart
    draw.polygon([
        (cx, cy + 105), (cx - 60, cy + 20), (cx - 50, cy - 40),
        (cx, cy - 10), (cx + 50, cy - 40), (cx + 60, cy + 20)
    ], fill="#9B1D20")
    # Aorta arch
    draw.arc((cx - 35, cy - 65, cx + 15, cy - 15), start=180, end=0, fill="#2B3A67", width=16)
    # Vitality ECG pulse wave
    ecg = [
        (cx - 110, cy + 35), (cx - 60, cy + 35), (cx - 45, cy + 15),
        (cx - 30, cy + 60), (cx - 15, cy - 20), (cx, cy + 55),
        (cx + 15, cy + 35), (cx + 110, cy + 35)
    ]
    draw.line(ecg, fill="#FFD166", width=4)


def draw_mental_health_art(draw: ImageDraw.ImageDraw, cx: int, cy: int, palette: Dict[str, str]) -> None:
    """Mental Health: Profile silhouette with vibrant brain tree and mindfulness aura."""
    # Mindfulness aura halo
    draw.ellipse((cx - 180, cy - 180, cx + 180, cy + 180), fill="#F0F9FF")
    draw.ellipse((cx - 150, cy - 150, cx + 150, cy + 150), outline="#BAE6FD", width=3)
    # Head silhouette in deep navy
    head_pts = [
        (cx - 120, cy + 210), (cx - 120, cy + 90), (cx - 140, cy + 10),
        (cx - 130, cy - 80), (cx - 90, cy - 150), (cx - 10, cy - 180),
        (cx + 60, cy - 160), (cx + 110, cy - 90), (cx + 120, cy - 20),
        (cx + 100, cy + 10), (cx + 140, cy + 30), (cx + 110, cy + 60),
        (cx + 90, cy + 100), (cx + 120, cy + 130), (cx + 90, cy + 160),
        (cx + 40, cy + 170), (cx + 50, cy + 210)
    ]
    draw.polygon(head_pts, fill="#0D253F")
    # Tree trunk (brain stem)
    draw.rectangle((cx - 16, cy - 50, cx + 16, cy + 120), fill="#0D253F")
    # Foliage / neuro-leaves in vibrant autumn & teal colors
    leaf_colors = ["#F5A623", "#E63946", "#2EB872", "#457B9D", "#D4A373", "#F4A261", "#E76F51", "#2A9D8F"]
    for angle in range(0, 360, 20):
        rad = math.radians(angle)
        dist = 80 + (angle % 60) * 1.6
        lx = cx + dist * math.cos(rad) * 0.95
        ly = cy - 70 + dist * math.sin(rad) * 0.8
        color = leaf_colors[(angle // 20) % len(leaf_colors)]
        draw.ellipse((int(lx - 20), int(ly - 32), int(lx + 20), int(ly + 32)), fill=color)


def draw_blood_donation_art(draw: ImageDraw.ImageDraw, cx: int, cy: int, palette: Dict[str, str]) -> None:
    """Blood Donation: Modern layered teardrop emblem with healthcare cross and caring cradle."""
    # Ripple rings
    draw.ellipse((cx - 180, cy - 180, cx + 180, cy + 180), fill="#FFF5F2")
    draw.ellipse((cx - 150, cy - 150, cx + 150, cy + 150), outline="#FECDD3", width=3)
    # Caring hands contour cradling the drop
    draw.arc((cx - 150, cy - 20, cx + 150, cy + 190), 20, 160, fill="#FDA4AF", width=12)
    # Outer blood drop
    drop_pts = [
        (cx, cy - 170), (cx + 110, cy - 30), (cx + 110, cy + 70),
        (cx + 60, cy + 150), (cx, cy + 170), (cx - 60, cy + 150),
        (cx - 110, cy + 70), (cx - 110, cy - 30)
    ]
    draw.polygon(drop_pts, fill="#E63946")
    # Gradient contour line
    draw.polygon([
        (cx, cy - 130), (cx + 80, cy - 10), (cx + 80, cy + 60),
        (cx + 40, cy + 120), (cx, cy + 135), (cx - 40, cy + 120),
        (cx - 80, cy + 60), (cx - 80, cy - 10)
    ], fill="#FFF5F2")
    # Inner vibrant drop
    draw.polygon([
        (cx, cy - 90), (cx + 50, cy), (cx + 50, cy + 50),
        (cx, cy + 100), (cx - 50, cy + 50), (cx - 50, cy)
    ], fill="#E63946")
    # White healthcare cross centered in the drop
    draw.rectangle((cx - 8, cy - 10, cx + 8, cy + 50), fill="#FFFFFF")
    draw.rectangle((cx - 28, cy + 10, cx + 28, cy + 30), fill="#FFFFFF")


def draw_cancer_awareness_art(draw: ImageDraw.ImageDraw, cx: int, cy: int, palette: Dict[str, str]) -> None:
    """Cancer Awareness: Elegant awareness ribbon intertwined with blooming botanical hope flowers."""
    # Hope & solidarity background halo
    draw.ellipse((cx - 180, cy - 180, cx + 180, cy + 180), fill="#FAF5FF")
    draw.ellipse((cx - 150, cy - 150, cx + 150, cy + 150), outline="#E9D5FF", width=3)
    # Blooming botanical petals clustered around the ribbon
    petal_colors = ["#F43F5E", "#FB7185", "#FDA4AF", "#F472B6", "#C084FC"]
    for angle in range(0, 360, 45):
        rad = math.radians(angle)
        px = cx + 120 * math.cos(rad)
        py = cy + 120 * math.sin(rad)
        draw.ellipse((int(px - 22), int(py - 22), int(px + 22), int(py + 22)),
                     fill=petal_colors[(angle // 45) % len(petal_colors)])
    # Organic green leaves
    leaf_pts = [
        [(cx - 130, cy + 80), (cx - 180, cy + 50), (cx - 150, cy + 120)],
        [(cx + 130, cy + 80), (cx + 180, cy + 50), (cx + 150, cy + 120)],
        [(cx - 100, cy - 120), (cx - 140, cy - 150), (cx - 80, cy - 160)],
        [(cx + 100, cy - 120), (cx + 140, cy - 150), (cx + 80, cy - 160)]
    ]
    for pts in leaf_pts:
        draw.polygon(pts, fill="#10B981")
    # Main Awareness Ribbon (Plum / Lavender & Hope Pink)
    # Right ribbon tail (underneath)
    draw.polygon([
        (cx, cy - 10), (cx + 80, cy + 130), (cx + 110, cy + 190),
        (cx + 60, cy + 180), (cx + 40, cy + 140), (cx - 10, cy + 40)
    ], fill="#701A75")
    # Left ribbon tail (over)
    draw.polygon([
        (cx - 10, cy - 10), (cx - 80, cy + 130), (cx - 110, cy + 190),
        (cx - 60, cy + 180), (cx - 40, cy + 140), (cx + 10, cy + 40)
    ], fill="#9333EA")
    # Upper ribbon loop
    draw.ellipse((cx - 80, cy - 160, cx + 80, cy + 10), fill="#A855F7")
    # Inner loop cutout
    draw.ellipse((cx - 42, cy - 125, cx + 42, cy - 25), fill="#FAF5FF")
    # Highlight contour in hope pink
    draw.arc((cx - 75, cy - 155, cx + 75, cy + 5), start=180, end=360, fill="#F43F5E", width=10)
    # Hope sparkle stars
    for sx, sy in [(cx - 120, cy - 50), (cx + 120, cy - 50), (cx, cy - 180), (cx, cy + 130)]:
        pts = [
            (sx, sy - 14), (sx + 4, sy - 4), (sx + 14, sy), (sx + 4, sy + 4),
            (sx, sy + 14), (sx - 4, sy + 4), (sx - 14, sy), (sx - 4, sy - 4)
        ]
        draw.polygon(pts, fill="#F59E0B")


def draw_healthy_eating_art(draw: ImageDraw.ImageDraw, cx: int, cy: int, palette: Dict[str, str]) -> None:
    """Healthy Eating: Fresh salad bowl overflowing with avocado, tomato, citrus, and leafy greens."""
    # Nutrition halo
    draw.ellipse((cx - 180, cy - 180, cx + 180, cy + 180), fill="#F7FEE7")
    # Spinach / leafy greens sprouting behind bowl
    leaves = [
        [(cx - 80, cy - 30), (cx - 145, cy - 120), (cx - 40, cy - 90)],
        [(cx + 80, cy - 30), (cx + 145, cy - 120), (cx + 40, cy - 90)],
        [(cx - 20, cy - 40), (cx, cy - 150), (cx + 30, cy - 80)]
    ]
    for pts in leaves:
        draw.polygon(pts, fill="#15803D")
        draw.line((pts[0][0], pts[0][1], pts[1][0], pts[1][1]), fill="#86EFAC", width=3)
    # Avocado half on left
    draw.ellipse((cx - 130, cy - 80, cx - 10, cy + 50), fill="#166534")
    draw.ellipse((cx - 122, cy - 72, cx - 18, cy + 42), fill="#BEF264")
    draw.ellipse((cx - 85, cy - 35, cx - 45, cy + 10), fill="#78350F")
    # Fresh tomato slice on right
    draw.ellipse((cx + 15, cy - 85, cx + 125, cy + 25), fill="#DC2626")
    draw.ellipse((cx + 25, cy - 75, cx + 115, cy + 15), fill="#EF4444")
    for rot in range(0, 360, 60):
        rad = math.radians(rot)
        tx = cx + 70 + 26 * math.cos(rad)
        ty = cy - 30 + 26 * math.sin(rad)
        draw.ellipse((int(tx - 6), int(ty - 6), int(tx + 6), int(ty + 6)), fill="#FEE2E2")
        draw.ellipse((int(tx - 3), int(ty - 3), int(tx + 3), int(ty + 3)), fill="#F59E0B")
    # Citrus / orange wedge
    draw.pieslice((cx - 50, cy - 120, cx + 50, cy - 20), 45, 135, fill="#EA580C")
    draw.pieslice((cx - 42, cy - 114, cx + 42, cy - 26), 50, 130, fill="#FDBA74")
    # Modern ceramic salad bowl
    draw.pieslice((cx - 165, cy - 20, cx + 165, cy + 190), 0, 180, fill="#0F766E")
    draw.ellipse((cx - 170, cy - 35, cx + 170, cy + 10), fill="#134E4A")
    draw.ellipse((cx - 160, cy - 28, cx + 160, cy + 2), fill="#CCFBF1")
    # Fork accent
    draw.line((cx + 120, cy + 30, cx + 155, cy + 150), fill="#94A3B8", width=8)


def draw_diabetes_awareness_art(draw: ImageDraw.ImageDraw, cx: int, cy: int, palette: Dict[str, str]) -> None:
    """Diabetes Awareness: Iconic World Diabetes Day blue circle, balanced red apple & glucose check."""
    # Iconic World Diabetes Day Blue Circle
    draw.ellipse((cx - 170, cy - 170, cx + 170, cy + 170), fill="#F0F9FF")
    draw.ellipse((cx - 150, cy - 150, cx + 150, cy + 150), outline="#0284C7", width=26)
    # Balanced Nutrition Apple inside circle
    draw.ellipse((cx - 80, cy - 60, cx + 10, cy + 70), fill="#DC2626")
    draw.ellipse((cx - 10, cy - 60, cx + 80, cy + 70), fill="#DC2626")
    draw.ellipse((cx - 50, cy - 68, cx + 50, cy + 50), fill="#B91C1C")
    # Apple leaf & stem
    draw.line((cx, cy - 65, cx - 12, cy - 95), fill="#78350F", width=6)
    draw.ellipse((cx - 8, cy - 105, cx + 38, cy - 75), fill="#16A34A")
    # Glucose balance droplet check in azure
    draw.polygon([(cx, cy - 10), (cx + 36, cy + 40), (cx - 36, cy + 40)], fill="#38BDF8")
    draw.ellipse((cx - 36, cy + 10, cx + 36, cy + 75), fill="#38BDF8")
    # White checkmark in droplet
    draw.line((cx - 16, cy + 42, cx - 4, cy + 54), fill="#FFFFFF", width=5)
    draw.line((cx - 4, cy + 54, cx + 18, cy + 32), fill="#FFFFFF", width=5)


def draw_physical_fitness_art(draw: ImageDraw.ImageDraw, cx: int, cy: int, palette: Dict[str, str]) -> None:
    """Physical Fitness: Dynamic athletic runner silhouette with rising sunburst and speed arcs."""
    # Rising sunrise vitality backdrop
    draw.pieslice((cx - 180, cy - 90, cx + 180, cy + 200), 180, 360, fill="#FFFBEB")
    draw.arc((cx - 160, cy - 70, cx + 160, cy + 190), 180, 360, fill="#F59E0B", width=8)
    # Aerodynamic motion trails
    for offset, col in [(-60, "#F97316"), (-20, "#06B6D4"), (20, "#14B8A6"), (60, "#F97316")]:
        draw.arc((cx - 190, cy - 120 + offset, cx + 90, cy + 140 + offset), 150, 240, fill=col, width=8)
    # Athletic runner silhouette
    # Head
    draw.ellipse((cx + 10, cy - 140, cx + 55, cy - 95), fill="#0F766E")
    # Forward-leaning torso
    draw.polygon([(cx + 20, cy - 95), (cx + 45, cy - 90), (cx - 10, cy + 10), (cx - 35, cy)], fill="#0F766E")
    # Forward lead leg (bent knee)
    draw.line([(cx - 10, cy + 5), (cx + 60, cy + 30)], fill="#0F766E", width=18)
    draw.line([(cx + 60, cy + 30), (cx + 50, cy + 120)], fill="#0F766E", width=15)
    # Rear trailing leg
    draw.line([(cx - 30, cy), (cx - 90, cy + 45)], fill="#0F766E", width=18)
    draw.line([(cx - 90, cy + 45), (cx - 150, cy + 85)], fill="#0F766E", width=14)
    # Arms pumping
    draw.line([(cx + 30, cy - 70), (cx + 80, cy - 40)], fill="#0F766E", width=14)
    draw.line([(cx + 80, cy - 40), (cx + 70, cy - 10)], fill="#0F766E", width=12)
    draw.line([(cx + 10, cy - 65), (cx - 45, cy - 40)], fill="#0F766E", width=14)


def draw_womens_health_art(draw: ImageDraw.ImageDraw, cx: int, cy: int, palette: Dict[str, str]) -> None:
    """Women's Health Awareness: Elegant feminine profile silhouette with blooming floral botanical crown."""
    # Soft rose nurture halo
    draw.ellipse((cx - 180, cy - 180, cx + 180, cy + 180), fill="#FDF2F8")
    draw.ellipse((cx - 150, cy - 150, cx + 150, cy + 150), outline="#FBCFE8", width=4)
    # Female profile silhouette
    head_pts = [
        (cx - 50, cy + 180), (cx - 50, cy + 60), (cx - 80, cy + 10),
        (cx - 85, cy - 60), (cx - 55, cy - 130), (cx + 10, cy - 150),
        (cx + 65, cy - 120), (cx + 80, cy - 80), (cx + 65, cy - 50),
        (cx + 85, cy - 20), (cx + 60, cy + 10), (cx + 85, cy + 40),
        (cx + 60, cy + 70), (cx + 30, cy + 110), (cx + 40, cy + 180)
    ]
    draw.polygon(head_pts, fill="#831843")
    # Botanical flower crown on hair
    flower_colors = ["#F472B6", "#FDA4AF", "#FB7185", "#F43F5E", "#F9A8D4"]
    for angle in range(120, 260, 25):
        rad = math.radians(angle)
        fx = cx + 80 * math.cos(rad)
        fy = cy - 70 + 80 * math.sin(rad)
        draw.ellipse((int(fx - 18), int(fy - 18), int(fx + 18), int(fy + 18)),
                     fill=flower_colors[(angle // 25) % len(flower_colors)])
        draw.ellipse((int(fx - 6), int(fy - 6), int(fx + 6), int(fy + 6)), fill="#FEF08A")
    # Soft green leaves intertwined in crown
    for lx, ly in [(cx - 90, cy - 110), (cx - 30, cy - 150), (cx + 40, cy - 130)]:
        draw.ellipse((lx - 12, ly - 22, lx + 12, ly + 22), fill="#059669")
    # Women's awareness ribbon accent in pink
    draw.arc((cx + 80, cy + 60, cx + 130, cy + 130), 180, 0, fill="#DB2777", width=8)
    draw.line((cx + 85, cy + 95, cx + 135, cy + 160), fill="#DB2777", width=8)
    draw.line((cx + 125, cy + 95, cx + 75, cy + 160), fill="#DB2777", width=8)


def draw_world_health_art(draw: ImageDraw.ImageDraw, cx: int, cy: int, palette: Dict[str, str]) -> None:
    """World Health Day: Stylized earth globe with diverse community figures and medical crosses."""
    # Earth globe
    draw.ellipse((cx - 150, cy - 150, cx + 150, cy + 150), fill="#A7D7C5")
    draw.ellipse((cx - 100, cy - 110, cx + 60, cy + 70), fill="#52B788")
    draw.ellipse((cx + 10, cy - 60, cx + 120, cy + 100), fill="#52B788")
    # Latitude / longitude equator arc
    draw.arc((cx - 150, cy - 60, cx + 150, cy + 60), 0, 180, fill="#38BDF8", width=4)
    # Figures around globe
    draw.ellipse((cx - 160, cy - 40, cx - 130, cy - 10), fill="#2D3748")  # Head left
    draw.line((cx - 145, cy - 10, cx - 145, cy + 60), fill="#2D3748", width=12)
    draw.ellipse((cx + 130, cy - 40, cx + 160, cy - 10), fill="#2D3748")  # Head right
    draw.line((cx + 145, cy - 10, cx + 145, cy + 60), fill="#2D3748", width=12)
    # Red health crosses
    draw_health_cross(draw, cx - 180, cy - 120, 36, "#E63946")
    draw_health_cross(draw, cx + 150, cy - 120, 36, "#E63946")


def draw_general_wellness_art(draw: ImageDraw.ImageDraw, cx: int, cy: int, palette: Dict[str, str]) -> None:
    """General Wellness: Peaceful meditating/stretching figure with dual flourishing leaves and sun rays."""
    # Radiant sunlight rays
    for angle in range(0, 360, 30):
        rad = math.radians(angle)
        x1 = cx + 100 * math.cos(rad)
        y1 = cy + 100 * math.sin(rad)
        x2 = cx + 175 * math.cos(rad)
        y2 = cy + 175 * math.sin(rad)
        draw.line((int(x1), int(y1), int(x2), int(y2)), fill="#FEF08A", width=5)
    # Calming zen circle
    draw.ellipse((cx - 150, cy - 150, cx + 150, cy + 150), fill="#F0FDFA")
    draw.ellipse((cx - 130, cy - 130, cx + 130, cy + 130), outline="#14B8A6", width=6)
    # Dual flourishing leaves arching like wings
    draw.pieslice((cx - 140, cy - 90, cx + 10, cy + 70), 90, 180, fill="#10B981")
    draw.pieslice((cx - 140, cy - 90, cx + 10, cy + 70), 270, 360, fill="#34D399")
    draw.line((cx - 130, cy - 80, cx, cy + 60), fill="#FFFFFF", width=3)
    draw.pieslice((cx - 10, cy - 90, cx + 140, cy + 70), 0, 90, fill="#34D399")
    draw.pieslice((cx - 10, cy - 90, cx + 140, cy + 70), 180, 270, fill="#10B981")
    draw.line((cx + 130, cy - 80, cx, cy + 60), fill="#FFFFFF", width=3)
    # Meditating human figure in center
    draw.ellipse((cx - 24, cy - 80, cx + 24, cy - 32), fill="#0F766E")
    draw.polygon([(cx - 18, cy - 30), (cx + 18, cy - 30), (cx + 28, cy + 60), (cx - 28, cy + 60)], fill="#0F766E")
    # Outstretched peaceful arms
    draw.arc((cx - 70, cy - 35, cx + 70, cy + 35), 0, 180, fill="#0F766E", width=12)


def draw_gradient_fill(draw: ImageDraw.ImageDraw, width: int, height: int,
                       color_top: str, color_bottom: str) -> None:
    """Draw a smooth 2-stop vertical gradient background."""
    def parse_hex(c_str):
        c_str = (c_str or "#0F172A").lstrip("#")
        if len(c_str) == 6:
            try:
                return tuple(int(c_str[i:i+2], 16) for i in (0, 2, 4))
            except ValueError:
                pass
        return (15, 23, 42)

    r1, g1, b1 = parse_hex(color_top)
    r2, g2, b2 = parse_hex(color_bottom)

    for y in range(height):
        factor = y / max(1, height - 1)
        r = int(r1 + (r2 - r1) * factor)
        g = int(g1 + (g2 - g1) * factor)
        b = int(b1 + (b2 - b1) * factor)
        draw.line([(0, y), (width, y)], fill=(r, g, b))


def draw_ambient_elements(draw: ImageDraw.ImageDraw, width: int, height: int,
                          category: str, palette: Dict[str, str], prompt_low: str) -> None:
    """Draw atmospheric ambient lighting, rays, ripples, and decorative accents."""
    cx, cy = width // 2, height // 2
    accent = palette.get("accent", "#38BDF8")
    accent2 = palette.get("accent2", "#F5A623")
    cat_low = (category or "").lower()

    # Soft glowing radial aura in the center
    for r in range(240, 140, -30):
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=palette.get("primary", "#1E3A8A"), width=2)

    if "cancer" in cat_low or "women" in cat_low:
        # Floating starlight sparkles & soft floral aura
        for angle in range(15, 360, 45):
            rad = math.radians(angle)
            sx = int(cx + 220 * math.cos(rad))
            sy = int(cy + 220 * math.sin(rad))
            draw.line([(sx - 8, sy), (sx + 8, sy)], fill="#FFFFFF", width=2)
            draw.line([(sx, sy - 8), (sx, sy + 8)], fill="#FFFFFF", width=2)
    elif "heart" in cat_low or "fitness" in cat_low or "run" in prompt_low:
        # Dynamic radiating sunburst rays
        for angle in range(0, 360, 30):
            rad = math.radians(angle)
            x1 = int(cx + 170 * math.cos(rad))
            y1 = int(cy + 170 * math.sin(rad))
            x2 = int(cx + 260 * math.cos(rad))
            y2 = int(cy + 260 * math.sin(rad))
            draw.line([(x1, y1), (x2, y2)], fill=accent, width=2)
    elif "blood" in cat_low:
        # Concentric ripple life rings
        for r in [170, 210, 255]:
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=accent, width=2)
    elif "eat" in cat_low or "nutrition" in cat_low:
        # Organic floating leaves
        for offset in [-190, 190]:
            draw.arc((cx + offset - 40, cy - 120, cx + offset + 40, cy - 40), 45, 225, fill=accent, width=4)
    elif "mental" in cat_low or "mind" in cat_low:
        # Concentric mindfulness wave arcs
        for r in [180, 220, 260]:
            draw.arc((cx - r, cy - r, cx + r, cy + r), 200, 340, fill=accent2, width=2)
    else:
        # Clean geometric corner accents
        draw.line([(30, 30), (70, 30)], fill=accent, width=3)
        draw.line([(30, 30), (30, 70)], fill=accent, width=3)
        draw.line([(width - 30, height - 30), (width - 70, height - 30)], fill=accent, width=3)
        draw.line([(width - 30, height - 30), (width - 30, height - 70)], fill=accent, width=3)


def generate_procedural_vector_art(
    category: str,
    prompt: str = "",
    width: int = 600,
    height: int = 600,
    theme_choice: Optional[str] = None
) -> Image.Image:
    """
    Generate rich, high-resolution graphic centerpiece artwork instantly.
    Renders dynamic themed gradient backgrounds, ambient atmospheric elements,
    and campaign-specific or custom subject centerpiece illustrations.
    """
    canvas = Image.new("RGB", (width, height), "#0F172A")
    draw = ImageDraw.Draw(canvas)
    cx, cy = width // 2, height // 2
    cat_low = (category or "").lower()
    prompt_low = (prompt or "").lower()
    base_pal = config.POSTER_PALETTES.get(category, config.POSTER_PALETTES.get("General Wellness", {}))
    palette = extract_palette_from_prompt(prompt, base_pal, theme_choice)

    # 1. Rich category/theme atmospheric gradient background (never a blank white box!)
    c_top = palette.get("bg_dark", "#0F172A")
    c_bottom = palette.get("primary", "#1E3A8A")
    draw_gradient_fill(draw, width, height, c_top, c_bottom)

    # 2. Ambient decorative elements (rays, sparkles, ripple rings, organic arcs)
    draw_ambient_elements(draw, width, height, category, palette, prompt_low)

    # 3. Dedicated Focal Art / Custom Subject
    if any(k in prompt_low for k in ["people", "person", "doctor", "nurse", "family", "patient", "elder", "children", "team", "crowd", "human"]):
        draw_people_and_scene_art(draw, cx, cy, palette, prompt)
    elif "cancer" in cat_low:
        draw_cancer_awareness_art(draw, cx, cy, palette)
    elif "healthy eating" in cat_low or "eat" in cat_low or "nutrition" in cat_low:
        draw_healthy_eating_art(draw, cx, cy, palette)
    elif "diabetes" in cat_low:
        draw_diabetes_awareness_art(draw, cx, cy, palette)
    elif "fitness" in cat_low or "physical" in cat_low or "sport" in cat_low:
        draw_physical_fitness_art(draw, cx, cy, palette)
    elif "women" in cat_low:
        draw_womens_health_art(draw, cx, cy, palette)
    elif "blood" in cat_low or "donor" in cat_low:
        draw_blood_donation_art(draw, cx, cy, palette)
    elif "mental" in cat_low or "mind" in cat_low:
        draw_mental_health_art(draw, cx, cy, palette)
    elif "heart" in cat_low or "cardio" in cat_low:
        draw_heart_and_lungs_art(draw, cx, cy, palette)
    elif "world health" in cat_low:
        draw_world_health_art(draw, cx, cy, palette)
    elif "wellness" in cat_low:
        draw_general_wellness_art(draw, cx, cy, palette)
    elif any(k in prompt_low for k in ["cancer", "ribbon", "oncolog", "tumor"]):
        draw_cancer_awareness_art(draw, cx, cy, palette)
    elif any(k in prompt_low for k in ["salad", "food", "eat", "fruit", "vegetable", "nutrition", "avocado", "diet"]):
        draw_healthy_eating_art(draw, cx, cy, palette)
    elif any(k in prompt_low for k in ["diabet", "sugar", "glucose", "insulin"]):
        draw_diabetes_awareness_art(draw, cx, cy, palette)
    elif any(k in prompt_low for k in ["run", "fitness", "exercise", "workout", "gym", "sport", "athletic"]):
        draw_physical_fitness_art(draw, cx, cy, palette)
    elif any(k in prompt_low for k in ["women", "female", "maternal", "breast"]):
        draw_womens_health_art(draw, cx, cy, palette)
    elif any(k in prompt_low for k in ["blood", "donor", "donation"]):
        draw_blood_donation_art(draw, cx, cy, palette)
    elif any(k in prompt_low for k in ["mental", "mind", "brain", "stress", "meditat", "inside"]):
        draw_mental_health_art(draw, cx, cy, palette)
    elif any(k in prompt_low for k in ["heart", "lung", "cardio", "pulse"]):
        draw_heart_and_lungs_art(draw, cx, cy, palette)
    elif any(k in prompt_low for k in ["globe", "earth", "world", "community"]):
        draw_world_health_art(draw, cx, cy, palette)
    else:
        draw_general_wellness_art(draw, cx, cy, palette)

    return canvas


# ------------------------------------------------------------------
# Artwork Edge Feathering & Smart Background Integration
# ------------------------------------------------------------------
def prepare_artwork_circle(image: Image.Image, diameter: int) -> Tuple[Image.Image, Image.Image]:
    """Fit artwork into an anti-aliased circle with soft feathered edges."""
    src = image.convert("RGB")
    ratio = src.width / src.height
    if ratio > 1.0:
        crop_w = int(src.height)
        left = (src.width - crop_w) // 2
        src = src.crop((left, 0, left + crop_w, src.height))
    else:
        crop_h = int(src.width)
        top = (src.height - crop_h) // 2
        src = src.crop((0, top, src.width, top + crop_h))

    src = _resize_artwork(src, (diameter, diameter))

    # Circular mask with slight anti-aliasing
    mask = Image.new("L", (diameter, diameter), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((2, 2, diameter - 3, diameter - 3), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=1.5))

    return src, mask


# ------------------------------------------------------------------
# POSTER LAYOUT 1: Split Editorial & Health Tips (Reference 3)
# ------------------------------------------------------------------
def create_split_editorial_poster(
    image: Image.Image, copy: PosterCopy, palette: Dict[str, str],
    category: str = "General Wellness",
) -> Image.Image:
    width, height = config.POSTER_WIDTH, config.POSTER_HEIGHT
    card_text = _readable_text_color(palette["bg_dark"], "#FFFFFF")
    card_accent = _readable_text_color(palette["bg_dark"], palette["accent"])
    poster = Image.new("RGB", (width, height), palette["bg_light"])
    draw = ImageDraw.Draw(poster)

    # 1. Top Header & Organization Badges
    org_font = _load_font(20, bold=True)
    draw.text((70, 48), "WELLNESS AWARENESS", fill=palette["primary"], font=org_font)
    draw.text((width - 320, 48), copy.org_name[:24].upper(), fill=palette["primary"], font=org_font)

    # Decorative header badges
    draw.ellipse((42, 48, 60, 66), fill=palette["accent"])
    draw.ellipse((width - 340, 48, width - 322, 66), fill=palette["accent"])

    # 2. Arched Slogan / Header Badge
    slogan = copy.slogan.strip() or "Take Care of What's Inside"
    slogan_font, slogan_lines = _fit_text(draw, f"• {slogan} •", 880, 2, 48, 30, bold=True)
    sy = 100
    for line in slogan_lines:
        lw = draw.textlength(line, font=slogan_font)
        draw.text(((width - lw) // 2, sy), line, fill=palette["primary"], font=slogan_font)
        sy += int(slogan_font.size * 1.15)

    # 3. Circular Artwork Frame with Glowing Border
    art_size = 500
    art_x = (width - art_size) // 2
    art_y = sy + 15
    art_cx = art_x + art_size // 2
    art_cy = art_y + art_size // 2

    # Elegant circular background halo
    draw.ellipse((art_x - 12, art_y - 12, art_x + art_size + 12, art_y + art_size + 12),
                 fill=palette.get("accent2", "#F0F8FA"))
    draw.ellipse((art_x - 6, art_y - 6, art_x + art_size + 6, art_y + art_size + 6),
                 fill="#FFFFFF")

    # Paste circular artwork
    art_img, art_mask = prepare_artwork_circle(image, art_size)
    poster.paste(art_img, (art_x, art_y), art_mask)

    # Thin gold accent ring around the circle
    draw.ellipse((art_x, art_y, art_x + art_size, art_y + art_size),
                 outline=palette["accent"], width=4)

    # 4. Deep Contrasting Lower Section (Dark Card)
    card_y = art_y + art_size - 40
    draw.rectangle((0, card_y, width, height), fill=palette["bg_dark"])

    # Smooth curved arch connecting top and bottom
    draw.pieslice((0, card_y - 50, width, card_y + 50), start=180, end=360, fill=palette["bg_dark"])

    # Re-draw the lower overlapping circle cleanly
    lower_crop_y = card_y + 10
    overlap_h = (art_y + art_size) - lower_crop_y
    if overlap_h > 0:
        sub_art = art_img.crop((0, art_size - overlap_h, art_size, art_size))
        sub_mask = art_mask.crop((0, art_size - overlap_h, art_size, art_size))
        poster.paste(sub_art, (art_x, lower_crop_y), sub_mask)
        draw.ellipse((art_x, art_y, art_x + art_size, art_y + art_size), outline=palette["accent"], width=3)

    # 5. Educational Body Copy inside Dark Card
    content_y = art_y + art_size + 40
    body_font, body_lines = _fit_text(draw, copy.body.strip(), 880, 4, 25, 18)
    for line in body_lines:
        lw = draw.textlength(line, font=body_font)
        draw.text(((width - lw) // 2, content_y), line, fill=card_text, font=body_font)
        content_y += int(body_font.size * 1.35)

    # 6. Actionable Health Tips with Custom Theme Bullet Markers
    content_y += 24
    content_y = _draw_wrapped_bullets(
        draw, copy.bullet_points, 140, content_y, 720, card_text,
        card_accent, category, font_size=20, max_items=4,
    )

    # 7. Inspirational Quote in Gold
    content_y += 18
    quote_text = f'"{copy.quote.strip()}"'
    quote_font, quote_lines = _fit_text(draw, quote_text, 860, 2, 23, 16, bold=True)
    for qline in quote_lines:
        qlw = draw.textlength(qline, font=quote_font)
        draw.text(((width - qlw) // 2, content_y), qline, fill=card_accent, font=quote_font)
        content_y += int(quote_font.size * 1.25)

    # 8. Footer (Helpline & Social Icons)
    foot_y = height - 70
    contact_font = _load_font(18, bold=True)
    draw.text((70, foot_y), copy.helpline, fill=card_text, font=contact_font)
    draw_social_icons(draw, width - 180, foot_y - 2, 22, card_accent)

    return poster


# ------------------------------------------------------------------
# POSTER LAYOUT 2: Radiant Sunburst Awareness (Reference 1)
# ------------------------------------------------------------------
def create_sunburst_poster(
    image: Image.Image, copy: PosterCopy, palette: Dict[str, str],
    category: str = "General Wellness",
) -> Image.Image:
    width, height = config.POSTER_WIDTH, config.POSTER_HEIGHT
    accent_text = _readable_text_color("#FFFFFF", palette["accent"])
    poster = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(poster)

    # 1. Radiant Sunburst Background
    sun_center = (width // 2, 460)
    sun_radius = 800
    color1 = palette.get("bg_light", "#F0F9FB")
    color2 = palette.get("primary", "#0B7A8B")
    draw_sunburst(draw, sun_center, sun_radius, color1, color2, num_rays=32)

    # Floating Soft Clouds in Sunburst Sky
    draw_cloud(draw, 40, 100, 220, 80, "#E8F4F8")
    draw_cloud(draw, width - 260, 140, 220, 80, "#E8F4F8")
    draw_cloud(draw, 50, 450, 180, 60, "#D2ECF2")
    draw_cloud(draw, width - 220, 480, 180, 60, "#D2ECF2")

    # 2. Glowing White Cloud Backplate for Artwork
    art_size = 520
    art_x = (width - art_size) // 2
    art_y = 210
    draw.ellipse((art_x - 30, art_y - 30, art_x + art_size + 30, art_y + art_size + 30), fill="#FFFFFF")

    # Paste Artwork
    art_img, art_mask = prepare_artwork_circle(image, art_size)
    poster.paste(art_img, (art_x, art_y), art_mask)

    # 3. Clean White/Light Bottom Panel for Content
    panel_y = 750
    draw.rectangle((0, panel_y, width, height), fill="#FFFFFF")
    # Soft cloud edge at top of panel
    for cx_step in range(0, width, 180):
        draw.ellipse((cx_step, panel_y - 30, cx_step + 200, panel_y + 40), fill="#FFFFFF")

    # 4. Bold Banner Headline (WORLD HEALTHY DAY)
    headline = copy.headline.strip() or "WORLD HEALTHY DAY"
    hl_font, hl_lines = _fit_text(draw, headline, 880, 2, 70, 42, bold=True)
    cur_y = panel_y + 35
    for line in hl_lines:
        lw = draw.textlength(line, font=hl_font)
        draw.text(((width - lw) // 2, cur_y), line, fill="#111827", font=hl_font)
        cur_y += int(hl_font.size * 1.1)

    # Slogan / Sub-headline
    slogan = copy.slogan.strip()
    s_font, s_lines = _fit_text(draw, slogan, 860, 2, 28, 20, bold=True)
    cur_y += 10
    for sline in s_lines:
        slw = draw.textlength(sline, font=s_font)
        draw.text(((width - slw) // 2, cur_y), sline, fill=accent_text, font=s_font)
        cur_y += int(s_font.size * 1.25)

    # Body educational paragraph
    cur_y += 15
    body_font, body_lines = _fit_text(draw, copy.body.strip(), 860, 5, 23, 17)
    for bline in body_lines:
        draw.text((80, cur_y), bline, fill="#374151", font=body_font)
        cur_y += int(body_font.size * 1.35)

    # Actionable Bullet Points
    cur_y += 20
    cur_y = _draw_wrapped_bullets(
        draw, copy.bullet_points, 80, cur_y, 860, "#1F2937",
        accent_text, category, font_size=19,
    )
    cur_y += 4
    cur_y = _draw_quote_block(
        draw, copy.quote, cur_y, width,
        accent_text, max_size=18,
    )

    # Footer
    foot_y = height - 60
    f_font = _load_font(18, bold=True)
    draw.text((80, foot_y), copy.org_name, fill=palette["primary"], font=f_font)
    draw.text((width - 340, foot_y), "REVIEW BEFORE PUBLICATION", fill="#6B7280", font=f_font)

    return poster


# ------------------------------------------------------------------
# POSTER LAYOUT 3: World Health Day / Community Card (Reference 2)
# ------------------------------------------------------------------
def create_community_poster(
    image: Image.Image, copy: PosterCopy, palette: Dict[str, str],
    category: str = "General Wellness",
) -> Image.Image:
    width, height = config.POSTER_WIDTH, config.POSTER_HEIGHT
    poster = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(poster)

    # 1. Cheerful Sun and Clouds in Upper Corners
    draw.ellipse((width - 160, 20, width - 40, 140), fill="#FCD34D")
    draw_cloud(draw, 30, 40, 200, 70, "#DCEFF2")
    draw_cloud(draw, width - 260, 70, 180, 60, "#F0F8FA")

    # 2. Cursive Script Accent ("World" or "National")
    script_font = _load_font(60, script=True)
    draw.text((360, 50), "World", fill="#111827", font=script_font)

    # Bold Sans-Serif Title ("HEALTH DAY")
    title = copy.headline.replace("WORLD", "").replace("World", "").strip() or "HEALTH DAY"
    title_font, title_lines = _fit_text(draw, title, 880, 2, 76, 48, bold=True)
    ty = 120
    for line in title_lines:
        lw = draw.textlength(line, font=title_font)
        draw.text(((width - lw) // 2, ty), line, fill="#0F172A", font=title_font)
        ty += int(title_font.size * 1.08)

    # 3. Organization Header & Logo Badge
    org_font = _load_font(21, bold=True)
    org_text = copy.org_name.strip() or "Your Organization Name"
    olw = draw.textlength(org_text, font=org_font)
    draw.ellipse(((width - olw) // 2 - 32, ty + 12, (width - olw) // 2 - 14, ty + 30), fill=palette["accent"])
    draw.text(((width - olw) // 2, ty + 8), org_text, fill="#475569", font=org_font)

    # 4. Central Illustrative Scene with Red Crosses & Clouds
    art_size = 540
    art_x = (width - art_size) // 2
    art_y = ty + 50
    draw.ellipse((art_x - 10, art_y - 10, art_x + art_size + 10, art_y + art_size + 10), fill="#F1F5F9")
    art_img, art_mask = prepare_artwork_circle(image, art_size)
    poster.paste(art_img, (art_x, art_y), art_mask)

    # Health cross accents around center
    draw_health_cross(draw, art_x - 40, art_y + 80, 36, "#EF4444")
    draw_health_cross(draw, art_x + art_size + 10, art_y + 120, 36, "#EF4444")
    draw_health_cross(draw, art_x + 60, art_y + art_size - 20, 30, "#EF4444")

    # 5. Bottom Tips Card
    card_y = art_y + art_size + 30
    draw.rounded_rectangle((70, card_y, width - 70, height - 90), radius=28, fill=palette.get("bg_light", "#F8FAFC"))

    # Slogan banner inside card
    slogan_font, slogan_lines = _fit_text(draw, copy.slogan.strip(), 820, 2, 28, 20, bold=True)
    cy = card_y + 24
    for sline in slogan_lines:
        slw = draw.textlength(sline, font=slogan_font)
        draw.text(((width - slw) // 2, cy), sline, fill=palette["primary"], font=slogan_font)
        cy += int(slogan_font.size * 1.2)

    # Bullet points
    cy += 10
    card_fill = palette.get("bg_light", "#F8FAFC")
    cy = _draw_wrapped_bullets(
        draw, copy.bullet_points, 110, cy, 780,
        _readable_text_color(card_fill, "#1E293B"), palette["accent"], category, font_size=19,
    )
    cy += 4
    cy = _draw_quote_block(
        draw, copy.quote, cy, width,
        _readable_text_color(card_fill, palette["primary"]), max_size=18,
    )

    # Footer
    draw.text((70, height - 60), copy.helpline, fill="#64748B", font=_load_font(18, bold=True))

    return poster


# ------------------------------------------------------------------
# POSTER LAYOUT 4: Minimalist Modern Geometry (Reference 4)
# ------------------------------------------------------------------
def create_minimalist_poster(
    image: Image.Image, copy: PosterCopy, palette: Dict[str, str],
    category: str = "General Wellness",
) -> Image.Image:
    width, height = config.POSTER_WIDTH, config.POSTER_HEIGHT
    poster = Image.new("RGB", (width, height), palette.get("bg_light", "#FFF5F2"))
    draw = ImageDraw.Draw(poster)

    # Split title words: First half at Top-Left, Second half at Bottom-Right
    headline = copy.headline.strip() or "SAFE BLOOD SAVES LIVES"
    words = headline.split()
    mid = max(1, len(words) // 2)
    top_title = " ".join(words[:mid])
    bot_title = " ".join(words[mid:])

    # 1. Top-Left Bold Headline
    top_font, top_lines = _fit_text(draw, top_title, 480, 2, 74, 46, bold=True)
    ty = 80
    for line in top_lines:
        draw.text((70, ty), line, fill=palette["primary"], font=top_font)
        ty += int(top_font.size * 1.08)

    # 2. Modern Geometric Line Accent Frame
    line_color = palette["accent"]
    lw = 8
    # Top border line
    draw.line([(width // 2, 90), (width - 90, 90)], fill=line_color, width=lw)
    draw.line([(width - 90, 90), (width - 90, 420)], fill=line_color, width=lw)
    draw.line([(width - 90, 420), (width // 2 + 100, 520)], fill=line_color, width=lw)

    # Central Artwork
    art_size = 500
    art_x = (width - art_size) // 2
    art_y = 380
    art_img, art_mask = prepare_artwork_circle(image, art_size)
    poster.paste(art_img, (art_x, art_y), art_mask)

    # Bottom border line
    draw.line([(90, 680), (90, 1020)], fill=line_color, width=lw)
    draw.line([(90, 1020), (450, 1020)], fill=line_color, width=lw)

    # 3. Bottom-Right Bold Headline
    bot_font, bot_lines = _fit_text(draw, bot_title, 480, 2, 74, 46, bold=True)
    by = 880
    for line in bot_lines:
        draw.text((width - 480, by), line, fill=palette["primary"], font=bot_font)
        by += int(bot_font.size * 1.08)

    # 4. Slogan Banner
    slogan = copy.slogan.strip().upper()
    s_font, s_lines = _fit_text(draw, slogan, 880, 2, 28, 18, bold=True)
    sy = 1140
    for sline in s_lines:
        slw = draw.textlength(sline, font=s_font)
        draw.text(((width - slw) // 2, sy), sline, fill=palette["primary"], font=s_font)
        sy += int(s_font.size * 1.25)

    # 5. Motivational quote
    quote_y = max(sy + 8, 1240)
    _draw_quote_block(
        draw, copy.quote, quote_y, width,
        _readable_text_color(palette.get("bg_light", "#FFF5F2"), palette["primary"]),
        max_size=18,
    )

    # 6. Dual Partner Badges (WHO & World Donor Day style)
    fy = height - 90
    badge_font = _load_font(20, bold=True)
    draw.ellipse((70, fy - 2, 94, fy + 22), fill=palette["accent"])
    draw.text((106, fy), copy.org_name or "Your Organization Name", fill=palette["primary"], font=badge_font)

    draw.ellipse((width - 320, fy - 2, width - 296, fy + 22), fill=palette["accent"])
    draw.text((width - 286, fy), copy.helpline or "Your helpline / website here", fill=palette["primary"], font=badge_font)

    return poster


def cover_fit_image(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize and center crop an image so it fills (target_w, target_h) with zero distortion."""
    src = img.convert("RGB")
    src_w, src_h = src.size
    target_aspect = target_w / target_h
    src_aspect = src_w / src_h

    if src_aspect > target_aspect:
        new_h = target_h
        new_w = int(src_w * (target_h / src_h))
        resized = _resize_artwork(src, (new_w, new_h))
        left = (new_w - target_w) // 2
        return resized.crop((left, 0, left + target_w, target_h))
    else:
        new_w = target_w
        new_h = int(src_h * (target_w / src_w))
        resized = _resize_artwork(src, (new_w, new_h))
        top = (new_h - target_h) // 2
        return resized.crop((0, top, target_w, top + target_h))


# ------------------------------------------------------------------
# POSTER LAYOUT 5: Modern Visual Hero (Full-Bleed Showcase)
# ------------------------------------------------------------------
def create_modern_hero_poster(
    image: Image.Image, copy: PosterCopy, palette: Dict[str, str],
    category: str = "General Wellness",
) -> Image.Image:
    """
    Modern Full-Bleed Hero Poster:
    Showcases the full AI-generated or vector visual across the top 62% of the poster,
    merging seamlessly into a dark modern lower card with bold typography and health tips.
    """
    width, height = config.POSTER_WIDTH, config.POSTER_HEIGHT
    dark_c = palette.get("bg_dark", "#0F172A")
    text_c = _readable_text_color(dark_c, palette.get("text_light", "#FFFFFF"))
    accent_c = _readable_text_color(dark_c, palette.get("accent", "#38BDF8"))
    poster = Image.new("RGB", (width, height), dark_c)
    draw = ImageDraw.Draw(poster)

    # 1. Full-Bleed Hero Visual across Top 62% with proper aspect cover
    hero_h = int(height * 0.62)
    hero_img = cover_fit_image(image, width, hero_h)
    poster.paste(hero_img, (0, 0))

    # Gradient fade overlay at bottom of image
    for y_fade in range(hero_h - 90, hero_h):
        draw.line([(0, y_fade), (width, y_fade)], fill=dark_c, width=1)

    # 2. Modern Header Badge over top of visual
    badge_w = 460
    badge_x = 50
    badge_y = 40
    draw.rounded_rectangle((badge_x, badge_y, badge_x + badge_w, badge_y + 44), radius=14, fill=dark_c)
    badge_font, badge_lines = _fit_text(
        draw, f"HEALTH & WELLNESS · {category.upper()}", badge_w - 40, 1, 18, 13, bold=True,
    )
    draw.text((badge_x + 20, badge_y + 11), badge_lines[0],
              fill=accent_c, font=badge_font)

    # 3. Lower Content Card
    card_y = hero_h - 40
    draw.rounded_rectangle((0, card_y, width, height), radius=36, fill=dark_c)

    # Glowing Accent Bar
    draw.line([(60, card_y + 20), (width - 60, card_y + 20)], fill=palette.get("accent", "#38BDF8"), width=4)

    # 4. Bold Headline
    headline = copy.headline.strip() or "WELLNESS FOR ALL"
    hl_font, hl_lines = _fit_text(draw, headline, 880, 2, 64, 38, bold=True)
    cur_y = card_y + 40
    for line in hl_lines:
        lw = draw.textlength(line, font=hl_font)
        draw.text(((width - lw) // 2, cur_y), line, fill=text_c, font=hl_font)
        cur_y += int(hl_font.size * 1.1)

    # 5. Slogan
    if copy.slogan:
        s_font, s_lines = _fit_text(draw, copy.slogan.strip(), 880, 2, 28, 18, bold=True)
        cur_y += 10
        for sline in s_lines:
            slw = draw.textlength(sline, font=s_font)
            draw.text(((width - slw) // 2, cur_y), sline, fill=accent_c, font=s_font)
            cur_y += int(s_font.size * 1.25)

    # 6. Educational Message
    if copy.body:
        b_font, b_lines = _fit_text(draw, copy.body.strip(), 880, 3, 22, 16)
        cur_y += 14
        for bline in b_lines:
            draw.text((80, cur_y), bline, fill=text_c, font=b_font)
            cur_y += int(b_font.size * 1.35)

    # 7. Actionable Health Tips with Category Bullet Icons
    cur_y += 18
    cur_y = _draw_wrapped_bullets(
        draw, copy.bullet_points, 80, cur_y, 840, text_c, accent_c, category, font_size=19,
    )

    cur_y += 1
    quote_color = _readable_text_color(dark_c, palette.get("accent", "#F5A623"))
    _draw_quote_block(draw, copy.quote, cur_y, width, quote_color, max_size=17)

    # 8. Footer
    foot_y = height - 70
    footer_font, footer_lines = _fit_text(draw, copy.helpline, 620, 1, 17, 13, bold=True)
    draw.text((70, foot_y), footer_lines[0], fill=text_c, font=footer_font)
    org_font, org_lines = _fit_text(draw, copy.org_name, 270, 1, 17, 13, bold=True)
    org_width = draw.textlength(org_lines[0], font=org_font)
    draw.text((width - 70 - org_width, foot_y), org_lines[0], fill=accent_c, font=org_font)

    return poster


def create_clean_campaign_poster(
    image: Image.Image,
    copy: PosterCopy,
    palette: Dict[str, str],
    category: str,
) -> Image.Image:
    """Render artwork and editable copy in separate, measured, overflow-safe regions."""
    width = config.POSTER_WIDTH
    art_height = 820
    padding = 72
    text_width = width - padding * 2
    art_bg = palette.get("bg_light", "#F2FAFB")
    canvas = Image.new("RGB", (width, art_height), art_bg)
    artwork = image.convert("RGB")
    max_art_width, max_art_height = width - padding * 2, art_height - 40
    scale = min(max_art_width / artwork.width, max_art_height / artwork.height)
    artwork = artwork.resize(
        (max(1, round(artwork.width * scale)), max(1, round(artwork.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas.paste(artwork, ((width - artwork.width) // 2, (art_height - artwork.height) // 2))
    draw = ImageDraw.Draw(canvas)

    text_color = _readable_text_color("#FFFFFF", palette.get("text_dark", "#0B1D30"))
    sections = [
        (copy.headline.strip(), 48, True, palette.get("primary", "#0B7B78"), 24, 2),
        (copy.slogan.strip(), 27, True, palette.get("primary", "#0B7B78"), 22, 2),
        (copy.body.strip(), 21, False, text_color, 22, 6),
    ]
    for bullet in copy.bullet_points:
        sections.append((f"-  {bullet.strip()}", 18, False, text_color, 11, 2))
    if copy.quote.strip():
        sections.append((copy.quote.strip(), 17, True, palette.get("primary", "#0B7B78"), 24, 3))
    sections.extend([
        (copy.org_name.strip(), 16, True, text_color, 8, 2),
        (copy.helpline.strip(), 16, False, text_color, 20, 2),
    ])

    prepared = []
    estimated_height = art_height + 44
    probe = ImageDraw.Draw(canvas)
    for text, max_size, bold, color, gap, max_lines in sections:
        if not text:
            continue
        font = _load_font(max_size, bold=bold)
        lines = _wrap_text(probe, text, font, text_width)
        while max_size > 12 and len(lines) > max_lines:
            max_size -= 1
            font = _load_font(max_size, bold=bold)
            lines = _wrap_text(probe, text, font, text_width)
        box = probe.textbbox((0, 0), "Ag", font=font)
        line_height = max(14, box[3] - box[1] + 5)
        block_height = len(lines) * line_height
        estimated_height += block_height + gap
        prepared.append((lines, font, color, line_height, gap))

    height = max(config.POSTER_HEIGHT, estimated_height + 50)
    poster = Image.new("RGB", (width, height), "#FFFFFF")
    poster.paste(canvas, (0, 0))
    draw = ImageDraw.Draw(poster)
    y = art_height + 34
    draw.line((padding, y, width - padding, y), fill=palette.get("accent", "#14B8A6"), width=4)
    y += 20
    for lines, font, color, line_height, gap in prepared:
        for line in lines:
            box = draw.textbbox((0, 0), line, font=font)
            line_width = box[2] - box[0]
            draw.text(((width - line_width) // 2, y - box[1]), line, fill=color, font=font)
            y += line_height
        y += gap
    return poster


# ------------------------------------------------------------------
# Main Unified Poster Factory
# ------------------------------------------------------------------
def create_campaign_poster(
    image: Image.Image,
    copy_or_headline: Union[PosterCopy, str],
    body: str = "",
    category: str = "General Wellness",
    layout_name: Optional[str] = None,
    bullet_points: Optional[List[str]] = None,
    slogan: str = "",
    quote: str = "",
    org_name: str = "",
    user_prompt: str = "",
    theme_choice: Optional[str] = None,
    custom_palette: Optional[Dict[str, str]] = None,
) -> Image.Image:
    """
    Factory creating a high-resolution campaign poster matching reference styles.
    Supports both structured PosterCopy and legacy argument signatures.
    """
    base_palette = config.POSTER_PALETTES.get(category, config.POSTER_PALETTES["General Wellness"])
    palette = extract_palette_from_prompt(
        user_prompt or body, base_palette,
        theme_choice=theme_choice, custom_palette=custom_palette
    )

    if isinstance(copy_or_headline, PosterCopy):
        copy = copy_or_headline
    else:
        # Construct from individual fields or fallback to preset
        base = get_campaign_preset(category)
        copy = PosterCopy(
            headline=copy_or_headline.strip() or base.headline,
            slogan=slogan.strip() or base.slogan,
            body=body.strip() or base.body,
            bullet_points=bullet_points if bullet_points is not None else base.bullet_points,
            quote=quote.strip() or base.quote,
            org_name=org_name.strip() or base.org_name,
            helpline=base.helpline,
        )

    # Layout selection
    target_layout = layout_name or config.DEFAULT_POSTER_LAYOUT

    return create_clean_campaign_poster(image, copy, palette, category)


# ------------------------------------------------------------------
# Disclaimer footer bar
# ------------------------------------------------------------------
def add_disclaimer_footer(image: Image.Image, text: str = config.DISCLAIMER_SHORT) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    font = _load_font(12, bold=True)
    draw_probe = ImageDraw.Draw(image)
    lines = _wrap_text(draw_probe, text, font, w - 32)
    box = draw_probe.textbbox((0, 0), "Ag", font=font)
    line_height = max(14, box[3] - box[1] + 2)
    bar_h = max(36, len(lines) * line_height + 12)
    canvas = Image.new("RGB", (w, h + bar_h), "#0B1D30")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)

    y = h + 6
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        draw.text(((w - (box[2] - box[0])) // 2, y - box[1]), line, fill="#FFFFFF", font=font)
        y += line_height
    return canvas
