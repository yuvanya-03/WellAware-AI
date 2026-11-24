"""Persistence helpers for ready-made campaign posters."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import config
from modules.image_utils import png_bytes_to_pil, slugify


def load_saved_preset_posters(directory: Optional[Path] = None) -> Dict[str, Dict[str, object]]:
    poster_dir = directory or config.PRESET_POSTER_DIR
    posters: Dict[str, Dict[str, object]] = {}
    for category, preset in config.CAMPAIGN_PRESETS.items():
        filename = f"{slugify(category)}.png"
        path = poster_dir / filename
        if not path.is_file():
            continue

        png_bytes = path.read_bytes()
        image = png_bytes_to_pil(png_bytes)
        posters[category] = {
            "id": f"preset-{slugify(category)}",
            "png_bytes": png_bytes,
            "filename": filename,
            "saved_path": str(path),
            "category": category,
            "layout": config.DEFAULT_POSTER_LAYOUT,
            "poster_title": preset["headline"],
            "poster_slogan": preset["slogan"],
            "size": f"{image.width}×{image.height}",
            "created": datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b %Y, %H:%M:%S"),
            "preset_campaign": True,
        }
    return posters


def load_saved_gallery_posters(directory: Optional[Path] = None) -> list[dict]:
    output_dir = directory or config.OUTPUT_DIR
    category_names = set(config.CATEGORIES) | set(config.CAMPAIGN_PRESETS)
    category_by_slug = {slugify(category): category for category in category_names}
    filename_pattern = re.compile(
        r"^wellaware_(?P<category>[a-z0-9-]+)_(?P<timestamp>\d{8}_\d{6})_(?P<seed>\d+)\.png$"
    )
    posters = []

    if output_dir.is_dir():
        files = sorted(
            (path for path in output_dir.iterdir() if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in files:
            match = filename_pattern.fullmatch(path.name)
            if not match:
                continue
            png_bytes = path.read_bytes()
            image = png_bytes_to_pil(png_bytes)
            category_slug = match.group("category")
            category = category_by_slug.get(category_slug, category_slug.replace("-", " ").title())
            preset = config.CAMPAIGN_PRESETS.get(category)
            created = datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b %Y, %H:%M:%S")
            posters.append({
                "id": f"saved-{path.stem}",
                "png_bytes": png_bytes,
                "filename": path.name,
                "saved_path": str(path),
                "category": category,
                "layout": config.DEFAULT_POSTER_LAYOUT,
                "poster_title": preset["headline"] if preset else category,
                "size": f"{image.width}×{image.height}",
                "created": created,
                "preset_campaign": False,
            })

    preset_dir = config.PRESET_POSTER_DIR if directory is None else output_dir / "preset_campaigns"
    posters.extend(load_saved_preset_posters(preset_dir).values())
    return posters


def delete_saved_gallery_poster(saved_path: str) -> None:
    target = Path(saved_path).resolve()
    allowed_directories = {
        config.OUTPUT_DIR.resolve(),
        config.PRESET_POSTER_DIR.resolve(),
    }
    if target.parent not in allowed_directories:
        raise ValueError("Poster path is outside the generated poster folders.")
    if target.is_file():
        target.unlink()
