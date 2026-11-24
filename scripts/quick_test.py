"""
scripts/quick_test.py
---------------------
Command-line smoke test: checks your environment, loads the model and
generates ONE image without starting the web app.

Run from the project root:
    python scripts/quick_test.py
    python scripts/quick_test.py --model "Stable Diffusion v1.5 (classic, higher detail)"
"""

import argparse
import sys
from pathlib import Path

# Make the project root importable when running this file directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from modules import image_utils  # noqa: E402
from modules.image_generator import (  # noqa: E402
    GenerationError,
    WellnessImageGenerator,
    check_dependencies,
    get_device_info,
)
from modules.prompt_processor import build_prompt, validate_user_input  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="WellAware AI quick test")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, choices=list(config.MODELS))
    parser.add_argument("--prompt", default="A happy family walking together in a sunny park")
    parser.add_argument("--category", default="Heart Health", choices=list(config.CATEGORIES))
    parser.add_argument("--style", default=config.DEFAULT_STYLE, choices=list(config.STYLES))
    args = parser.parse_args()

    print("1) Checking dependencies…")
    dep_error = check_dependencies()
    if dep_error:
        print("   ❌", dep_error)
        return 1
    import torch
    import diffusers
    print(f"   ✅ torch {torch.__version__} | diffusers {diffusers.__version__}")

    info = get_device_info()
    print(f"2) Device: {info.device} ({info.name}) dtype={info.dtype_name}"
          + (f" VRAM={info.vram_gb} GB" if info.vram_gb else ""))
    if not info.is_gpu:
        print("   ⚠️ No GPU found – this will be slow but should still work with SD-Turbo.")

    print("3) Validating & building prompt…")
    check = validate_user_input(args.prompt)
    if not check.ok:
        print("   ❌", check.errors)
        return 1
    bundle = build_prompt(args.prompt, args.category, args.style)
    print("   Prompt:", bundle.positive[:160], "…")

    print(f"4) Loading model '{args.model}' (first run downloads several GB)…")
    try:
        generator = WellnessImageGenerator(args.model).load()
        print("5) Generating…")
        result = generator.generate(bundle.positive, bundle.negative, 512, 512, num_images=1)[0]
    except GenerationError as exc:
        print("   ❌", exc)
        return 1

    image = image_utils.add_disclaimer_footer(result.image)
    path = image_utils.save_image(image, "quick_test.png")
    print(f"   ✅ Done in {result.seconds}s (seed {result.seed}). Saved to: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
