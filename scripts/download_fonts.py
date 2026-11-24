"""Download the OFL-licensed font files required by the poster renderer."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "assets" / "fonts"
FONT_URLS = {
    "Inter-400.ttf": "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuLyfMZg.ttf",
    "Inter-700.ttf": "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuFuYMZg.ttf",
    "Poppins-700.ttf": "https://fonts.gstatic.com/s/poppins/v24/pxiByp8kv8JHgFVrLCz7V1s.ttf",
    "OFL-Inter.txt": "https://raw.githubusercontent.com/google/fonts/main/ofl/inter/OFL.txt",
    "OFL-Poppins.txt": "https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/OFL.txt",
}


def main() -> None:
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in FONT_URLS.items():
        target = FONT_DIR / filename
        if target.exists() and target.stat().st_size:
            continue
        try:
            with urlopen(url, timeout=20) as response:
                data = response.read()
        except OSError as exc:
            raise RuntimeError(f"Could not download required font asset {filename}: {exc}") from exc
        if not data:
            raise RuntimeError(f"Font server returned an empty file for {filename}.")
        target.write_bytes(data)
    print(f"Installed bundled font assets in {FONT_DIR}")


if __name__ == "__main__":
    main()
