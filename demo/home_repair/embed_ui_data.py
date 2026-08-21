#!/usr/bin/env python3
"""Embed output/ui_data.json into index.html as the page's first-paint fallback.

index.html fetches ``output/ui_data.json`` at runtime; the embedded copy only
keeps the page from looking empty when it is opened before a run has finished
(e.g. straight from GitHub Pages or a fresh Colab).  Re-run after regenerating
``ui_data.json`` so both stay on the same schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_START = "/* MOCK_DATA_START */"
_END = "/* MOCK_DATA_END */"


def embed(html: str, ui_data: dict) -> str:
    start = html.index(_START) + len(_START)
    end = html.index(_END)
    payload = json.dumps(ui_data, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</script", "<\\/script")
    return f"{html[:start]} {payload} {html[end:]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--ui-data", default=str(here / "output" / "ui_data.json"))
    parser.add_argument("--html", default=str(here / "index.html"))
    args = parser.parse_args()
    html_path = Path(args.html)
    ui_data = json.loads(Path(args.ui_data).read_text())
    html_path.write_text(embed(html_path.read_text(), ui_data))
    print(f"Embedded {args.ui_data} into {html_path}")


if __name__ == "__main__":
    main()
