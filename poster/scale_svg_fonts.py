#!/usr/bin/env python3
"""Scale every font-size in an SVG by a factor, in place.

Used to enlarge text in the poster-specific copies of the architecture
diagrams (poster/images/*.svg) so labels stay legible at poster size. The
paper's source SVGs in paper/images/ are left untouched.

Usage: python3 scale_svg_fonts.py <file.svg> <factor>
"""
import re
import sys


def main(path: str, factor: float) -> None:
    with open(path, encoding="utf-8") as fh:
        svg = fh.read()

    def repl(m: re.Match) -> str:
        return f'font-size="{round(float(m.group(1)) * factor, 1)}"'

    new, n = re.subn(r'font-size="([0-9.]+)"', repl, svg)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new)
    print(f"scaled {n} font-size values by {factor} in {path}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]))
