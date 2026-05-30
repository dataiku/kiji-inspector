"""Build a 4-slides-per-page handout PDF with speaker notes."""
import base64
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
MD_FILE = HERE / "Opening the Black Box Mechanistic Interpretability of Agent Tool Selection with Sparse Autoencoders.md"
SLIDES_DIR = HERE / "handout_slides"
HANDOUT_HTML = HERE / "handout.html"
HANDOUT_PDF = HERE / "Opening the Black Box - Handout.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def parse_slides(md_text: str):
    body = re.split(r"^---\s*$", md_text, maxsplit=1, flags=re.MULTILINE)[1]
    body = re.split(r"<style>.*?</style>", body, maxsplit=1, flags=re.DOTALL)[1]
    parts = re.split(r"^---\s*$", body, flags=re.MULTILINE)

    slides = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        title = None
        for line in part.splitlines():
            m = re.match(r"^#{1,3}\s+(.+?)\s*$", line.strip())
            if m:
                title = re.sub(r"<[^>]+>", "", m.group(1)).strip()
                title = re.sub(r"&mdash;", "—", title)
                break
        comments = re.findall(r"<!--(.*?)-->", part, flags=re.DOTALL)
        notes = []
        for c in comments:
            c_stripped = c.strip()
            if not c_stripped:
                continue
            if c_stripped.startswith("_"):
                continue
            if c_stripped.startswith("<") and c_stripped.endswith(">"):
                continue
            if c_stripped.startswith("- "):
                continue
            notes.append(c_stripped)
        note_text = "\n\n".join(notes) if notes else ""
        slides.append({"title": title or "(untitled)", "notes": note_text})
    return slides


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_html(slides):
    images = sorted(SLIDES_DIR.glob("slide.*.png"))
    assert len(images) == len(slides), f"slides {len(slides)} vs images {len(images)}"

    cards = []
    for i, (slide, img) in enumerate(zip(slides, images), start=1):
        data_uri = f"data:image/png;base64,{encode_image(img)}"
        title = slide["title"].replace("&", "&amp;").replace("<", "&lt;")
        notes = slide["notes"].replace("&", "&amp;").replace("<", "&lt;")
        cards.append(f"""
        <div class="card">
          <div class="card-head">
            <span class="num">{i:02d}</span>
            <span class="title">{title}</span>
          </div>
          <img class="thumb" src="{data_uri}" alt="Slide {i}" />
          <div class="notes">{notes}</div>
        </div>
        """)

    pages = []
    for i in range(0, len(cards), 4):
        chunk = cards[i:i + 4]
        while len(chunk) < 4:
            chunk.append('<div class="card empty"></div>')
        pages.append(f'<section class="page">{"".join(chunk)}</section>')

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Opening the Black Box — Handout</title>
<style>
  @page {{ size: A4 portrait; margin: 10mm; }}
  html, body {{ margin: 0; padding: 0; font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; color: #0F1314; }}
  .page {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 6mm;
    width: 190mm;
    height: 277mm;
    page-break-after: always;
    box-sizing: border-box;
  }}
  .page:last-child {{ page-break-after: auto; }}
  .card {{
    display: flex;
    flex-direction: column;
    border: 1px solid #d4dcd6;
    border-radius: 4px;
    padding: 3mm;
    box-sizing: border-box;
    overflow: hidden;
  }}
  .card.empty {{ border: none; }}
  .card-head {{
    display: flex;
    align-items: baseline;
    gap: 4mm;
    border-bottom: 1px solid #e5ebe7;
    padding-bottom: 1.5mm;
    margin-bottom: 2mm;
  }}
  .num {{
    font-size: 8pt;
    font-weight: 700;
    color: #0A2A1F;
    letter-spacing: 0.06em;
    background: #D1FFE6;
    padding: 0.5mm 2mm;
    border-radius: 2mm;
  }}
  .title {{
    font-size: 9pt;
    font-weight: 600;
    color: #0A2A1F;
    line-height: 1.2;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  .thumb {{
    width: 100%;
    height: auto;
    border: 1px solid #e5ebe7;
    border-radius: 2px;
    object-fit: contain;
    background: #f4f7f2;
  }}
  .notes {{
    font-size: 7.5pt;
    line-height: 1.35;
    color: #2B3B64;
    margin-top: 2mm;
    overflow: hidden;
    white-space: pre-wrap;
    flex: 1;
  }}
</style>
</head>
<body>
{"".join(pages)}
</body>
</html>
"""


def main():
    md = MD_FILE.read_text(encoding="utf-8")
    slides = parse_slides(md)
    print(f"parsed {len(slides)} slides", file=sys.stderr)
    no_notes = [i + 1 for i, s in enumerate(slides) if not s["notes"]]
    if no_notes:
        print(f"slides without notes: {no_notes}", file=sys.stderr)
    HANDOUT_HTML.write_text(build_html(slides), encoding="utf-8")
    print(f"wrote {HANDOUT_HTML}", file=sys.stderr)
    subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={HANDOUT_PDF}",
            HANDOUT_HTML.as_uri(),
        ],
        check=True,
    )
    print(f"wrote {HANDOUT_PDF}", file=sys.stderr)


if __name__ == "__main__":
    main()
