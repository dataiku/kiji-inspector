# Presentation

Slide deck built with [Marp](https://marp.app/) using a custom theme (`theme.css`).

## Prerequisites

Install the Marp CLI:

```bash
npm install -g @marp-team/marp-cli
```

## Generating Output

All commands should be run from the `presentation/` directory.

**PDF:**

```bash
marp --theme theme.css --pdf \
  "Opening the Black Box Mechanistic Interpretability of Agent Tool Selection with Sparse Autoencoders.md"
```

**PPTX:**

```bash
marp --theme theme.css --pptx \
  "Opening the Black Box Mechanistic Interpretability of Agent Tool Selection with Sparse Autoencoders.md"
```

Both commands write the output file alongside the source Markdown.

## Notes

- The `--html` flag may be needed if the slides use raw HTML beyond what Marp allows by default.
- Images are referenced from `../paper/images/`. Make sure the `paper/images/` directory is present at the repository root.
