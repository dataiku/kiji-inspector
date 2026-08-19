# Home Repair Agent Demo

A scripted multi-step agent that diagnoses three appliance problems (dishwasher leak, stuck garbage disposal, noisy water heater), captures its activations at training-compatible tool-decision points, and decomposes them through a trained Sparse Autoencoder.

Each problem has one short, natural initial tool query rather than an SAE
snapshot fabricated for every scripted evidence source. The query uses the
same scenario system message, tool inventory, chat template, disabled-thinking
setting, and `I'll use the` decision prefill as activation extraction during
training. The four manual/parts/tutorial/quote panels are evidence gathering,
not four separate model decisions.

## Files

| File | Purpose |
|---|---|
| `home_repair_demo.py` | CLI entrypoint. Runs the full pipeline end-to-end. |
| `home_repair_colab.ipynb` | Colab notebook walkthrough of the same pipeline. |
| `home_repair.json` | Scenario config — system prompt, tool list, contrast type descriptions. |
| `index.html` | Interactive viewer for the generated `ui_data.json`. |

## Run locally

```bash
pip install 'kiji-inspector[huggingface]'
huggingface-cli login
uv run python demo/home_repair/home_repair_demo.py
```

The script writes `analysis_results.json`, `agent_output.txt`, `per_problem_analyses.json`, and `ui_data.json` to `demo/home_repair/output/`. Open `index.html` from a local web server (e.g. `python -m http.server` in this directory) to see the interactive explanation.

If the activations were extracted and evaluated with the modified vLLM path,
build the UI payload directly from that report instead of re-extracting them
through HuggingFace:

```bash
uv run python demo/home_repair/home_repair_demo.py \
  --ui-from-evaluation demo/home_repair/output/prompt_alignment/vllm_native_evaluation.json \
  --sae-layer 27 \
  --output-dir demo/home_repair/output
```

The command verifies that the report contains the exact three decisions used
by the current demo before writing `ui_data.json`.

## Run in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dataiku/kiji-inspector/blob/main/demo/home_repair/home_repair_colab.ipynb)

The notebook needs an **A100 high-RAM** runtime (the base model is
`nvidia/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16`, ~30B parameters). Add a Colab
Secret named `HF_TOKEN` before running. Optionally add `YOUTUBE_API_KEY` to
fetch real tutorial results instead of the mock data.

## Hosting `index.html` from Colab

`index.html` loads its data with `fetch('output/ui_data.json')`, so it needs a real HTTP server — opening it through a `file://` URL or a notebook iframe won't work. The notebook's last cell handles this for you, but here's what it does:

1. **Stage the files** the way the page expects them — `index.html` at the root, `ui_data.json` under an `output/` subdirectory:

   ```python
   import shutil, urllib.request
   from pathlib import Path

   SERVE_ROOT = Path("/content/serve")
   (SERVE_ROOT / "output").mkdir(parents=True, exist_ok=True)

   urllib.request.urlretrieve(
       "https://raw.githubusercontent.com/dataiku/kiji-inspector/main/demo/home_repair/index.html",
       SERVE_ROOT / "index.html",
   )
   shutil.copy(OUTPUT_DIR / "ui_data.json", SERVE_ROOT / "output" / "ui_data.json")
   ```

2. **Start a static server** in the background:

   ```python
   import subprocess
   PORT = 8000
   subprocess.Popen(
       ["python", "-m", "http.server", str(PORT), "--directory", str(SERVE_ROOT)],
       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
   )
   ```

3. **Open it through Colab's port proxy** — Colab forwards localhost ports to a signed `*.googleusercontent.com` URL:

   ```python
   from google.colab import output
   output.serve_kernel_port_as_window(PORT)         # opens a new tab
   # output.serve_kernel_port_as_iframe(PORT, height="900")  # inline in the cell
   ```

`serve_kernel_port_as_window` pops the viewer into a new tab; `serve_kernel_port_as_iframe` embeds it directly in the notebook output. Pick whichever you prefer.

Re-running the cell spawns a second `http.server` on the same port; the duplicate fails silently. If you regenerate `ui_data.json`, just refresh the proxy tab — the running server picks up the new file automatically.
