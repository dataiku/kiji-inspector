# steering — feature-level steering on the shipped SAEs

Everything under this folder is built against the SAE release in `output/layer_<N>/`
(`575-lab/kiji-inspector-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, whose card declares
`base_model: nvidia/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16`) and against pairs drawn from the current
full sweep. It is kept separate from the older demos for one reason:

> **`../tool_selection/`, `../home_repair/` and `../spec_sheet/` embed feature data from a different
> dictionary.** Their feature indices, labels and L0 figures do not correspond to the shipped SAE, so
> numbers must never be carried across that boundary.

How far apart they are, measured on `../tool_selection/`'s own prompts at its own layer 43: it
reports mean L0 **84**, the shipped SAE gives **374**; **0 of the 5** features it displays for
`internal_vs_external_A` are active at all; those indices are undescribed in the shipped
`feature_descriptions.json`, and two of their four labels appear in no shipped layer. The base model
is *not* the difference — the two agree on the tool 14 / 14 within 0.05.

## Layout

| Path | What |
|---|---|
| `supply_chain/` | Four one-word-cue pairs, read at layer 43. A page. |
| `customer_support/` | Six framing-cue pairs, read at layer 34. A page. |
| `tool_selection/` | The published demo's seven pairs re-run against the shipped SAE — a layer study, no page. |
| `sweep/` | The three sweep stages and their outputs (`output/sweep_candidates/`, 2.8 G). |

The drivers stay in `../tool_selection/` (`capture_decisions.py`, `attribute_pairs.py`,
`trace_pairs.py`, `tool_selection_demo.py`) — they are scenario-general and shared. Pass
`--scenario <name>` and they resolve `pairs.json` / `probes.json` / `output/` under **this** folder;
run them with no `--scenario` and they drive the older published `../tool_selection/` demo.

## Where steering works: layers 34 and 43 only

The full ablation + cross-patch battery, run at all six SAE layers for all three scenarios
(18 runs). A cross-patch "flip" means clamping one side's features into the *other* side's prompt —
prompt unmodified — moved the recipient's tool to the donor's:

| layer | tool_selection | supply_chain | customer_support |
|---|---|---|---|
| 6 | 0/14 · 0/14 | 0/8 · 0/8 | 0/12 · 0/12 |
| 13 | 0/14 · 0/14 | 0/8 · 0/8 | 0/12 · 0/12 |
| 20 | 0/14 · 0/14 | 0/8 · 0/8 | 0/12 · 0/12 |
| 27 | 1/14 · 0/14 | 0/8 · 0/8 | 1/12 · 1/12 |
| 34 | 2/14 · 1/14 | 2/8 · 1/8 | **4/12 · 8/12** |
| 43 | 5/14 · 5/14 | **3/8 · 7/8** | 9/12 · 2/12 ⚠ |

*(cross-patch flips · ablation flips)*

Layers 6–20 are **inert** — not weak, zero flips in 102 attempts, best |Δp| under 0.08 against
control bands of ~0.02. The cue is described early and only becomes causal late.

## Reading the layer-43 column: agreement vs inversion

`customer_support` at layer 43 looks like the best cell in the table and is the worst. Splitting its
flips by what was actually clamped:

| | cue families (6) | bulk-only | features clamped |
|---|---:|---:|---:|
| supply_chain L43 | 3/8 | 0/8 | 22 |
| tool_selection L43 | 2/14 | 3/14 | 316 |
| customer_support L34 | 2/12 | 2/12 | 49 |
| **customer_support L43** | **1/12** | **8/12** | **1,162** |

Only one of those nine flips comes from the cue families; the other eight need all 1,162 base-active
features — at that layer, the entire representation. That transplants the prompt rather than
steering a cue, and the inverted ablation score (2/12: removing six families out of 1,169 does
nothing) is the tell.

So **cross-patch and ablation agreeing is a cheap degeneracy check.** They track each other on a
healthy layer (supply_chain 43: 3 and 7; tool_selection 43: 5 and 5; customer_support 34: 4 and 8)
and invert when the dictionary has collapsed on that input distribution. It is a better signal than
raw L0 because it is measured on the intervention itself.

Mean L0 at layer 43 — 374 (tool_selection), 34 (supply_chain), **1,169** (customer_support) — says
the same thing: layer 43 is usable for two of the three, degenerate for one.

## Run

Each demo's README carries its own commands. The shared shape:

```bash
DOCKER="docker run --rm --gpus all -v $PWD:/workspace -v /path/to/models:/models:ro \
  -v ${HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONPATH=/workspace/src:/workspace/demo/tool_selection -w /workspace 575lab/kiji-inspector:dev"
MODEL=/models/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16-no-mtp

$DOCKER python demo/tool_selection/capture_decisions.py --model-name $MODEL --scenario <name>
$DOCKER bash -c 'pip install -q "kernels>=0.15.2,<0.16" && \
  python demo/tool_selection/attribute_pairs.py --model-name '"$MODEL"' --scenario <name> --layer <L> \
    --activations demo/steering/<name>/output/capture/activations.npz'
```

Needs the SAE under `output/layer_<N>/` as **real files, not symlinks into the HF cache** — the
container mounts the cache elsewhere, so an absolute symlink dangles inside it.
