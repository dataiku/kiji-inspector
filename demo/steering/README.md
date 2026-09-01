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
| `supply_chain_expanded/` | 32 pairs sampled from the full gate-passing population — rate estimation for the paper, no page. |
| `customer_support_expanded/` | Same for customer support, read at layer 34. |
| `*_heldout/` | Pairs written along contrast axes absent from the training corpus — an out-of-distribution probe. |
| `*_l27/` | The expanded pairs re-run at layer 27, a layer no selection step touched. |
| `*_seed1/` | The expanded sample redrawn with a second seed. |
| `*_early/` | The expanded pairs at layers 6 and 20. |
| `sweep/` | The three sweep stages and their outputs (`output/sweep_candidates/`, 2.8 G). |

The drivers stay in `../tool_selection/` (`capture_decisions.py`, `attribute_pairs.py`,
`trace_pairs.py`, `ceiling_pairs.py`, `tool_selection_demo.py`) — they are scenario-general and
shared. Pass `--scenario <name>` and they resolve `pairs.json` / `probes.json` / `output/` under
**this** folder; run them with no `--scenario` and they drive the older published
`../tool_selection/` demo.

| Driver | Writes | What it answers |
|---|---|---|
| `capture_decisions.py` | `output/capture/` | What tool does each side pick, and what is active at the decision token? |
| `attribute_pairs.py` | `output/steering_layer<L>/` | Ablation and cross-patch on the cue families, with three control families each. |
| `trace_pairs.py` | `output/trace_layer<L>/` | Dose curves, position sweeps, and free generation under the clamp. |
| `ceiling_pairs.py` | `output/ceiling_layer<L>/` | **How much causal signal is at that token at all** — a full residual patch, difference-in-means, and random directions, with no dictionary in the path. |

`ceiling_pairs.py` is what turns a flip count into a recovery fraction. Give it `--battery
<steering_results.json>` and it additionally runs difference-in-means and random directions at the
norm of the cue clamp's *own* residual change, which is the size-matched comparison; without it the
extra arms act at the full donor-minus-recipient norm and only bound the result.

## Three control families

Every intervention is compared against random sets of features drawn from the same prompt. Which
random set is the right one depends on what is being claimed, so `attribute_pairs.py` stores three:

| Family | Matched on | The band for |
|---|---|---|
| `controls` | one cue family's count and activation mass | a single reported row |
| `setControls` | the **whole** cue set's count and mass | the set-level effect (`allRows`) |
| `contrastControls` (ablation) / `deltaControls` (cross-patch) | how much the set **differs across the pair**, `Σ\|X−Y\|` | whether the effect comes from the cue-ness or from the size |

The third is the one that tests the *selection rule*: cue families are picked because they differ
across the pair, so a control matched only on mass leaves that uncontrolled. Draws accumulate until
they reach the target, so a matched draw always carries at least as much as the set it controls for.
When the eligible pool cannot reach the target the draw is the whole pool and the record is marked
`massMatched: false` — a **ceiling**, which is itself informative: nothing else on that prompt
differs across the pair that much. Ceilings are reported separately and kept out of matched tests.

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
| 34 | 2/14 · 0/14 | 2/8 · 1/8 | **4/12 · 6/12** |
| 43 | 5/14 · 3/14 | **3/8 · 7/8** | 9/12 · 1/12 ⚠ |

*(cross-patch flips · ablation flips)*

These are **directed** flips, the same definition `paper/steering/` uses: the argmax moved *and*
landed on the other side's tool. An argmax change onto some third tool is disruption, not steering,
and is counted separately in the report as `ablationFlipsAnyTool` / `crossPatchFlipsAnyTool`. The two
definitions differ in five of these 36 cells, never by more than two flips — for example
`customer_support` L34 ablation is 6 directed against 8 any-tool.

Layers 6–20 are **inert** — not weak, zero flips in 102 attempts per arm, best |Δp| 0.151 for
ablation and 0.093 for cross-patch, and never clear of the set-matched control band at that layer.
Nor is that an artefact of intervening in the wrong place: at layer 20 the same features ablated at
the *request* tokens, at every position but the decision token, or everywhere at once still flip
nothing (`paper/steering/`, Table 4). The cue is described early and only becomes causal late.

## Reading the layer-43 column: agreement vs inversion

`customer_support` at layer 43 looks like the best cell in the table and is the worst. Splitting its
flips by what was actually clamped:

| | cue families | bulk-only | median features clamped |
|---|---:|---:|---:|
| supply_chain L43 | 3/8 | 0/8 | 22 |
| tool_selection L43 | 2/14 | 3/14 | 352 |
| customer_support L34 | 2/12 | 2/12 | 48 |
| **customer_support L43** | **1/12** | **8/12** | **1,163** |

Only one of those nine flips comes from the cue families; the other eight need every base-active
feature — at that layer, the entire representation. That transplants the prompt rather than
steering a cue, and the inverted ablation score (1/12: removing six families out of ~1,163 does
nothing) is the tell.

So **cross-patch and ablation agreeing is a cheap degeneracy check.** They track each other on a
healthy layer (supply_chain 43: 3 and 7; tool_selection 43: 5 and 3; customer_support 34: 4 and 6)
and invert when the dictionary has collapsed on that input distribution. It is measured on the
intervention itself rather than on the dictionary in the abstract — but it is only defined where the
ablation arm flips at least once, which is 6 of these 18 cells, so read a zero-ablation cell as
uninformative rather than as collapsed.

Two cheaper signals separate the collapsed cell without any intervention at all, and the paper leads
with them: the share of the active code that fires on *every* prompt (**0.97** for customer_support
L43 against at most 0.42 anywhere else) and the median count of features active on one side of a
pair and not the other (**1** against 5–69).

Mean L0 at layer 43 — 374 (tool_selection), 32 (supply_chain), **1,167** (customer_support) — points
the same way but is the weakest of the three: this dictionary already runs at mean L0 557 on its own
training data, so 1,167 is the top of a range it always occupied, and tool_selection's 374 is its
training median on a scenario that works fine. Density is not collapse.

## Run

Each demo's README carries its own commands. The shared shape:

```bash
DOCKER="docker run --rm --gpus all -v $PWD:/workspace -v /path/to/models:/models:ro \
  -v ${HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONPATH=/workspace/src:/workspace/demo/tool_selection -w /workspace 575lab/kiji-inspector:dev"
MODEL=/models/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16-no-mtp   # built by strip_mtp; see docs/index.md

$DOCKER python demo/tool_selection/capture_decisions.py --model-name $MODEL --scenario <name>
$DOCKER bash -c 'pip install -q "kernels>=0.15.2,<0.16" && \
  python demo/tool_selection/attribute_pairs.py --model-name '"$MODEL"' --scenario <name> --layer <L> \
    --activations demo/steering/<name>/output/capture/activations.npz'
```

Needs the SAE under `output/layer_<N>/` as **real files, not symlinks into the HF cache** — the
container mounts the cache elsewhere, so an absolute symlink dangles inside it.
