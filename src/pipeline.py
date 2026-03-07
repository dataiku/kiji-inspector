#!/usr/bin/env python3
"""
End-to-end pipeline for contrastive SAE analysis of tool-selection decisions.

Step 1: Extract raw activations from the subject model as numpy shards
Step 2: Train a JumpReLU SAE on the raw activations
Step 3: Identify decision-relevant SAE features using contrastive pairs
Step 4: Interpret features -- label with LLM, generate decision report
Step 5: Evaluate feature explanations via fuzzing

Pair generation is a separate CLI tool:
    uv run python -m generate_pairs 1300

Usage:
    # Run all steps (1-5)
    uv run python -m pipeline

    # Individual steps
    uv run python -m pipeline --step 1
    uv run python -m pipeline --step 2
    uv run python -m pipeline --step 3
    uv run python -m pipeline --step 4
    uv run python -m pipeline --step 5

    # Multi-layer: extract 3 layers in one pass, train 3 SAEs
    uv run python -m pipeline --layers 10 20 30

    # Use a different subject model
    uv run python -m pipeline --subject-model Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SUBJECT_MODEL_DEFAULT = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def _resolve_model_defaults(args: argparse.Namespace) -> None:
    """Auto-detect layer and d_sae defaults based on the subject model config.

    Called after parse_args() so we can inspect the model's actual architecture
    before any heavy loading happens.
    """
    needs_layer = args.layers is None
    needs_d_sae = args.d_sae is None
    if not needs_layer and not needs_d_sae:
        return

    from transformers import AutoConfig

    print(f"  Auto-detecting model config for {args.subject_model}...")
    config = AutoConfig.from_pretrained(args.subject_model, trust_remote_code=True)

    if needs_layer:
        num_layers = getattr(config, "num_hidden_layers", 30)
        default_layer = int(num_layers * 2 / 3)
        args.layers = [default_layer]
        print(f"  Auto-selected layer {default_layer} (~2/3 of {num_layers} layers)")

    if needs_d_sae:
        hidden_size = getattr(config, "hidden_size", 4096)
        args.d_sae = 4 * hidden_size
        print(f"  Auto-selected d_sae={args.d_sae} (4x hidden_size={hidden_size})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Contrastive SAE analysis pipeline (steps 1-5).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Base output directory. Per-layer outputs go to output/layer_N/ (default: output).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="GPU batch size (prompts) for activation extraction (default: 512).",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=500_000,
        help="Activation vectors per numpy shard (default: 500000).",
    )
    # -- Judging model (Steps 4, 5) --
    p.add_argument(
        "--judging-model",
        type=str,
        default="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        dest="judging_model",
        help="HuggingFace model for feature labeling and fuzzing judging via vLLM.",
    )
    # Backward-compatible alias (hidden from help)
    p.add_argument(
        "--qwen-model",
        type=str,
        default=argparse.SUPPRESS,
        dest="judging_model",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--generation-tp-size",
        type=int,
        default=4,
        help="Tensor parallel size for vLLM generation (default: 4 for 4xGB200).",
    )
    p.add_argument(
        "--generation-dp-size",
        type=int,
        default=1,
        help="Data parallel size for vLLM generation: runs N model copies "
        "on N GPUs for N× throughput (default: 1).",
    )
    p.add_argument(
        "--extraction-dp-size",
        type=int,
        default=1,
        help="Data parallel size for subject model extraction: runs N model copies "
        "on N GPUs for N× throughput (default: 1).",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Max sequence length for vLLM generation model (default: 16384).",
    )
    # -- Subject model (Step 1) --
    p.add_argument(
        "--subject-model",
        type=str,
        default=_SUBJECT_MODEL_DEFAULT,
        dest="subject_model",
        help="HuggingFace model ID for activation extraction — the model under study "
        f"(default: {_SUBJECT_MODEL_DEFAULT}).",
    )
    # Backward-compatible alias (hidden from help)
    p.add_argument(
        "--nemotron-model",
        type=str,
        default=argparse.SUPPRESS,
        dest="subject_model",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Transformer layers to extract and analyze (default: auto-detect ~2/3 depth). "
        "Multiple layers produce per-layer SAEs and reports.",
    )
    # -- SAE training (Step 2) --
    p.add_argument(
        "--d-sae",
        type=int,
        default=None,
        help="SAE hidden dimension (default: auto 4x d_model).",
    )
    p.add_argument(
        "--sae-lr",
        type=float,
        default=3e-4,
        help="SAE training learning rate (default: 3e-4).",
    )
    p.add_argument(
        "--sae-batch-size",
        type=int,
        default=8192,
        help="SAE training batch size in tokens (default: 8192).",
    )
    p.add_argument(
        "--sae-steps",
        type=int,
        default=None,
        help="SAE training steps (default: auto from data size).",
    )
    p.add_argument(
        "--sae-epochs",
        type=int,
        default=10,
        help="SAE training epochs (default: 10).",
    )
    p.add_argument(
        "--no-auto-scale-steps",
        action="store_true",
        default=False,
        help="Disable auto-scaling of warmup/checkpoint step counts.",
    )
    p.add_argument(
        "--l1-coefficient",
        type=float,
        default=5e-3,
        help="SAE sparsity penalty (default: 5e-3).",
    )
    p.add_argument(
        "--sae-checkpoint-dir",
        type=str,
        default=None,
        help="SAE checkpoint directory (default: output/sae_checkpoints).",
    )
    p.add_argument(
        "--sae-resume",
        type=str,
        default=None,
        help="Resume SAE training from checkpoint.",
    )
    # -- Feature identification (Step 3) --
    p.add_argument(
        "--sae-checkpoint",
        type=str,
        default=None,
        help="Path to trained SAE checkpoint for step 3 (default: auto from step 2).",
    )
    p.add_argument(
        "--top-k-features",
        type=int,
        default=200,
        help="Number of top features to report per contrast type (default: 200).",
    )
    p.add_argument(
        "--min-effect-size",
        type=float,
        default=0.3,
        help="Minimum Cohen's d for contrastive features (default: 0.3).",
    )
    p.add_argument(
        "--min-activation",
        type=float,
        default=0.01,
        help="Minimum mean activation for at least one condition (default: 0.01).",
    )
    # -- Feature interpretation (Step 4) --
    p.add_argument(
        "--label-top-n",
        type=int,
        default=20,
        help="Top-activating examples per feature for labeling (default: 20).",
    )
    p.add_argument(
        "--label-bottom-n",
        type=int,
        default=10,
        help="Near-zero examples per feature for labeling (default: 10).",
    )
    p.add_argument(
        "--label-batch-size",
        type=int,
        default=512,
        help="GPU batch size for activation extraction in step 4a (default: 512).",
    )
    # -- Fuzzing evaluation (Step 5) --
    p.add_argument(
        "--fuzz-top-k-tokens",
        type=int,
        default=5,
        help="Tokens to highlight per fuzzing example (default: 5).",
    )
    p.add_argument(
        "--fuzz-examples-per-feature",
        type=int,
        default=10,
        help="Max fuzzing examples per feature (default: 10).",
    )
    p.add_argument(
        "--fuzz-batch-size",
        type=int,
        default=64,
        help="GPU batch size for per-token extraction in step 5a (default: 64).",
    )
    # -- Flow control --
    p.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "5", "6", "all"],
        help="Run a single step: 1=extract activations, 2=train SAE, "
        "3=identify features, 4=interpret features, 5=fuzzing eval. "
        "all=1+2+3+4+5 (default: all).",
    )
    p.add_argument(
        "--pairs-dir",
        type=str,
        default="output/pairs",
        help="Path to pre-generated pairs parquet directory (default: output/pairs).",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "hf"],
        help="Backend for activation extraction: 'vllm' (fast, default) "
        "or 'hf' (HuggingFace Transformers, required for ablation).",
    )

    args = p.parse_args()
    return args


# ---------------------------------------------------------------------------
# Step 1: Extract raw activations (numpy shards)
# ---------------------------------------------------------------------------


def extract_activations(
    pairs: list,
    output_dir: str,
    subject_model: str,
    layers: list[int],
    batch_size: int,
    shard_size: int,
    scenarios_meta: dict | None = None,
    backend: str = "vllm",
    dp_size: int = 1,
) -> dict[str, Path]:
    """Load subject model, extract raw activations for all layers, save as numpy shards.

    Returns:
        Dict mapping layer_key (e.g. "residual_20") to its activations directory.
    """
    from extraction import create_extractor
    from extraction.extractor import RawActivationExtractor

    layer_keys = [f"residual_{layer}" for layer in layers]

    if dp_size > 1 and backend == "vllm":
        # DP path: don't load the full model in the main process — only
        # load the tokenizer and config so we can format prompts and read
        # hidden_size.  The DP workers will each load their own model copy.
        from transformers import AutoConfig, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(subject_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        hf_config = AutoConfig.from_pretrained(subject_model, trust_remote_code=True)
        hidden_size = hf_config.hidden_size

        # Build a lightweight stub so RawActivationExtractor can format
        # prompts and read hidden_size without a full model.
        class _ConfigStub:
            def __init__(self):
                self.model_name = subject_model
                self.layers = layers
                self.token_positions = "decision"
                self.gpu_memory_utilization = 0.90
                self.max_model_len = 8192
                self.trust_remote_code = True

        class _ExtractorStub:
            def __init__(self):
                self.config = _ConfigStub()
                self.tokenizer = tokenizer
                self.hidden_size = hidden_size

        extractor = _ExtractorStub()
    else:
        extractor = create_extractor(
            backend=backend,
            model_name=subject_model,
            layers=layers,
            token_positions="decision",
        )

    raw_extractor = RawActivationExtractor(
        base_extractor=extractor,
    )

    layer_dirs = raw_extractor.extract_to_shards(
        pairs=pairs,
        output_dir=output_dir,
        layer_keys=layer_keys,
        batch_size=batch_size,
        shard_size=shard_size,
        scenarios_meta=scenarios_meta,
        dp_size=dp_size,
    )

    if hasattr(extractor, "cleanup"):
        extractor.cleanup()
    return layer_dirs


# ---------------------------------------------------------------------------
# Step 2: Train JumpReLU SAE
# ---------------------------------------------------------------------------


def train_sae_step(
    activations_dir: str,
    checkpoint_dir: str,
    d_sae: int,
    learning_rate: float,
    batch_size: int,
    l1_coefficient: float,
    total_steps: int | None,
    num_epochs: int,
    resume_from: str | None,
    auto_scale_steps: bool = True,
) -> str:
    """Train a JumpReLU SAE on the numpy activation shards from Step 1."""
    from sae.trainer import SAETrainingConfig, train_sae

    config = SAETrainingConfig(
        d_sae=d_sae,
        batch_size=batch_size,
        learning_rate=learning_rate,
        l1_coefficient=l1_coefficient,
        total_steps=total_steps,
        num_epochs=num_epochs,
        output_dir=checkpoint_dir,
        resume_from=resume_from,
        auto_scale_steps=auto_scale_steps,
    )

    return train_sae(activations_dir=activations_dir, config=config)


# ---------------------------------------------------------------------------
# Step 3: Contrastive feature identification (see analysis/contrastive_features.py)
# ---------------------------------------------------------------------------

from analysis.contrastive_features import identify_contrastive_features  # noqa: E402

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_pairs(pairs_dir: str) -> list:
    from data.contrastive_dataset import ContrastiveDataset

    src_path = Path(pairs_dir)
    if not src_path.exists() or not list(src_path.glob("shard_*.parquet")):
        print(f"  ERROR: no pair parquet shards found in {pairs_dir}")
        print("  Run generate_pairs first: uv run python -m generate_pairs <num_samples>")
        sys.exit(1)
    dataset = ContrastiveDataset.from_parquet(pairs_dir)
    pairs = dataset.pairs
    print(f"  Loaded {len(pairs)} pairs from {pairs_dir}")
    return pairs


def _run_step1(args, pairs_dir: str) -> dict[str, Path]:
    pairs = _load_pairs(pairs_dir)

    from data.scenario import load_scenarios_meta

    scenarios_meta = load_scenarios_meta(Path(pairs_dir))
    scenario_names = list(scenarios_meta.keys())

    print("\n[Step 1] Extracting raw activations to numpy shards...")
    print(f"  Output: {args.output_dir}")
    print(f"  Layers: {args.layers}")
    print(f"  Scenarios: {', '.join(scenario_names)}")
    print("  Each pair -> 2 activation vectors (anchor + contrast)")
    print(f"  Total prompts: {len(pairs) * 2}")
    t0 = time.time()
    layer_dirs = extract_activations(
        pairs=pairs,
        output_dir=args.output_dir,
        subject_model=args.subject_model,
        layers=args.layers,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        scenarios_meta=scenarios_meta,
        backend=args.backend,
        dp_size=args.extraction_dp_size,
    )
    elapsed = time.time() - t0
    print(f"  Extraction complete ({elapsed:.1f}s)")
    return layer_dirs


def _run_step2(args) -> dict[str, str]:
    """Train one SAE per layer. Returns {layer_key: checkpoint_path}."""
    print(f"\n[Step 2] Training JumpReLU SAEs for {len(args.layers)} layer(s)...")
    print(f"  d_sae: {args.d_sae}")

    sae_checkpoints: dict[str, str] = {}
    t0_total = time.time()

    for layer in args.layers:
        layer_key = f"residual_{layer}"
        layer_dir = Path(args.output_dir) / f"layer_{layer}"
        activations_dir = str(layer_dir / "activations")
        checkpoint_dir = args.sae_checkpoint_dir or str(layer_dir / "sae_checkpoints")

        print(f"\n  [Layer {layer}] Training SAE...")
        print(f"    Activations: {activations_dir}")
        print(f"    Checkpoints: {checkpoint_dir}")
        t0 = time.time()
        final_path = train_sae_step(
            activations_dir=activations_dir,
            checkpoint_dir=checkpoint_dir,
            d_sae=args.d_sae,
            learning_rate=args.sae_lr,
            batch_size=args.sae_batch_size,
            l1_coefficient=args.l1_coefficient,
            total_steps=args.sae_steps,
            num_epochs=args.sae_epochs,
            resume_from=args.sae_resume,
            auto_scale_steps=not args.no_auto_scale_steps,
        )
        elapsed = time.time() - t0
        print(f"    SAE training complete ({elapsed:.1f}s): {final_path}")
        sae_checkpoints[layer_key] = final_path

    elapsed_total = time.time() - t0_total
    print(f"\n  All SAEs trained ({elapsed_total:.1f}s)")
    return sae_checkpoints


def _run_step3(args, pairs_dir: str, sae_checkpoints: dict[str, str] | None = None) -> None:
    checkpoints = sae_checkpoints or _resolve_sae_checkpoints(args)

    pairs = _load_pairs(pairs_dir)

    from data.scenario import load_scenarios_meta

    scenarios_meta = load_scenarios_meta(Path(pairs_dir))

    print(f"\n[Step 3] Identifying contrastive SAE features for {len(args.layers)} layer(s)...")
    print(f"  Scenarios: {', '.join(scenarios_meta.keys())}")
    t0 = time.time()
    identify_contrastive_features(
        pairs=pairs,
        sae_checkpoints=checkpoints,
        subject_model=args.subject_model,
        layers=args.layers,
        batch_size=args.batch_size,
        top_k=args.top_k_features,
        base_output_dir=args.output_dir,
        min_effect_size=args.min_effect_size,
        min_activation=args.min_activation,
        scenarios_meta=scenarios_meta,
        backend=args.backend,
    )
    elapsed = time.time() - t0
    print(f"  Feature identification complete ({elapsed:.1f}s)")


def _resolve_sae_checkpoints(args) -> dict[str, str]:
    """Resolve SAE checkpoint paths for all layers.

    Returns:
        Dict mapping layer_key (e.g. "residual_20") to checkpoint path.
    """
    checkpoints: dict[str, str] = {}
    for layer in args.layers:
        layer_key = f"residual_{layer}"
        if args.sae_checkpoint:
            # Single explicit checkpoint — only valid for single-layer
            checkpoints[layer_key] = args.sae_checkpoint
        else:
            layer_dir = Path(args.output_dir) / f"layer_{layer}"
            default_path = (
                Path(args.sae_checkpoint_dir or str(layer_dir / "sae_checkpoints")) / "sae_final.pt"
            )
            if default_path.exists():
                checkpoints[layer_key] = str(default_path)
            else:
                print(f"\n  ERROR: SAE checkpoint not found for layer {layer}.")
                print(f"  Expected at: {default_path}")
                print("  Run step 2 first, or provide --sae-checkpoint.")
                sys.exit(1)
    return checkpoints


def _resolve_contrastive_features(output_dir: str | Path, layer: int) -> str:
    """Find the contrastive_features.json for a specific layer."""
    layer_dir = Path(output_dir) / f"layer_{layer}"
    path = layer_dir / "activations" / "contrastive_features.json"
    if path.exists():
        return str(path)
    print(f"\n  ERROR: contrastive_features.json not found for layer {layer}.")
    print(f"  Expected at: {path}")
    print("  Run step 3 first.")
    sys.exit(1)


def _run_step4(args, sae_checkpoints: dict[str, str] | None = None) -> None:
    """Step 4: Per-layer feature interpretation.

    For each layer: load activation shards, encode through that layer's SAE,
    label features via LLM subprocess, generate report.
    """
    from analysis.feature_interpreter import (
        collect_max_activating_examples,
        generate_explanation_report,
        label_features_via_llm,
        load_activations_from_shards,
    )

    checkpoints = sae_checkpoints or _resolve_sae_checkpoints(args)

    print(f"\n[Step 4] Interpreting features for {len(args.layers)} layer(s)...")
    t0_total = time.time()

    for layer in args.layers:
        layer_key = f"residual_{layer}"
        layer_dir = Path(args.output_dir) / f"layer_{layer}"
        activations_dir = str(layer_dir / "activations")
        checkpoint = checkpoints[layer_key]
        contrastive_path = _resolve_contrastive_features(args.output_dir, layer)

        print(f"\n  --- Layer {layer} ---")

        # Determine which features to analyze from contrastive_features.json
        with open(contrastive_path) as f:
            contrastive = json.load(f)

        feature_indices_set: set[int] = set()
        for ct_info in contrastive.values():
            for feat in ct_info.get("top_features", []):
                feature_indices_set.add(feat["feature_index"])
        feature_indices = sorted(feature_indices_set)
        print(f"  {len(feature_indices)} unique features to interpret")

        # 4a: Load activations from Step 1 numpy shards
        print(f"\n  [Layer {layer}] 4a: Loading activations from numpy shards...")
        t0 = time.time()
        prompts, activations = load_activations_from_shards(
            activations_dir=activations_dir,
        )
        elapsed = time.time() - t0
        print(
            f"    4a complete ({elapsed:.1f}s): {len(prompts)} prompts, shape {activations.shape}"
        )

        # 4b: Encode through SAE, collect examples
        print(f"\n  [Layer {layer}] 4b: Encoding through SAE...")
        t0 = time.time()
        feature_examples = collect_max_activating_examples(
            prompts=prompts,
            activations=activations,
            sae_checkpoint=checkpoint,
            feature_indices=feature_indices,
            top_n=args.label_top_n,
            bottom_n=args.label_bottom_n,
        )
        elapsed = time.time() - t0
        print(f"    4b complete ({elapsed:.1f}s): {len(feature_examples)} features analyzed")

        del activations

        # 4c: Label features via LLM (subprocess)
        print(f"\n  [Layer {layer}] 4c: Labeling features via LLM (subprocess)...")
        t0 = time.time()
        feature_labels = label_features_via_llm(
            feature_examples=feature_examples,
            judging_model=args.judging_model,
            tp_size=args.generation_tp_size,
            dp_size=args.generation_dp_size,
            max_model_len=args.max_model_len,
            output_dir=activations_dir,
        )
        elapsed = time.time() - t0
        print(f"    4c complete ({elapsed:.1f}s): {len(feature_labels)} features labeled")

        # 4d: Generate report
        print(f"\n  [Layer {layer}] 4d: Generating explanation report...")
        t0 = time.time()
        report_path = generate_explanation_report(
            contrastive_features_path=contrastive_path,
            feature_examples=feature_examples,
            feature_labels=feature_labels,
            output_dir=activations_dir,
        )
        elapsed = time.time() - t0
        print(f"    4d complete ({elapsed:.1f}s): {report_path}")

    elapsed_total = time.time() - t0_total
    print(f"\n  Step 4 complete for all layers ({elapsed_total:.1f}s)")


def _resolve_feature_descriptions(output_dir: str | Path, layer: int) -> str:
    """Find the feature_descriptions.json for a specific layer."""
    layer_dir = Path(output_dir) / f"layer_{layer}"
    candidates = [
        layer_dir / "activations" / "feature_descriptions.json",
        layer_dir / "feature_descriptions.json",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    print(f"\n  ERROR: feature_descriptions.json not found for layer {layer}.")
    print(f"  Searched: {[str(p) for p in candidates]}")
    print("  Run step 4 first.")
    sys.exit(1)


def _run_step5(args, pairs_dir: str, sae_checkpoints: dict[str, str] | None = None) -> None:
    """Step 5: Evaluate feature explanations via fuzzing.

    Loads subject model once for per-token extraction, then loops per-layer
    for SAE encoding, fuzzing example building, and LLM judging.
    """
    from analysis.fuzzing_evaluator import (
        build_fuzzing_examples,
        compute_fuzzing_metrics,
        evaluate_fuzzing,
        extract_per_token_activations,
        save_fuzzing_report,
    )
    from data.scenario import default_scenario, load_scenarios_meta
    from extraction.extractor import build_agent_prompt

    checkpoints = sae_checkpoints or _resolve_sae_checkpoints(args)

    # Load scenarios and pairs (shared across layers)
    scenarios_meta = load_scenarios_meta(Path(pairs_dir))
    _default_scenario = default_scenario()

    pairs = _load_pairs(pairs_dir)
    prompt_to_scenario: dict[str, str] = {}
    for pair in pairs:
        prompt_to_scenario[pair.anchor_prompt] = pair.scenario_name
        prompt_to_scenario[pair.contrast_prompt] = pair.scenario_name

    print(f"\n[Step 5] Fuzzing evaluation for {len(args.layers)} layer(s)")
    print(f"  Scenarios: {', '.join(scenarios_meta.keys())}")

    # Collect all unique prompts across ALL layers' feature descriptions
    all_prompts: list[str] = []
    seen: set[str] = set()
    per_layer_descs: dict[int, dict] = {}

    for layer in args.layers:
        desc_path = _resolve_feature_descriptions(args.output_dir, layer)
        with open(desc_path) as f:
            feature_descriptions = json.load(f)
        per_layer_descs[layer] = feature_descriptions

        for desc in feature_descriptions.values():
            for prompt in desc.get("top_examples", []) + desc.get("bottom_examples", []):
                if prompt not in seen:
                    seen.add(prompt)
                    all_prompts.append(prompt)

    print(f"  Unique prompts across all layers: {len(all_prompts)}")

    # Load tokenizer once for prompt formatting and text reconstruction
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.subject_model,
        trust_remote_code=True,
    )

    # Build formatted prompts using subject model's chat template
    formatted_prompts = []
    for req in all_prompts:
        scenario_name = prompt_to_scenario.get(req, "")
        scenario = scenarios_meta.get(scenario_name, _default_scenario)
        formatted_prompts.append(
            build_agent_prompt(
                system_prompt=scenario.system_prompt,
                tools=scenario.tools,
                user_request=req,
                tokenizer=tokenizer,
            )
        )

    # 5a: Extract per-token activations ONCE for all layers
    print("\n[Step 5a] Extracting per-token activations (shared across layers)...")
    t0 = time.time()
    token_strings_list, token_activations_list = extract_per_token_activations(
        prompts=all_prompts,
        formatted_prompts=formatted_prompts,
        subject_model=args.subject_model,
        layers=args.layers,
        batch_size=args.fuzz_batch_size,
        backend=args.backend,
        dp_size=args.extraction_dp_size,
    )
    elapsed = time.time() - t0
    print(f"  5a complete ({elapsed:.1f}s): {len(token_strings_list)} prompts")

    prompt_to_idx = {p: i for i, p in enumerate(all_prompts)}

    t0_total = time.time()

    # Per-layer: build fuzzing examples, judge, compute metrics
    for layer in args.layers:
        layer_key = f"residual_{layer}"
        layer_dir = Path(args.output_dir) / f"layer_{layer}"
        activations_dir = str(layer_dir / "activations")
        checkpoint = checkpoints[layer_key]
        feature_descriptions = per_layer_descs[layer]

        print(f"\n  --- Layer {layer} ---")
        print(f"  {len(feature_descriptions)} features to evaluate")

        # 5b: Build fuzzing examples for this layer's SAE
        print(f"\n  [Layer {layer}] 5b: Building fuzzing examples...")
        t0 = time.time()
        examples = build_fuzzing_examples(
            feature_descriptions=feature_descriptions,
            prompt_to_idx=prompt_to_idx,
            token_strings_list=token_strings_list,
            token_activations_list=token_activations_list,
            layer_key=layer_key,
            sae_checkpoint=checkpoint,
            tokenizer=tokenizer,
            top_k_tokens=args.fuzz_top_k_tokens,
            max_examples_per_feature=args.fuzz_examples_per_feature,
        )
        elapsed = time.time() - t0
        print(f"    5b complete ({elapsed:.1f}s): {len(examples)} examples")

        # 5c: LLM judge (subprocess) — one call per layer
        print(f"\n  [Layer {layer}] 5c: Running LLM judge (subprocess)...")
        t0 = time.time()
        results = evaluate_fuzzing(
            examples=examples,
            feature_descriptions=feature_descriptions,
            judging_model=args.judging_model,
            tp_size=args.generation_tp_size,
            dp_size=args.generation_dp_size,
            max_model_len=args.max_model_len,
            output_dir=activations_dir,
        )
        elapsed = time.time() - t0
        print(f"    5c complete ({elapsed:.1f}s): {len(results)} judgments")

        # 5d: Compute metrics and save
        print(f"\n  [Layer {layer}] 5d: Computing metrics and saving report...")
        t0 = time.time()
        per_feature, summary = compute_fuzzing_metrics(results, feature_descriptions)
        save_fuzzing_report(per_feature, summary, results, activations_dir)
        elapsed = time.time() - t0
        print(f"    5d complete ({elapsed:.1f}s)")

    # Free per-token data
    del token_strings_list, token_activations_list
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed_total = time.time() - t0_total
    print(f"\n  Step 5 complete for all layers ({elapsed_total:.1f}s)")


def main() -> None:
    args = parse_args()

    # "all" runs steps 1 through 5
    if args.step == "all":
        steps = ["1", "2", "3", "4", "5"]
    else:
        steps = [args.step]

    # Auto-detect layer/d_sae defaults from model config
    _resolve_model_defaults(args)

    pairs_dir = args.pairs_dir

    # Load scenarios metadata from pairs directory for display
    from data.scenario import load_scenarios_meta

    scenarios_meta = load_scenarios_meta(Path(pairs_dir))
    scenario_names = list(scenarios_meta.keys())
    total_contrast_types = sum(len(s.contrast_types) for s in scenarios_meta.values())

    print("=" * 60)
    print("  Contrastive SAE Pipeline")
    print(f"  Output directory  : {args.output_dir}")
    print(f"  Pairs directory   : {pairs_dir}")
    print(f"  Scenarios         : {', '.join(scenario_names)}")
    print(
        f"  Contrast types    : {total_contrast_types} total across {len(scenarios_meta)} scenario(s)"
    )
    print(f"  Steps to run      : {', '.join(steps)}")
    if "1" in steps:
        print(f"  Subject model     : {args.subject_model}")
        print(f"  Extraction backend: {args.backend}")
        print(f"  Extraction DP size: {args.extraction_dp_size}")
        print(f"  Layers            : {args.layers}")
        print(f"  GPU batch size    : {args.batch_size}")
        print(f"  Shard size        : {args.shard_size:,} vectors/shard")
    if "2" in steps:
        print(f"  SAE d_sae         : {args.d_sae}")
        print(f"  SAE batch size    : {args.sae_batch_size:,}")
        print(f"  SAE lr            : {args.sae_lr}")
        print(f"  SAE L1 coeff      : {args.l1_coefficient}")
    if "3" in steps:
        print(f"  SAE checkpoint    : {args.sae_checkpoint or '(auto from step 2)'}")
        print(f"  Top-K features    : {args.top_k_features}")
    if "4" in steps:
        print(f"  SAE checkpoint    : {args.sae_checkpoint or '(auto from step 2)'}")
        print(f"  Label top-N       : {args.label_top_n}")
        print(f"  Label bottom-N    : {args.label_bottom_n}")
        print(
            f"  Labeling model    : {args.judging_model} (vLLM, TP={args.generation_tp_size}, DP={args.generation_dp_size})"
        )
    if "5" in steps:
        print(f"  SAE checkpoint    : {args.sae_checkpoint or '(auto from step 2)'}")
        print(f"  Fuzz top-K tokens : {args.fuzz_top_k_tokens}")
        print(f"  Fuzz examples/feat: {args.fuzz_examples_per_feature}")
        print(f"  Fuzz batch size   : {args.fuzz_batch_size}")
        print(
            f"  Judge model       : {args.judging_model} (vLLM, TP={args.generation_tp_size}, DP={args.generation_dp_size})"
        )
    print("=" * 60)

    sae_checkpoints: dict[str, str] | None = None

    if "1" in steps:
        _run_step1(args, pairs_dir)

    if "2" in steps:
        sae_checkpoints = _run_step2(args)

    if "3" in steps:
        _run_step3(args, pairs_dir, sae_checkpoints=sae_checkpoints)

    if "4" in steps:
        _run_step4(args, sae_checkpoints=sae_checkpoints)

    if "5" in steps:
        _run_step5(args, pairs_dir, sae_checkpoints=sae_checkpoints)

    print("\nDone.")


if __name__ == "__main__":
    main()
