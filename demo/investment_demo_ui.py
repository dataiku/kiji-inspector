"""
Investment Decision Demo with Rich Terminal UI.

A beautiful terminal interface showing:
  - Live agent state and progress
  - Step-by-step analysis with activation insights
  - Final explanations in formatted panels

Usage:
    uv run python demo/investment_demo_ui.py
    uv run python demo/investment_demo_ui.py --sae-checkpoint output/sae_checkpoints/sae_final.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

# Add src/ and demo/ to path for project imports
src_path = Path(__file__).resolve().parent.parent / "src"
demo_path = Path(__file__).resolve().parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(demo_path) not in sys.path:
    sys.path.insert(0, str(demo_path))

console = Console()

# ---------------------------------------------------------------------------
# Mock tool results (same as original demo)
# ---------------------------------------------------------------------------

MOCK_TOOL_RESULTS = {
    "NVDA": {
        "MarketData": {
            "price": 875.50,
            "volume": "45.2M",
            "pe_ratio": 65.3,
            "market_cap": "2.15T",
            "52w_high": 950.00,
            "52w_low": 450.25,
            "avg_volume": "38.5M",
        },
        "Financials": {
            "revenue_growth": "122%",
            "gross_margin": "74.5%",
            "net_margin": "55.2%",
            "debt_to_equity": 0.41,
            "free_cash_flow": "28.5B",
            "eps_growth": "145%",
        },
        "Risk": {
            "beta": 1.72,
            "volatility": "42%",
            "sharpe_ratio": 1.85,
            "max_drawdown": "-25%",
            "var_95": "-4.2%",
        },
        "News": {
            "sentiment": "very_positive",
            "recent_headlines": [
                "NVIDIA announces next-gen AI chips",
                "Data center revenue exceeds expectations",
                "New partnerships with major cloud providers",
            ],
        },
    },
    "TSLA": {
        "MarketData": {
            "price": 245.80,
            "volume": "125.3M",
            "pe_ratio": 72.1,
            "market_cap": "780B",
            "52w_high": 300.00,
            "52w_low": 138.80,
            "avg_volume": "98.2M",
        },
        "Financials": {
            "revenue_growth": "18%",
            "gross_margin": "18.2%",
            "net_margin": "11.4%",
            "debt_to_equity": 0.08,
            "free_cash_flow": "4.2B",
            "eps_growth": "-23%",
        },
        "Risk": {
            "beta": 2.05,
            "volatility": "58%",
            "sharpe_ratio": 0.92,
            "max_drawdown": "-45%",
            "var_95": "-6.8%",
        },
        "News": {
            "sentiment": "mixed",
            "recent_headlines": [
                "Tesla price cuts pressure margins",
                "Cybertruck production ramps up",
                "Competition intensifies in EV market",
            ],
        },
    },
    "AAPL": {
        "MarketData": {
            "price": 185.20,
            "volume": "52.1M",
            "pe_ratio": 28.5,
            "market_cap": "2.85T",
            "52w_high": 199.62,
            "52w_low": 164.08,
            "avg_volume": "48.3M",
        },
        "Financials": {
            "revenue_growth": "2%",
            "gross_margin": "44.1%",
            "net_margin": "25.3%",
            "debt_to_equity": 1.52,
            "free_cash_flow": "99.6B",
            "eps_growth": "8%",
        },
        "Risk": {
            "beta": 1.28,
            "volatility": "24%",
            "sharpe_ratio": 1.45,
            "max_drawdown": "-18%",
            "var_95": "-2.8%",
        },
        "News": {
            "sentiment": "positive",
            "recent_headlines": [
                "iPhone sales remain strong in emerging markets",
                "Services revenue hits new record",
                "Apple Vision Pro receives mixed reviews",
            ],
        },
    },
}

TOOLS = ["MarketData", "Financials", "Risk", "News"]
TICKERS = ["NVDA", "TSLA", "AAPL"]


def build_analysis_prompt(ticker: str, tool: str, tool_result: dict) -> str:
    """Build a ChatML prompt for analyzing tool results."""
    system = (
        "You are an investment analyst. Analyze the provided data concisely. "
        "Focus on key insights relevant to investment decisions."
    )
    user = f"""Analyze this {tool} data for {ticker}:

{json.dumps(tool_result, indent=2)}

Provide a brief analysis (2-3 sentences) focusing on investment implications."""

    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_final_prompt(all_analyses: dict[str, str]) -> str:
    """Build prompt for final allocation decision."""
    system = (
        "You are a portfolio manager. Based on the analyses provided, "
        "recommend a portfolio allocation across the three stocks. "
        "Provide specific percentages and brief justification."
    )

    analyses_text = "\n\n".join(
        f"### {step}\n{analysis}" for step, analysis in all_analyses.items()
    )

    user = f"""Based on these analyses, recommend a portfolio allocation:

{analyses_text}

Provide allocation percentages for NVDA, TSLA, and AAPL that sum to 100%."""

    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------


def create_header() -> Panel:
    """Create the header panel."""
    title = Text()
    title.append("Investment Analysis Demo", style="bold cyan")
    title.append(" with ", style="dim")
    title.append("SAE Explainability", style="bold magenta")
    return Panel(title, style="blue", padding=(0, 2))


def create_agent_state_panel(
    current_phase: str,
    current_step: str,
    completed_steps: list[str],
    ticker_progress: dict[str, str],
) -> Panel:
    """Create a panel showing current agent state."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Label", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Phase:", current_phase)
    table.add_row("Current Step:", current_step)
    table.add_row("Completed:", str(len(completed_steps)))

    # Ticker status icons
    ticker_status = []
    for ticker in TICKERS:
        status = ticker_progress.get(ticker, "pending")
        if status == "done":
            ticker_status.append(f"[green]{ticker}[/green]")
        elif status == "active":
            ticker_status.append(f"[yellow]{ticker}[/yellow]")
        else:
            ticker_status.append(f"[dim]{ticker}[/dim]")
    table.add_row("Tickers:", " | ".join(ticker_status))

    return Panel(table, title="[bold]Agent State[/bold]", border_style="green")


def create_steps_table(steps_data: list[dict]) -> Table:
    """Create a table showing completed analysis steps."""
    table = Table(title="Analysis Steps", show_lines=True)
    table.add_column("Step", style="cyan", width=20)
    table.add_column("Tokens", justify="right", width=8)
    table.add_column("Top Feature", width=35)
    table.add_column("Activation", justify="right", width=10)

    for step in steps_data[-8:]:  # Show last 8 steps
        top_feat = step.get("top_feature", "-")
        top_act = step.get("top_activation", "-")
        if isinstance(top_act, float):
            top_act = f"{top_act:.2f}"
        table.add_row(
            step["name"],
            str(step.get("tokens", "-")),
            top_feat[:33] + "..." if len(str(top_feat)) > 35 else str(top_feat),
            str(top_act),
        )

    return table


def create_feature_panel(features: list[tuple[str, float]]) -> Panel:
    """Create a panel showing top activated features."""
    if not features:
        return Panel("[dim]No SAE features available[/dim]", title="Top Features")

    lines = []
    for name, activation in features[:5]:
        bar_len = int(activation * 3)
        bar = "[cyan]" + "█" * bar_len + "[/cyan]"
        lines.append(f"{bar} {activation:.2f}  {name[:40]}")

    return Panel("\n".join(lines), title="[bold]Top SAE Features[/bold]", border_style="magenta")


# ---------------------------------------------------------------------------
# SAE Analysis Functions
# ---------------------------------------------------------------------------


def extract_activations_for_prompts(
    prompt_log: list[tuple[str, str]],
    layers: list[int],
    console: Console,
) -> list[tuple[str, dict[str, np.ndarray]]]:
    """Load HF model and extract activations for all recorded prompts."""
    from activation_extractor import ActivationConfig, ActivationExtractor

    console.print("[dim]Loading HF model for activation extraction...[/dim]")
    config = ActivationConfig(layers=layers)
    extractor = ActivationExtractor(config=config)

    activation_log = []
    with console.status("[bold cyan]Extracting activations...[/bold cyan]") as status:
        for i, (step_label, prompt_text) in enumerate(prompt_log):
            status.update(
                f"[bold cyan]Extracting activations ({i + 1}/{len(prompt_log)})...[/bold cyan]"
            )
            acts = extractor.extract(prompt_text, decision_token_offset=-1)
            activation_log.append((step_label, acts))

    extractor.cleanup()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return activation_log


def analyze_with_sae(
    activation_log: list[tuple[str, dict[str, np.ndarray]]],
    sae_checkpoint: str,
    feature_descriptions_path: str,
    layer_key: str = "residual_20",
) -> dict:
    """Encode captured activations through SAE and map to feature descriptions."""
    from sae_model import JumpReLUSAE

    results = {"steps": [], "sae_available": False, "features_available": False}

    # Tier 1: Raw activation statistics
    for step_label, acts in activation_log:
        step_info = {
            "step": step_label,
            "layers_captured": list(acts.keys()),
            "raw_stats": {},
        }
        for layer_name, vec in acts.items():
            step_info["raw_stats"][layer_name] = {
                "mean": float(np.mean(vec)),
                "std": float(np.std(vec)),
                "l2_norm": float(np.linalg.norm(vec)),
                "max_abs": float(np.max(np.abs(vec))),
                "sparsity": float(np.mean(np.abs(vec) < 0.01)),
            }
        results["steps"].append(step_info)

    # Tier 2: SAE feature decomposition
    if not Path(sae_checkpoint).exists():
        return results

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = JumpReLUSAE.from_pretrained(sae_checkpoint, device=device)
    sae.eval()
    results["sae_available"] = True
    sae_dtype = next(sae.parameters()).dtype

    for step_info, (step_label, acts) in zip(results["steps"], activation_log):
        if layer_key not in acts:
            continue
        vec = acts[layer_key]
        vec_tensor = torch.from_numpy(vec).unsqueeze(0).to(device=device, dtype=sae_dtype)
        with torch.no_grad():
            features = sae.encode(vec_tensor)
        features_np = features.squeeze(0).cpu().float().numpy()

        nonzero_mask = features_np > 0
        nonzero_indices = np.where(nonzero_mask)[0]
        nonzero_values = features_np[nonzero_indices]

        sort_order = np.argsort(-nonzero_values)
        top_k = min(20, len(sort_order))
        top_indices = nonzero_indices[sort_order[:top_k]]
        top_values = nonzero_values[sort_order[:top_k]]

        step_info["sae_features"] = {
            "num_active": int(nonzero_mask.sum()),
            "total_features": int(features_np.shape[0]),
            "sparsity_pct": float((1.0 - nonzero_mask.mean()) * 100),
            "top_features": [
                {"index": int(idx), "activation": float(val)}
                for idx, val in zip(top_indices, top_values)
            ],
        }

    del sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Tier 3: Feature label mapping
    if not Path(feature_descriptions_path).exists():
        return results

    with open(feature_descriptions_path) as f:
        feature_descs = json.load(f)

    results["features_available"] = True

    for step_info in results["steps"]:
        if "sae_features" not in step_info:
            continue
        for feat in step_info["sae_features"]["top_features"]:
            desc = feature_descs.get(str(feat["index"]))
            if desc:
                feat["label"] = desc.get("label", "unknown")
                feat["description"] = desc.get("description", "")
                feat["confidence"] = desc.get("confidence", "low")

    return results


# ---------------------------------------------------------------------------
# Main Demo with UI
# ---------------------------------------------------------------------------


def run_demo_with_ui(args):
    """Run the investment demo with rich terminal UI."""
    from vllm import LLM, SamplingParams

    # State tracking
    prompt_log = []
    completed_steps = []
    steps_data = []
    all_analyses = {}
    current_phase = "Initialization"
    current_step = "Loading models"
    ticker_progress = {t: "pending" for t in TICKERS}
    top_features = []

    # Create progress bars
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    total_steps = len(TICKERS) * len(TOOLS) + 1  # +1 for final allocation
    main_task = progress.add_task("[cyan]Running analysis...", total=total_steps)

    def update_display():
        """Create the current display layout."""
        # Build bottom panel content
        if steps_data:
            bottom_content = create_steps_table(steps_data)
        else:
            bottom_content = Panel(
                "[dim]Waiting for analysis steps...[/dim]", title="Analysis Steps"
            )

        # Build layout from scratch each time
        layout = Layout()
        layout.split_column(
            Layout(create_header(), name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(bottom_content, name="bottom", size=12),
        )

        layout["main"].split_row(
            Layout(
                create_agent_state_panel(
                    current_phase, current_step, completed_steps, ticker_progress
                ),
                name="state",
                ratio=1,
            ),
            Layout(create_feature_panel(top_features), name="features", ratio=2),
        )

        return Group(layout, progress)

    # Phase 1: Run vLLM generation
    console.print("[dim]Initializing vLLM engine...[/dim]")
    llm = LLM(
        model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=300,
        repetition_penalty=1.1,
    )

    # Run analysis with live display
    with Live(update_display(), console=console, refresh_per_second=4) as live:
        current_phase = "Data Analysis"

        for ticker in TICKERS:
            ticker_progress[ticker] = "active"

            for tool in TOOLS:
                step_name = f"{ticker}_{tool}"
                current_step = step_name

                # Build and run prompt
                tool_result = MOCK_TOOL_RESULTS[ticker][tool]
                prompt = build_analysis_prompt(ticker, tool, tool_result)
                prompt_log.append((step_name, prompt))

                live.update(update_display())

                outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
                response = outputs[0].outputs[0].text
                n_tokens = len(outputs[0].outputs[0].token_ids)

                all_analyses[step_name] = response
                completed_steps.append(step_name)

                # Placeholder step info (will be updated after SAE analysis)
                step_info = {
                    "name": step_name,
                    "tokens": n_tokens,
                    "top_feature": "[pending SAE]",
                    "top_activation": "-",
                }
                steps_data.append(step_info)
                progress.update(main_task, advance=1)
                live.update(update_display())

            ticker_progress[ticker] = "done"

        # Final allocation
        current_phase = "Final Decision"
        current_step = "Portfolio Allocation"
        live.update(update_display())

        final_prompt = build_final_prompt(all_analyses)
        prompt_log.append(("final_allocation", final_prompt))

        final_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=500,
            repetition_penalty=1.1,
        )
        outputs = llm.generate([final_prompt], final_params, use_tqdm=False)
        final_recommendation = outputs[0].outputs[0].text

        completed_steps.append("final_allocation")
        steps_data.append(
            {
                "name": "Final Allocation",
                "tokens": len(outputs[0].outputs[0].token_ids),
                "top_feature": "[pending SAE]",
                "top_activation": "-",
            }
        )
        progress.update(main_task, advance=1)
        current_phase = "Complete"
        current_step = "Done"
        live.update(update_display())

    # Cleanup vLLM to free GPU memory
    console.print("[dim]Releasing vLLM engine...[/dim]")
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 2: Extract activations via HF transformers
    console.print()
    console.print(Panel("[bold cyan]Phase 2: Extracting Activations[/bold cyan]", style="cyan"))

    activation_log = extract_activations_for_prompts(prompt_log, args.layers, console)

    # Phase 3: Analyze with SAE
    console.print()
    console.print(Panel("[bold magenta]Phase 3: SAE Analysis[/bold magenta]", style="magenta"))

    analysis = analyze_with_sae(
        activation_log=activation_log,
        sae_checkpoint=args.sae_checkpoint,
        feature_descriptions_path=args.feature_descriptions,
    )

    # Update steps_data with real SAE features
    for i, step_info in enumerate(analysis["steps"]):
        if i < len(steps_data):
            sae_feats = step_info.get("sae_features")
            if sae_feats and sae_feats["top_features"]:
                top = sae_feats["top_features"][0]
                steps_data[i]["top_feature"] = top.get("label", f"Feature #{top['index']}")
                steps_data[i]["top_activation"] = top["activation"]
            else:
                steps_data[i]["top_feature"] = "-"
                steps_data[i]["top_activation"] = "-"

    # Build top_features list for display
    if analysis["steps"] and analysis["steps"][-1].get("sae_features"):
        sae_feats = analysis["steps"][-1]["sae_features"]["top_features"][:5]
        top_features = [
            (f.get("label", f"Feature #{f['index']}"), f["activation"]) for f in sae_feats
        ]

    # Display final results
    console.print()
    console.print(Panel("[bold green]Analysis Complete![/bold green]", style="green"))

    # Show updated table with real features
    console.print()
    console.print(create_steps_table(steps_data))

    # Show top features panel
    console.print()
    console.print(create_feature_panel(top_features))

    # Show final recommendation
    console.print()
    console.print(
        Panel(
            Markdown(final_recommendation),
            title="[bold cyan]Portfolio Recommendation[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    # Phase 4: Generate explanations
    if analysis["features_available"]:
        console.print()
        console.print(
            Panel("[bold yellow]Phase 4: Generating Explanations[/bold yellow]", style="yellow")
        )

        from investment_demo import generate_decision_explanations

        technical, layman = generate_decision_explanations(
            analysis_results=analysis,
            agent_output=final_recommendation,
            tp_size=args.tp_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        console.print()
        console.print(
            Panel(
                Markdown(technical),
                title="[bold magenta]Technical Explanation[/bold magenta]",
                border_style="magenta",
                padding=(1, 2),
            )
        )

        console.print()
        console.print(
            Panel(
                Markdown(layman),
                title="[bold yellow]Plain Language Summary[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    # Save results
    output_path = Path("demo/output")
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "agent_output.txt", "w") as f:
        f.write(final_recommendation)
    with open(output_path / "all_analyses.json", "w") as f:
        json.dump(all_analyses, f, indent=2)
    with open(output_path / "analysis_results.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    console.print()
    console.print(f"[dim]Results saved to {output_path}/[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Investment Demo with Rich UI")
    parser.add_argument(
        "--sae-checkpoint",
        type=str,
        default="output/sae_checkpoints/sae_final.pt",
        help="Path to trained SAE checkpoint",
    )
    parser.add_argument(
        "--feature-descriptions",
        type=str,
        default="output/activations/feature_descriptions.json",
        help="Path to feature descriptions JSON",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[8, 12, 16, 20, 24],
        help="Transformer layers to hook for activation extraction",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=4,
        help="Tensor parallel size for vLLM",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.45,
        help="GPU memory utilization for vLLM",
    )

    args = parser.parse_args()

    console.print()
    console.print(
        Panel(
            "[bold]Investment Analysis Demo[/bold]\n"
            "[dim]Analyzing NVDA, TSLA, AAPL with SAE explainability[/dim]",
            style="blue",
            padding=(1, 4),
        )
    )
    console.print()

    run_demo_with_ui(args)


if __name__ == "__main__":
    main()
