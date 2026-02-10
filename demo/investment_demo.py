"""
Investment Decision Demo with SAE Activation Analysis.

Orchestrates a multi-step investment analysis using Nemotron-3-Nano-30B:
  1. For each stock (NVDA, TSLA, AAPL), the model is presented with tool
     results and asked to analyze them -- one LLM call per (stock, tool) pair
  2. After all data is gathered, the model produces a final allocation
  3. Activations are extracted at each decision point via HF transformers
  4. Activations are mapped through a trained SAE to explain the reasoning

Uses vLLM for fast text generation, HF transformers for activation capture.

Usage:
    uv run python demo/investment_demo.py
    uv run python demo/investment_demo.py \
        --sae-checkpoint output/sae_checkpoints/sae_final.pt \
        --feature-descriptions output/activations/feature_descriptions.json
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch

# Add src/ to path for project imports
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from sae_model import JumpReLUSAE

_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

_SYSTEM_PROMPT = (
    "You are a senior investment analyst at a top hedge fund with 15 years "
    "of experience in technology sector investments. You are thorough, "
    "data-driven, and always reference specific numbers from the data provided. "
    "Keep your analysis concise but insightful."
)

_TICKERS = ["NVDA", "TSLA", "AAPL"]


# ---------------------------------------------------------------------------
# Section A: vLLM generation engine
# ---------------------------------------------------------------------------


class VLLMEngine:
    """Thin wrapper around vLLM for text generation with prompt logging."""

    def __init__(
        self,
        tp_size: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        gpu_memory_utilization: float = 0.45,
    ):
        from vllm import LLM, SamplingParams

        self.max_new_tokens = max_new_tokens
        self.prompt_log: list[tuple[str, str]] = []  # (step_label, prompt_text)

        print(f"  Loading vLLM engine: {_MODEL_NAME}")
        self._llm = LLM(
            model=_MODEL_NAME,
            tensor_parallel_size=tp_size,
            max_model_len=8192,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            dtype="bfloat16",
        )
        self._default_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_new_tokens,
            repetition_penalty=1.1,
        )
        print("  vLLM engine ready.")

    def generate(
        self,
        prompt: str,
        step_label: str,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text and log the prompt for later activation extraction."""
        from vllm import SamplingParams

        self.prompt_log.append((step_label, prompt))

        params = self._default_params
        if max_tokens is not None:
            params = SamplingParams(
                temperature=self._default_params.temperature,
                top_p=self._default_params.top_p,
                max_tokens=max_tokens,
                repetition_penalty=1.1,
            )

        outputs = self._llm.generate([prompt], params)
        response = outputs[0].outputs[0].text
        n_tokens = len(outputs[0].outputs[0].token_ids)
        print(f"  [{step_label}] Generated {n_tokens} tokens")
        return response

    def cleanup(self):
        """Release vLLM engine to free GPU memory."""
        if hasattr(self, "_llm") and self._llm is not None:
            del self._llm
            self._llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        print("  vLLM engine released, GPU memory freed.")


# ---------------------------------------------------------------------------
# Section B: Mock investment data
# ---------------------------------------------------------------------------

_MARKET_DATA = {
    "NVDA": {
        "ticker": "NVDA",
        "company": "NVIDIA Corporation",
        "price": 142.50,
        "change_pct": 2.3,
        "volume": "45.2M",
        "pe_ratio": 65.3,
        "market_cap": "3.5T",
        "52w_high": 153.13,
        "52w_low": 75.80,
        "sector": "Technology - Semiconductors",
        "dividend_yield": 0.02,
    },
    "TSLA": {
        "ticker": "TSLA",
        "company": "Tesla, Inc.",
        "price": 248.40,
        "change_pct": -1.1,
        "volume": "62.8M",
        "pe_ratio": 78.2,
        "market_cap": "795B",
        "52w_high": 299.29,
        "52w_low": 138.80,
        "sector": "Consumer Cyclical - Auto Manufacturers",
        "dividend_yield": 0.0,
    },
    "AAPL": {
        "ticker": "AAPL",
        "company": "Apple Inc.",
        "price": 213.25,
        "change_pct": 0.4,
        "volume": "38.1M",
        "pe_ratio": 33.5,
        "market_cap": "3.3T",
        "52w_high": 237.49,
        "52w_low": 164.08,
        "sector": "Technology - Consumer Electronics",
        "dividend_yield": 0.44,
    },
}

_FINANCIAL_DATA = {
    "NVDA": {
        "ticker": "NVDA",
        "revenue_growth_yoy": 122.4,
        "gross_margin": 74.6,
        "operating_margin": 61.8,
        "net_margin": 55.0,
        "debt_to_equity": 0.41,
        "current_ratio": 4.17,
        "roe": 115.7,
        "analyst_rating": "Strong Buy",
        "price_target_mean": 165.0,
        "price_target_high": 200.0,
        "price_target_low": 110.0,
        "earnings_surprise_last_4q": [12.5, 8.3, 15.1, 9.7],
    },
    "TSLA": {
        "ticker": "TSLA",
        "revenue_growth_yoy": 8.2,
        "gross_margin": 18.2,
        "operating_margin": 7.6,
        "net_margin": 7.1,
        "debt_to_equity": 0.11,
        "current_ratio": 1.73,
        "roe": 20.3,
        "analyst_rating": "Hold",
        "price_target_mean": 225.0,
        "price_target_high": 350.0,
        "price_target_low": 85.0,
        "earnings_surprise_last_4q": [-3.2, 5.1, -8.4, 2.0],
    },
    "AAPL": {
        "ticker": "AAPL",
        "revenue_growth_yoy": 4.9,
        "gross_margin": 46.6,
        "operating_margin": 31.5,
        "net_margin": 25.3,
        "debt_to_equity": 1.87,
        "current_ratio": 0.99,
        "roe": 157.4,
        "analyst_rating": "Buy",
        "price_target_mean": 235.0,
        "price_target_high": 270.0,
        "price_target_low": 180.0,
        "earnings_surprise_last_4q": [4.2, 3.8, 5.5, 2.9],
    },
}

_RISK_DATA = {
    "NVDA": {
        "ticker": "NVDA",
        "beta": 1.68,
        "volatility_30d": 42.3,
        "max_drawdown_1y": -28.5,
        "sharpe_ratio": 1.85,
        "sector_risk": "High (semiconductor cyclicality, AI capex dependency)",
        "geopolitical_risk": "Moderate (China export controls, Taiwan supply chain)",
        "concentration_risk": "High (data center revenue ~80% of total)",
        "regulatory_risk": "Moderate (antitrust scrutiny, export restrictions)",
    },
    "TSLA": {
        "ticker": "TSLA",
        "beta": 2.05,
        "volatility_30d": 55.8,
        "max_drawdown_1y": -42.1,
        "sharpe_ratio": 0.72,
        "sector_risk": "High (EV competition intensifying, margin pressure)",
        "geopolitical_risk": "High (China operations, tariff exposure)",
        "concentration_risk": "Moderate (auto + energy + AI/robotics diversification)",
        "regulatory_risk": "High (autonomous driving regulation, CEO controversy)",
    },
    "AAPL": {
        "ticker": "AAPL",
        "beta": 1.21,
        "volatility_30d": 22.1,
        "max_drawdown_1y": -15.3,
        "sharpe_ratio": 1.42,
        "sector_risk": "Low-Moderate (mature product cycles, services growth)",
        "geopolitical_risk": "Moderate (China manufacturing, India expansion)",
        "concentration_risk": "Moderate (iPhone ~50% revenue, growing services)",
        "regulatory_risk": "Moderate (App Store antitrust, EU DMA compliance)",
    },
}

_NEWS_DATA = {
    "NVDA": [
        {
            "headline": "NVIDIA Q4 Earnings Crush Expectations, Data Center Revenue Surges 409%",
            "source": "Reuters",
            "summary": "NVIDIA reported record quarterly revenue driven by unprecedented "
            "demand for AI training and inference chips. The company raised guidance "
            "for the next quarter, citing continued strong demand from hyperscalers.",
        },
        {
            "headline": "NVIDIA Announces Next-Gen Blackwell Ultra GPUs for AI Workloads",
            "source": "TechCrunch",
            "summary": "The new Blackwell Ultra architecture promises 4x performance "
            "improvement for large language model inference, with major cloud "
            "providers already placing orders.",
        },
        {
            "headline": "Analysts Warn of NVIDIA Valuation Stretch Despite Strong Fundamentals",
            "source": "Bloomberg",
            "summary": "Several Wall Street analysts note that NVIDIA's forward P/E "
            "remains elevated relative to historical semiconductor valuations, "
            "suggesting limited upside despite strong execution.",
        },
    ],
    "TSLA": [
        {
            "headline": "Tesla Deliveries Miss Estimates as Competition Intensifies in China",
            "source": "CNBC",
            "summary": "Tesla delivered fewer vehicles than expected last quarter, with "
            "Chinese competitors BYD and NIO gaining market share through aggressive "
            "pricing and new model launches.",
        },
        {
            "headline": "Tesla's Robotaxi Program Gets Regulatory Green Light in Texas",
            "source": "WSJ",
            "summary": "Texas regulators approved Tesla's autonomous ride-hailing service "
            "for limited deployment, marking a significant milestone for the company's "
            "self-driving ambitions.",
        },
        {
            "headline": "Tesla Energy Division Revenue Doubles Year-Over-Year",
            "source": "Financial Times",
            "summary": "Tesla's energy storage business showed explosive growth, with "
            "Megapack deployments exceeding expectations and providing meaningful "
            "margin contribution.",
        },
    ],
    "AAPL": [
        {
            "headline": "Apple Services Revenue Hits All-Time High, Offsetting iPhone Plateau",
            "source": "Bloomberg",
            "summary": "Apple's services segment including App Store, iCloud, and Apple TV+ "
            "generated record revenue, demonstrating the company's successful "
            "transition toward higher-margin recurring revenue streams.",
        },
        {
            "headline": "Apple Intelligence Rollout Drives Record iPhone Upgrade Cycle",
            "source": "WSJ",
            "summary": "Early data suggests Apple's AI features are driving a stronger "
            "than expected iPhone upgrade cycle, particularly in the Pro models "
            "which feature on-device AI processing.",
        },
        {
            "headline": "EU Fines Apple $2B Over App Store Practices",
            "source": "Reuters",
            "summary": "The European Commission imposed a significant fine on Apple for "
            "anti-competitive App Store practices, though analysts expect limited "
            "financial impact relative to Apple's cash reserves.",
        },
    ],
}

# Tool registry: name -> (data_source, description)
_TOOLS = {
    "MarketData": (_MARKET_DATA, "current market data"),
    "Financials": (_FINANCIAL_DATA, "fundamental financial analysis"),
    "Risk": (_RISK_DATA, "risk assessment"),
    "News": (_NEWS_DATA, "recent news and analyst reports"),
}


def _get_tool_result(tool_name: str, ticker: str) -> str:
    """Look up mock tool data for a ticker."""
    source = _TOOLS[tool_name][0]
    data = source.get(ticker)
    if data is None:
        return json.dumps({"error": f"Ticker '{ticker}' not found"})
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Section C: Scripted multi-step agent orchestration
# ---------------------------------------------------------------------------


def _build_chatml(system: str, user: str) -> str:
    """Build a ChatML prompt string."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run_investment_analysis(engine: VLLMEngine) -> tuple[str, dict[str, str]]:
    """Run the full multi-step investment analysis.

    For each ticker, calls each tool and asks the model to analyze the results.
    Then asks for a final portfolio allocation.

    Returns:
        (final_recommendation, per_stock_analyses)
    """
    per_stock_analyses: dict[str, str] = {}
    all_analyses_text = ""

    for ticker in _TICKERS:
        print(f"\n  --- Analyzing {ticker} ---")
        ticker_context = ""

        for tool_name, (_, tool_desc) in _TOOLS.items():
            # Get tool result
            tool_result = _get_tool_result(tool_name, ticker)
            step_label = f"{ticker}_{tool_name}"

            # Build prompt asking model to analyze this specific data
            user_msg = (
                f"You are analyzing {ticker} for a $1,000,000 portfolio allocation "
                f"across NVDA, TSLA, and AAPL.\n\n"
            )
            if ticker_context:
                user_msg += f"Your analysis of {ticker} so far:\n{ticker_context}\n\n"
            user_msg += (
                f"Here is the {tool_desc} for {ticker}:\n"
                f"```json\n{tool_result}\n```\n\n"
                f"Provide a brief analysis of this {tool_desc}. "
                f"Highlight key takeaways and any concerns. Be specific with numbers."
            )

            prompt = _build_chatml(_SYSTEM_PROMPT, user_msg)
            analysis = engine.generate(prompt, step_label, max_tokens=300)
            ticker_context += f"\n[{tool_name}] {analysis.strip()}\n"

        per_stock_analyses[ticker] = ticker_context
        all_analyses_text += f"\n=== {ticker} ===\n{ticker_context}\n"

    # Final allocation decision
    print("\n  --- Final Allocation Decision ---")
    final_user_msg = (
        "You have completed your analysis of NVDA, TSLA, and AAPL. "
        "Here is your full analysis:\n\n"
        f"{all_analyses_text}\n\n"
        "Now make your final portfolio allocation decision. "
        "Allocate exactly $1,000,000 across NVDA, TSLA, and AAPL.\n\n"
        "Provide:\n"
        "1. Your allocation (specific dollar amounts for each stock)\n"
        "2. The percentage breakdown\n"
        "3. Your key reasoning for each allocation\n"
        "4. The main risks of this portfolio"
    )

    final_prompt = _build_chatml(_SYSTEM_PROMPT, final_user_msg)
    final_recommendation = engine.generate(final_prompt, "final_allocation", max_tokens=800)

    return final_recommendation, per_stock_analyses


# ---------------------------------------------------------------------------
# Section D: Post-run activation extraction + SAE analysis
# ---------------------------------------------------------------------------


def extract_activations_for_prompts(
    prompt_log: list[tuple[str, str]],
    layers: list[int],
) -> list[tuple[str, dict[str, np.ndarray]]]:
    """Load HF model and extract activations for all recorded prompts."""
    from activation_extractor import ActivationConfig, ActivationExtractor

    print(f"  Loading HF model for activation extraction ({len(prompt_log)} prompts)...")
    config = ActivationConfig(layers=layers)
    extractor = ActivationExtractor(config=config)

    activation_log = []
    for step_label, prompt_text in prompt_log:
        acts = extractor.extract(prompt_text, decision_token_offset=-1)
        activation_log.append((step_label, acts))
        print(f"    {step_label}: extracted {len(acts)} layers")

    extractor.cleanup()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return activation_log


def _try_load_sae(checkpoint_path: str | None) -> JumpReLUSAE | None:
    """Attempt to load SAE, return None if not available."""
    paths_to_try = [
        checkpoint_path,
        "output/sae_checkpoints/sae_final.pt",
        "../output/sae_checkpoints/sae_final.pt",
    ]
    for p in paths_to_try:
        if p and Path(p).exists():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return JumpReLUSAE.from_pretrained(p, device=device)
    return None


def _try_load_feature_descriptions(path: str | None) -> dict | None:
    """Attempt to load feature descriptions, return None if not available."""
    paths_to_try = [
        path,
        "output/activations/feature_descriptions.json",
        "../output/activations/feature_descriptions.json",
    ]
    for p in paths_to_try:
        if p and Path(p).exists():
            with open(p) as f:
                return json.load(f)
    return None


def analyze_activations(
    activation_log: list[tuple[str, dict[str, np.ndarray]]],
    sae_checkpoint: str | None,
    feature_descriptions_path: str | None,
    layer_key: str = "residual_20",
) -> dict:
    """Encode captured activations through SAE and map to feature descriptions."""
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
    sae = _try_load_sae(sae_checkpoint)
    if sae is None:
        print("  SAE checkpoint not found -- showing raw activation stats only.")
        return results

    results["sae_available"] = True
    sae.eval()
    device = next(sae.parameters()).device
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
    feature_descs = _try_load_feature_descriptions(feature_descriptions_path)
    if feature_descs is None:
        print("  Feature descriptions not found -- showing SAE features without labels.")
        return results

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
# Section E: Explanation generation via vLLM
# ---------------------------------------------------------------------------


def _build_feature_summary(analysis_results: dict) -> str:
    """Format SAE analysis into text for the explanation prompt."""
    lines = []
    for step_info in analysis_results["steps"]:
        lines.append(f"## {step_info['step']}")

        for layer, stats in step_info.get("raw_stats", {}).items():
            if "residual_20" in layer:
                lines.append(
                    f"  L2={stats['l2_norm']:.2f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}"
                )

        sae = step_info.get("sae_features")
        if sae:
            lines.append(
                f"  Active SAE features: {sae['num_active']}/{sae['total_features']} "
                f"({100 - sae['sparsity_pct']:.1f}% active)"
            )
            for feat in sae["top_features"][:5]:
                label = feat.get("label", f"Feature #{feat['index']}")
                desc = feat.get("description", "")
                act = feat["activation"]
                line = f"    - {label} (activation={act:.4f})"
                if desc:
                    line += f": {desc}"
                lines.append(line)
        lines.append("")
    return "\n".join(lines)


def generate_decision_explanations(
    analysis_results: dict,
    agent_output: str,
    tp_size: int = 1,
    gpu_memory_utilization: float = 0.45,
) -> tuple[str, str]:
    """Generate both a technical and a layman explanation."""

    from vllm import LLM, SamplingParams

    feature_summary = _build_feature_summary(analysis_results)

    technical_prompt = _build_chatml(
        "You are an AI interpretability researcher. You have access to "
        "Sparse Autoencoder (SAE) analysis of an investment analyst AI's "
        "internal activations captured at each decision step. Explain what "
        "the activation patterns reveal about HOW the AI made its decisions, "
        "not just WHAT it decided.",
        "An AI investment analyst just completed a portfolio allocation analysis. "
        "Here is its final recommendation:\n\n"
        f"{agent_output[:2000]}\n\n"
        "Here is the SAE feature analysis of its internal decision-making at "
        "each step (steps are labeled as TICKER_ToolName, e.g. NVDA_MarketData "
        "means the model was analyzing NVDA market data):\n\n"
        f"{feature_summary}\n\n"
        "IMPORTANT CONTEXT: The SAE was originally trained on tool-selection "
        "decisions (choosing between tools like code_search, web_browser, etc). "
        "The feature LABELS (e.g. 'Validation and API Documentation Check') "
        "come from that domain and do not directly describe investment concepts. "
        "However, the activation PATTERNS (which features fire, their relative "
        "strengths, and how they change across steps) still reflect real "
        "differences in the model's internal processing. Focus your analysis "
        "on activation patterns and relative differences, not on interpreting "
        "the labels literally.\n\n"
        "Please explain:\n"
        "1. What patterns in the agent's internal representations drove its decisions?\n"
        "2. How did the active features change across stocks and tool types?\n"
        "3. What does this reveal about the agent's reasoning process?",
    )

    # Build structured per-step feature evidence for grounded layman prompt
    feature_evidence_lines = []
    for step_info in analysis_results["steps"]:
        step = step_info["step"]
        sae = step_info.get("sae_features")
        if not sae:
            continue
        top3 = sae["top_features"][:3]
        labeled = [f for f in top3 if f.get("label")]
        if not labeled:
            continue
        parts = ", ".join(f'"{f["label"]}" (strength {f["activation"]:.1f})' for f in labeled)
        # Make step labels readable (NVDA_MarketData -> "analyzing NVDA market data")
        step_readable = step.replace("_", " ").lower()
        if "final" in step_readable:
            step_readable = "making the final allocation decision"
        else:
            step_readable = f"analyzing {step_readable}"
        feature_evidence_lines.append(
            f"- While {step_readable}: strongest brain signals were {parts}"
        )
    feature_evidence = "\n".join(feature_evidence_lines) or (
        "- No labeled features were available."
    )

    layman_prompt = _build_chatml(
        "You explain AI decisions in plain, everyday language. No jargon, "
        "no technical terms. Write as if explaining to someone with no "
        "technical background.",
        "An AI investment analyst was asked to allocate $1,000,000 across "
        "NVDA (NVIDIA), TSLA (Tesla), and AAPL (Apple) stocks. "
        "Here is what it decided:\n\n"
        f"{agent_output[:1500]}\n\n"
        "We looked inside the AI's brain at each step of its reasoning. "
        "Here is exactly what we found:\n\n"
        f"{feature_evidence}\n\n"
        "Note: the signal names come from a different AI task (tool selection) "
        "so they sound technical, but the PATTERNS still reflect real "
        "differences in how the AI processed each stock.\n\n"
        "Using ONLY the brain signals listed above, explain in 4-6 plain "
        "sentences what was going through the agent's mind when making "
        "this investment decision. Reference which signals were strongest "
        "and how they differed between stocks. Start with 'The agent decided'.",
    )

    print("  Loading vLLM for explanation generation...")
    llm = LLM(
        model=_MODEL_NAME,
        tensor_parallel_size=tp_size,
        max_model_len=8192,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        dtype="bfloat16",
    )

    technical_params = SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=1024, repetition_penalty=1.1
    )
    layman_params = SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=512, repetition_penalty=1.1
    )

    outputs = llm.generate(
        [technical_prompt, layman_prompt],
        [technical_params, layman_params],
    )
    technical_explanation = outputs[0].outputs[0].text
    layman_explanation = outputs[1].outputs[0].text

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return technical_explanation, layman_explanation


# ---------------------------------------------------------------------------
# Section F: Summary printing
# ---------------------------------------------------------------------------


def render_markdown(text: str, width: int = 100) -> str:
    """Render markdown text for terminal display with proper table formatting.

    Parses markdown tables and renders them as ASCII box-drawing tables.
    Wraps regular text paragraphs to the specified width.
    """
    lines = text.split("\n")
    output = []
    table_buffer = []
    in_table = False

    def flush_table():
        """Render accumulated table lines as a formatted ASCII table."""
        if not table_buffer:
            return

        # Parse table rows
        rows = []
        for line in table_buffer:
            # Strip leading/trailing pipes and split
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append(cells)

        if len(rows) < 2:
            # Not a valid table, just output as-is
            output.extend(table_buffer)
            return

        # Skip separator row (contains only dashes)
        header = rows[0]
        data_rows = [r for r in rows[1:] if not all(set(c.strip()) <= {"-", ":"} for c in r)]

        # Calculate column widths (max of header and all data)
        n_cols = len(header)
        col_widths = [len(h) for h in header]
        for row in data_rows:
            for i, cell in enumerate(row[:n_cols]):
                col_widths[i] = max(col_widths[i], len(cell))

        # Cap column widths for readability
        max_col_width = 50
        col_widths = [min(w, max_col_width) for w in col_widths]

        # Build table
        def make_row(cells, widths):
            parts = []
            for cell, w in zip(cells, widths):
                # Truncate if needed
                if len(cell) > w:
                    cell = cell[: w - 3] + "..."
                parts.append(cell.ljust(w))
            return "| " + " | ".join(parts) + " |"

        def make_separator(widths):
            return "+-" + "-+-".join("-" * w for w in widths) + "-+"

        output.append(make_separator(col_widths))
        output.append(make_row(header, col_widths))
        output.append(make_separator(col_widths))
        for row in data_rows:
            # Pad row if needed
            row = row + [""] * (n_cols - len(row))
            output.append(make_row(row, col_widths))
        output.append(make_separator(col_widths))

    for line in lines:
        # Detect table lines (start with |)
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            if not in_table:
                in_table = True
                table_buffer = []
            table_buffer.append(line)
        else:
            if in_table:
                flush_table()
                table_buffer = []
                in_table = False

            # Handle headers
            if stripped.startswith("**") and stripped.endswith("**"):
                output.append("")
                output.append(stripped)
                output.append("")
            # Handle bullet points
            elif stripped.startswith("* ") or stripped.startswith("- "):
                # Wrap long bullet points
                wrapped = textwrap.fill(stripped, width=width, subsequent_indent="  ")
                output.append(wrapped)
            else:
                # Regular paragraph - wrap
                if stripped:
                    wrapped = textwrap.fill(stripped, width=width)
                    output.append(wrapped)
                else:
                    output.append("")

    # Flush any remaining table
    if in_table:
        flush_table()

    return "\n".join(output)


def print_analysis_summary(analysis: dict):
    """Print a human-readable summary of the analysis results."""
    print(f"\n{'=' * 60}")
    print("  ACTIVATION ANALYSIS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Decision steps captured: {len(analysis['steps'])}")
    print(f"  SAE available: {analysis['sae_available']}")
    print(f"  Feature labels available: {analysis['features_available']}")

    for step_info in analysis["steps"]:
        print(f"\n  --- {step_info['step']} ---")

        for layer, stats in step_info.get("raw_stats", {}).items():
            if "residual_20" in layer:
                print(f"    L2 norm: {stats['l2_norm']:.2f}, std: {stats['std']:.4f}")

        sae = step_info.get("sae_features")
        if sae:
            print(
                f"    Active features: {sae['num_active']}/{sae['total_features']} "
                f"({100 - sae['sparsity_pct']:.1f}%)"
            )
            for feat in sae["top_features"][:3]:
                label = feat.get("label", f"Feature #{feat['index']}")
                print(f"      {label}: {feat['activation']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Investment Decision Demo with SAE Activation Analysis"
    )
    parser.add_argument(
        "--sae-checkpoint",
        type=str,
        default=None,
        help="Path to trained SAE checkpoint (default: auto-detect in output/)",
    )
    parser.add_argument(
        "--feature-descriptions",
        type=str,
        default=None,
        help="Path to feature_descriptions.json (default: auto-detect in output/)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[8, 12, 16, 20, 24],
        help="Transformer layers to hook for activation extraction",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per LLM call",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (number of GPUs)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.45,
        help="vLLM GPU memory utilization fraction",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Investment Decision Demo")
    print("  Nemotron3 (vLLM) + SAE Activation Analysis")
    print("=" * 60)

    # Phase 1: Run scripted multi-step analysis with vLLM
    print("\n[Phase 1] Running investment analysis...")
    engine = VLLMEngine(
        tp_size=args.tp_size,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    final_recommendation, per_stock = run_investment_analysis(engine)
    prompt_log = engine.prompt_log

    print(f"\n{'=' * 60}")
    print(f"  Analysis complete. {len(prompt_log)} decision points recorded.")
    print(f"{'=' * 60}")
    print(f"\n  FINAL RECOMMENDATION:\n{final_recommendation}")

    # Release vLLM
    engine.cleanup()

    # Phase 2: Extract activations via HF transformers
    print("\n[Phase 2] Extracting activations via HF transformers...")
    activation_log = extract_activations_for_prompts(prompt_log, args.layers)

    # Phase 3: Analyze activations through SAE
    print("\n[Phase 3] Analyzing activations through SAE...")
    analysis = analyze_activations(
        activation_log=activation_log,
        sae_checkpoint=args.sae_checkpoint,
        feature_descriptions_path=args.feature_descriptions,
    )
    print_analysis_summary(analysis)

    if analysis["features_available"]:
        print(f"\n{'=' * 60}")
        print("  NOTE ON FEATURE LABELS")
        print(f"{'=' * 60}")
        print(
            "  The SAE was trained on tool-selection decisions (e.g. 'use the\n"
            "  code_search tool' vs 'use the web_browser tool'). Feature labels\n"
            "  like 'Validation and API Documentation Check' reflect THAT\n"
            "  training domain, not investment analysis.\n"
            "\n"
            "  When the same SAE is applied to investment prompts, it decomposes\n"
            "  activations into the only vocabulary it knows. The feature\n"
            "  ACTIVATIONS (which features fire, how strongly, and how they\n"
            "  differ across steps) are still meaningful -- the LABELS are just\n"
            "  borrowed from a different domain.\n"
            "\n"
            "  To get investment-specific labels, you would train an SAE on\n"
            "  investment-related contrastive pairs."
        )

    # Phase 4: Generate explanations
    print("\n[Phase 4] Generating decision explanations via Nemotron3...")
    technical_explanation, layman_explanation = generate_decision_explanations(
        analysis_results=analysis,
        agent_output=final_recommendation,
        tp_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    print(f"\n{'=' * 60}")
    print("  TECHNICAL EXPLANATION")
    print(f"{'=' * 60}")
    print(render_markdown(technical_explanation))

    print(f"\n{'=' * 60}")
    print("  PLAIN LANGUAGE SUMMARY")
    print(f"{'=' * 60}")
    print(render_markdown(layman_explanation))

    # Phase 5: Save results
    output_path = Path("demo/output")
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "analysis_results.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    with open(output_path / "agent_output.txt", "w") as f:
        f.write(final_recommendation)
    with open(output_path / "per_stock_analyses.json", "w") as f:
        json.dump(per_stock, f, indent=2)
    with open(output_path / "technical_explanation.txt", "w") as f:
        f.write(technical_explanation)
    with open(output_path / "layman_explanation.txt", "w") as f:
        f.write(layman_explanation)

    print(f"\n  Results saved to {output_path}/")
    print("\n  Done.")


if __name__ == "__main__":
    main()
