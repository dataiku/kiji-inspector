"""
Investment Decision Demo with Streamlit UI.

A web-based interface showing:
  - Live agent state and progress
  - Step-by-step analysis with activation insights
  - Final explanations in formatted cards

This demo runs the analysis in a subprocess to avoid conflicts between
Streamlit's execution model and vLLM's multiprocessing.

Usage:
    uv run streamlit run demo/investment_demo_streamlit.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Investment Analysis Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

TICKERS = ["NVDA", "TSLA", "AAPL"]
TOOLS = ["MarketData", "Financials", "Risk", "News"]
OUTPUT_DIR = Path("demo/output")


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------


def init_session_state():
    """Initialize session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.running = False
        st.session_state.completed = False
        st.session_state.process = None
        st.session_state.error = None
        # Try to load previous results
        load_results_from_disk()


def load_results_from_disk():
    """Load previously saved results if they exist."""
    st.session_state.final_recommendation = ""
    st.session_state.per_stock_analyses = {}
    st.session_state.analysis_results = None
    st.session_state.technical_explanation = ""
    st.session_state.layman_explanation = ""

    if (OUTPUT_DIR / "agent_output.txt").exists():
        st.session_state.final_recommendation = (OUTPUT_DIR / "agent_output.txt").read_text()
        st.session_state.completed = True

    if (OUTPUT_DIR / "per_stock_analyses.json").exists():
        with open(OUTPUT_DIR / "per_stock_analyses.json") as f:
            st.session_state.per_stock_analyses = json.load(f)

    if (OUTPUT_DIR / "analysis_results.json").exists():
        with open(OUTPUT_DIR / "analysis_results.json") as f:
            st.session_state.analysis_results = json.load(f)

    if (OUTPUT_DIR / "technical_explanation.txt").exists():
        st.session_state.technical_explanation = (
            OUTPUT_DIR / "technical_explanation.txt"
        ).read_text()

    if (OUTPUT_DIR / "layman_explanation.txt").exists():
        st.session_state.layman_explanation = (OUTPUT_DIR / "layman_explanation.txt").read_text()


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------


def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.title("Configuration")

    tp_size = st.sidebar.number_input(
        "Tensor Parallel Size",
        min_value=1,
        max_value=8,
        value=4,
        help="Number of GPUs for tensor parallelism",
    )

    gpu_mem = st.sidebar.slider(
        "GPU Memory Utilization",
        min_value=0.1,
        max_value=0.95,
        value=0.45,
        step=0.05,
        help="Fraction of GPU memory to use",
    )

    sae_checkpoint = st.sidebar.text_input(
        "SAE Checkpoint Path",
        value="output/sae_checkpoints/sae_final.pt",
        help="Path to trained SAE model",
    )

    feature_desc = st.sidebar.text_input(
        "Feature Descriptions Path",
        value="output/activations/feature_descriptions.json",
        help="Path to feature descriptions JSON",
    )

    st.sidebar.divider()

    # Status display
    st.sidebar.subheader("Status")
    if st.session_state.running:
        st.sidebar.warning("Analysis running in background...")
    elif st.session_state.completed:
        st.sidebar.success("Analysis Complete")
    elif st.session_state.error:
        st.sidebar.error("Analysis Failed")
    else:
        st.sidebar.info("Ready to start")

    return tp_size, gpu_mem, sae_checkpoint, feature_desc


def render_steps_table():
    """Render the analysis steps table from results."""
    analysis = st.session_state.analysis_results
    if not analysis or "steps" not in analysis:
        st.info("No analysis steps available yet. Run the analysis first.")
        return

    rows = []
    for step_info in analysis["steps"]:
        step_name = step_info.get("step", "unknown")

        # Get raw stats for the primary layer
        raw_stats = step_info.get("raw_stats", {})
        l2_norm = "-"
        for layer_name, stats in raw_stats.items():
            if "residual_20" in layer_name:
                l2_norm = f"{stats.get('l2_norm', 0):.2f}"
                break

        # Get SAE features
        sae_feats = step_info.get("sae_features")
        if sae_feats and sae_feats.get("top_features"):
            top = sae_feats["top_features"][0]
            top_feature = top.get("label", f"Feature #{top['index']}")
            top_activation = f"{top['activation']:.2f}"
            num_active = sae_feats.get("num_active", 0)
        else:
            top_feature = "-"
            top_activation = "-"
            num_active = 0

        rows.append(
            {
                "Step": step_name,
                "L2 Norm": l2_norm,
                "Active Features": num_active,
                "Top Feature": top_feature,
                "Activation": top_activation,
            }
        )

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Step": st.column_config.TextColumn("Step", width="medium"),
            "L2 Norm": st.column_config.TextColumn("L2 Norm", width="small"),
            "Active Features": st.column_config.NumberColumn("Active", width="small"),
            "Top Feature": st.column_config.TextColumn("Top Feature", width="large"),
            "Activation": st.column_config.TextColumn("Activation", width="small"),
        },
    )


def render_per_stock_analyses():
    """Render individual analyses in expandable sections."""
    analyses = st.session_state.per_stock_analyses
    if not analyses:
        return

    st.subheader("Individual Stock Analyses")

    for ticker in TICKERS:
        if ticker in analyses:
            with st.expander(f"{ticker} Analysis", expanded=False):
                st.markdown(analyses[ticker])


def render_final_results():
    """Render the final recommendation and explanations."""
    if not st.session_state.final_recommendation:
        return

    st.subheader("Portfolio Recommendation")
    st.markdown(st.session_state.final_recommendation)

    if st.session_state.technical_explanation or st.session_state.layman_explanation:
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Technical Explanation")
            if st.session_state.technical_explanation:
                st.markdown(st.session_state.technical_explanation)
            else:
                st.info("Not available")

        with col2:
            st.subheader("Plain Language Summary")
            if st.session_state.layman_explanation:
                st.markdown(st.session_state.layman_explanation)
            else:
                st.info("Not available")


def render_top_features_chart():
    """Render a bar chart of top features from the final decision."""
    analysis = st.session_state.analysis_results
    if not analysis or "steps" not in analysis:
        return

    # Get features from final allocation step
    final_step = None
    for step in analysis["steps"]:
        if "final" in step.get("step", "").lower():
            final_step = step
            break

    if not final_step:
        final_step = analysis["steps"][-1] if analysis["steps"] else None

    if not final_step or "sae_features" not in final_step:
        return

    sae_feats = final_step["sae_features"]
    if not sae_feats.get("top_features"):
        return

    st.subheader("Top SAE Features (Final Decision)")

    # Prepare data for chart
    features = sae_feats["top_features"][:10]
    chart_data = pd.DataFrame(
        {
            "Feature": [f.get("label", f"#{f['index']}") for f in features],
            "Activation": [f["activation"] for f in features],
        }
    )

    st.bar_chart(chart_data.set_index("Feature"))


# ---------------------------------------------------------------------------
# Analysis Runner
# ---------------------------------------------------------------------------


def run_analysis_subprocess(tp_size: int, gpu_mem: float, sae_checkpoint: str, feature_desc: str):
    """Run the analysis in a subprocess using the original demo script."""
    cmd = [
        sys.executable,
        "demo/investment_demo.py",
        "--tp-size",
        str(tp_size),
        "--gpu-memory-utilization",
        str(gpu_mem),
    ]

    if sae_checkpoint:
        cmd.extend(["--sae-checkpoint", sae_checkpoint])
    if feature_desc:
        cmd.extend(["--feature-descriptions", feature_desc])

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------


def main():
    init_session_state()

    # Header
    st.title("Investment Analysis Demo")
    st.markdown("*Analyzing NVDA, TSLA, AAPL with SAE Explainability*")

    # Sidebar
    tp_size, gpu_mem, sae_checkpoint, feature_desc = render_sidebar()

    st.divider()

    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        start_disabled = st.session_state.running
        if st.button("Run Analysis", type="primary", disabled=start_disabled):
            st.session_state.running = True
            st.session_state.completed = False
            st.session_state.error = None
            st.rerun()

    with col2:
        if st.button("Reload Results"):
            load_results_from_disk()
            st.rerun()

    # Handle running state
    if st.session_state.running:
        st.divider()
        st.subheader("Running Analysis...")

        with st.spinner("Analysis in progress. This may take several minutes..."):
            log_container = st.empty()
            progress_text = st.empty()

            # Run the subprocess
            proc = run_analysis_subprocess(tp_size, gpu_mem, sae_checkpoint, feature_desc)

            logs = []
            phase = "Initializing"

            try:
                for line in proc.stdout:
                    logs.append(line.rstrip())

                    # Parse phase from output
                    if "[Phase 1]" in line:
                        phase = "Phase 1: vLLM Generation"
                    elif "[Phase 2]" in line:
                        phase = "Phase 2: Activation Extraction"
                    elif "[Phase 3]" in line:
                        phase = "Phase 3: SAE Analysis"
                    elif "[Phase 4]" in line:
                        phase = "Phase 4: Generating Explanations"
                    elif "Done." in line:
                        phase = "Complete"

                    progress_text.text(f"Current phase: {phase}")

                    # Show last 20 lines of log
                    log_container.code("\n".join(logs[-20:]), language="text")

                proc.wait()

                if proc.returncode == 0:
                    st.session_state.completed = True
                    st.session_state.running = False
                    load_results_from_disk()
                    st.success("Analysis complete!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.session_state.error = f"Process exited with code {proc.returncode}"
                    st.session_state.running = False
                    st.error(st.session_state.error)

            except Exception as e:
                st.session_state.error = str(e)
                st.session_state.running = False
                st.error(f"Error: {e}")
                proc.kill()

    # Display results
    st.divider()

    # Analysis steps table
    st.subheader("Analysis Steps")
    render_steps_table()

    # Top features chart
    render_top_features_chart()

    # Individual analyses
    render_per_stock_analyses()

    st.divider()

    # Final results
    render_final_results()

    # Footer
    st.divider()
    st.caption("Results are saved to `demo/output/` and persist between sessions.")


if __name__ == "__main__":
    main()
