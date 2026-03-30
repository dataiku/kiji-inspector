"""
Scenario configuration for multi-domain SAE training.

A scenario defines a domain (tool selection, investment, manufacturing, etc.)
with its own tools, system prompt, and contrast types. The pipeline can
train on multiple scenarios to produce a generic SAE.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScenarioConfig:
    """Configuration for a single domain scenario."""

    name: str
    system_prompt: str
    tools: list[dict[str, str]]  # [{"name": ..., "description": ...}]
    contrast_types: dict[str, str]  # {type_name: explanation}

    @classmethod
    def from_json(cls, path: str | Path) -> ScenarioConfig:
        """Load and validate a scenario config file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scenario config not found: {path}")

        with open(path) as f:
            data = json.load(f)

        # Validate required fields
        for key in ("name", "system_prompt", "tools", "contrast_types"):
            if key not in data:
                raise ValueError(f"Scenario config {path} missing required field: {key}")

        # Validate tools structure
        for i, tool in enumerate(data["tools"]):
            if "name" not in tool or "description" not in tool:
                raise ValueError(f"Tool {i} in {path} must have 'name' and 'description' fields")

        if not data["contrast_types"]:
            raise ValueError(f"Scenario config {path} must have at least one contrast type")

        return cls(
            name=data["name"],
            system_prompt=data["system_prompt"],
            tools=data["tools"],
            contrast_types=data["contrast_types"],
        )

    def to_dict(self) -> dict:
        """Serialize for scenarios_meta.json or subprocess transfer."""
        return {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "contrast_types": self.contrast_types,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ScenarioConfig:
        """Reconstruct from serialized dict."""
        return cls(
            name=d["name"],
            system_prompt=d["system_prompt"],
            tools=d["tools"],
            contrast_types=d["contrast_types"],
        )


def load_scenarios(paths: list[str | Path]) -> list[ScenarioConfig]:
    """Load multiple scenario configs, validate no name collisions."""
    scenarios = [ScenarioConfig.from_json(p) for p in paths]
    names = [s.name for s in scenarios]
    dupes = [n for n in names if names.count(n) > 1]
    if dupes:
        raise ValueError(f"Duplicate scenario names: {set(dupes)}")
    return scenarios


def default_scenario() -> ScenarioConfig:
    """Return the built-in tool_selection scenario (backward compat).

    This matches the hardcoded TOOLS, SYSTEM_PROMPT, and
    CONTRAST_EXPLANATIONS that were previously in generator.py and
    generate_training_set.py.
    """
    return ScenarioConfig(
        name="tool_selection",
        system_prompt="You are a helpful assistant. Choose the best tool for each request.",
        tools=[
            {"name": "internal_search", "description": "Search internal company documentation"},
            {"name": "web_search", "description": "Search the public web"},
            {"name": "file_read", "description": "Read a local file"},
            {"name": "file_write", "description": "Write or update a local file"},
            {"name": "database_query", "description": "Query a SQL database"},
            {"name": "api_call", "description": "Call an external REST API"},
            {"name": "code_execute", "description": "Execute code in a sandbox"},
            {"name": "delegate_agent", "description": "Delegate to a sub-agent for complex tasks"},
        ],
        contrast_types={
            "internal_vs_external": (
                "One request needs information from internal/company sources, "
                "the other needs external/public web information. "
                'Example difference: "our docs" vs "online resources"'
            ),
            "local_vs_remote": (
                "One request targets local files/resources, "
                "the other targets remote/cloud resources. "
                'Example difference: "local config" vs "remote server config"'
            ),
            "cached_vs_live": (
                "One request can be served from cached/stored data, "
                "the other requires fresh/live data. "
                'Example difference: "last known" vs "current"'
            ),
            "read_vs_write": (
                "One request is read-only (fetching/viewing), "
                "the other requires writing/modification. "
                'Example difference: "show me" vs "update the"'
            ),
            "create_vs_update": (
                "One request creates a new resource, "
                "the other modifies an existing one. "
                'Example difference: "create a new" vs "modify the existing"'
            ),
            "query_vs_mutate": (
                "One request queries/reads state, "
                "the other mutates/changes state. "
                'Example difference: "what is" vs "set the"'
            ),
            "single_vs_batch": (
                "One request operates on a single item, "
                "the other operates on multiple items. "
                'Example difference: "this file" vs "all files matching"'
            ),
            "shallow_vs_deep": (
                "One request needs a surface-level answer, "
                "the other needs deep/recursive analysis. "
                'Example difference: "quick summary" vs "detailed breakdown"'
            ),
            "specific_vs_broad": (
                "One request targets a specific item, "
                "the other spans a broad scope. "
                'Example difference: "this function" vs "the entire module"'
            ),
            "authoritative_vs_general": (
                "One request needs an authoritative/official source, "
                "the other accepts general information. "
                'Example difference: "official documentation" vs "any tutorial"'
            ),
            "verified_vs_unverified": (
                "One request needs verified/trusted data, "
                "the other accepts unverified sources. "
                'Example difference: "peer-reviewed" vs "blog post"'
            ),
            "direct_vs_delegated": (
                "One request can be handled directly, "
                "the other is complex enough to require delegation/decomposition. "
                "Example difference: simple query vs multi-step analysis"
            ),
            "single_vs_multi_tool": (
                "One request can be solved with a single tool call, "
                "the other requires orchestrating multiple tools. "
                'Example difference: "look up X" vs "look up X then compare with Y"'
            ),
        },
    )


def discover_scenarios(scenario_paths: list[str | Path] | None = None) -> list[ScenarioConfig]:
    """Load scenario configs from explicit paths, or discover all *.json in scenarios/.

    When no paths are provided, all *.json files in the project's
    scenarios/ directory are loaded automatically.

    Args:
        scenario_paths: Explicit paths to scenario JSON files.
            If None, discovers all *.json in the project's scenarios/ directory.

    Returns:
        List of validated ScenarioConfig objects.
    """
    if scenario_paths:
        return load_scenarios(scenario_paths)

    scenarios_dir = Path(__file__).resolve().parent.parent.parent.parent / "scenarios"
    paths = sorted(scenarios_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No scenario files found in {scenarios_dir}. "
            "Create at least one .json file or use --scenario."
        )
    return load_scenarios(list(paths))


def save_scenarios_meta(scenarios: list[ScenarioConfig], output_dir: Path) -> Path:
    """Write scenarios_meta.json to the pairs output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {s.name: s.to_dict() for s in scenarios}
    path = output_dir / "scenarios_meta.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path


def load_scenarios_meta(pairs_dir: Path) -> dict[str, ScenarioConfig]:
    """Read scenarios_meta.json, return {name: ScenarioConfig}.

    Falls back to default_scenario() if file doesn't exist (backward compat).
    """
    pairs_dir = Path(pairs_dir)
    meta_path = pairs_dir / "scenarios_meta.json"
    if not meta_path.exists():
        ds = default_scenario()
        return {ds.name: ds}

    with open(meta_path) as f:
        raw = json.load(f)

    return {name: ScenarioConfig.from_dict(d) for name, d in raw.items()}
