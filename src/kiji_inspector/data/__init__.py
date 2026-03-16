from kiji_inspector.data.contrastive_dataset import ContrastiveDataset, ContrastivePair
from kiji_inspector.data.scenario import (
    ScenarioConfig,
    default_scenario,
    load_scenarios,
    load_scenarios_meta,
    save_scenarios_meta,
)


def __getattr__(name: str):
    if name == "ContrastivePairGenerator":
        from kiji_inspector.data.generator import ContrastivePairGenerator

        return ContrastivePairGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ContrastiveDataset",
    "ContrastivePair",
    "ContrastivePairGenerator",
    "ScenarioConfig",
    "default_scenario",
    "load_scenarios",
    "load_scenarios_meta",
    "save_scenarios_meta",
]
