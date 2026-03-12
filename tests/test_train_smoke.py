import importlib
import subprocess
import sys


def test_train_namespace_imports():
    training = importlib.import_module("kiji_inspector.training")
    analysis = importlib.import_module("kiji_inspector.analysis")
    extraction = importlib.import_module("kiji_inspector.extraction")

    assert hasattr(training, "train_sae")
    assert hasattr(analysis, "identify_contrastive_features")
    assert hasattr(extraction, "create_extractor")


def test_cli_module_help_smoke():
    for module_name in ("kiji_inspector.pipeline", "kiji_inspector.generate_pairs"):
        result = subprocess.run(
            [sys.executable, "-m", module_name, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()
