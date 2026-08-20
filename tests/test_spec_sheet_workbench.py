from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def wb():
    directory = Path(__file__).parents[1] / "demo" / "spec_sheet"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(
            "feature_workbench", directory / "feature_workbench.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_content_words_drop_stop_and_boilerplate(wb):
    words = wb.content_words("The anchor request requires authoritative internal documentation")
    assert "authoritative" in words and "internal" in words and "documentation" in words
    assert "anchor" not in words and "request" not in words and "the" not in words


def test_signal_sides_split_on_while(wb):
    anchor, contrast = wb.signal_sides(
        "The anchor request needs internal docs, while the contrast accepts public info."
    )
    assert "internal" in anchor and "public" not in anchor
    assert "public" in contrast
    same_a, same_b = wb.signal_sides("no split word here")
    assert same_a == same_b == "no split word here"


def test_prompt_tool_map_drops_conflicts(wb):
    rows = [
        {"anchor_prompt": "p1", "anchor_tool": "a", "contrast_prompt": "p2", "contrast_tool": "b"},
        {"anchor_prompt": "p1", "anchor_tool": "c", "contrast_prompt": "p3", "contrast_tool": "b"},
    ]
    tools, conflicting = wb.prompt_tool_map(rows)
    assert "p1" not in tools and tools["p2"] == "b" and tools["p3"] == "b"
    assert conflicting == 1


def test_top_side_features_signs(wb):
    diff = np.array([0.0, 5.0, -3.0, 1.0, -0.5])
    anchor, contrast = wb.top_side_features(diff, k=2)
    assert anchor == [1, 3]
    assert contrast == [2, 4]
    all_positive = np.array([1.0, 2.0])
    _, empty_contrast = wb.top_side_features(all_positive, k=2)
    assert empty_contrast == []


def test_label_overlap_counts_hits(wb):
    labels = {
        "3": {"label": "Internal documentation lookup", "description": ""},
        "7": {"label": "Weather chat", "description": "casual talk"},
    }
    clause = wb.content_words("needs internal documentation")
    assert wb.label_overlap([3, 7], labels, clause) == 1
    assert wb.label_overlap([7], labels, clause) == 0
    assert wb.label_overlap([99], labels, clause) == 0  # unlabeled feature


def test_bow_vocabulary_and_features(wb):
    texts = ["read the config file now"] * 5 + ["write the config file now"] * 5
    vocab = wb.build_vocabulary(texts, min_count=5)
    assert "config" in vocab and "read" in vocab and "write" in vocab
    x = wb.bow_features(["read the config"], vocab)
    assert x.shape == (1, len(vocab))
    assert x[0, vocab.index("read")] == 1.0
    assert x[0, vocab.index("write")] == 0.0


def test_fit_probe_learns_separable_data(wb):
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4)).astype(np.float32)
    y = np.where(x[:, 0] > 0, "left", "right")
    report = wb.fit_probe(x[:150], y[:150], x[150:], y[150:])
    assert report["accuracy"] > 0.9
    assert report["nClasses"] == 2 and report["nTest"] == 50
