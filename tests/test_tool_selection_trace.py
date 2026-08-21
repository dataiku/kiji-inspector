from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def trace():
    directory = Path(__file__).parents[1] / "demo" / "tool_selection"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("trace_pairs", directory / "trace_pairs.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_request_span_and_condition_positions(trace):
    prompt = "system: be good\nuser: read the local file\nassistant: I'll use the"
    request = "read the local file"
    # fake offsets: one token per word-ish chunk
    words, offsets, pos = prompt.split(" "), [], 0
    for w in words:
        offsets.append((pos, pos + len(w)))
        pos += len(w) + 1
    span = trace.request_span(offsets, prompt, request)
    assert span and prompt[offsets[span[0]][0] : offsets[span[-1]][1]].strip().startswith("read")
    n = len(words)
    assert trace.condition_positions("decision", n, span) == [n - 1]
    assert trace.condition_positions("request", n, span) == span
    assert trace.condition_positions("allButDecision", n, span) == list(range(n - 1))
    assert trace.condition_positions("all", n, span) == list(range(n))
    with pytest.raises(ValueError):
        trace.condition_positions("nope", n, span)


def test_family_profile_sums_merged_features(trace):
    feats = np.zeros((4, 6), dtype=np.float32)
    feats[:, 1] = [0, 1, 2, 3]
    feats[:, 4] = [1, 1, 1, 1]
    families = [
        {"index": 1, "label": "A", "merged": [4]},
        {"index": 2, "label": "B"},
        {"index": 3, "label": "C"},
    ]
    prof = trace.family_profile(feats, families, top=2)
    assert [f["label"] for f in prof] == ["A", "B"]
    assert prof[0]["perToken"] == [1.0, 2.0, 3.0, 4.0] and prof[0]["familySize"] == 2
    assert prof[1]["perToken"] == [0.0, 0.0, 0.0, 0.0]


def test_dose_summary_curves_and_bands(trace):
    def reading(p, tool="file_read", other="api_call"):
        return {"distribution": {tool: p, other: 1 - p}, "display": tool if p > 0.5 else other}

    baseline = reading(0.1)
    scales = [0.0, 1.0, 2.0]
    all_readings = [reading(0.1), reading(0.4), reading(0.8)]
    best = {"index": 7, "label": "Local file", "familySize": 2}
    best_readings = [reading(0.1), reading(0.3), reading(0.55)]
    controls = [
        [reading(0.1), reading(0.12), reading(0.15)],
        [reading(0.1), reading(0.09), reading(0.2)],
    ]
    d = trace.dose_summary(
        baseline, scales, all_readings, best, best_readings, controls, "file_read"
    )
    assert d["baselineP"] == pytest.approx(0.1) and d["baselineChoice"] == "api_call"
    assert [c["p"] for c in d["allRows"]] == [0.1, 0.4, 0.8]
    assert d["allRows"][2]["choice"] == "file_read"
    assert d["controlBand"] == pytest.approx([0.0, 0.02, 0.1])
    assert d["bestRow"]["label"] == "Local file" and d["bestRow"]["curve"][2]["p"] == 0.55
    assert trace.scaled_targets({"3": 2.0, 5: 1.0}, 1.5) == {3: 3.0, 5: 1.5}


def test_position_hook_edits_only_requested_positions(trace):
    import torch

    class FakeSAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            d, n = 4, 3
            self.W_dec = torch.nn.Parameter(torch.eye(n, d))
            self.rms_scale = 2.0

        def normalize_input(self, x):
            return x

        def encode(self, x):
            return torch.relu(x[:, :3])

    sae = FakeSAE()
    hidden = torch.ones(1, 5, 4)
    hook = trace.make_position_hook(sae, {0: None, 1: 3.0}, [1, 4])
    (out,), _ = hook(None, (hidden.clone(),), {})
    # feature 0 ablated (1 -> 0): x0 -= 2*1; feature 1 clamped to 3: x1 += 2*2
    assert torch.allclose(out[0, 1], torch.tensor([-1.0, 5.0, 1.0, 1.0]))
    assert torch.allclose(out[0, 4], torch.tensor([-1.0, 5.0, 1.0, 1.0]))
    assert torch.allclose(out[0, 0], hidden[0, 0]) and torch.allclose(out[0, 2], hidden[0, 2])
    # positions=None edits everything; out-of-range positions are skipped.
    (out_all,), _ = trace.make_position_hook(sae, {0: None}, None)(None, (hidden.clone(),), {})
    assert torch.allclose(out_all[0, :, 0], torch.full((5,), -1.0))
    assert trace.make_position_hook(sae, {0: None}, [9])(None, (hidden.clone(),), {}) is None
