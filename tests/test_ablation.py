import pytest
import torch

from kiji_inspector.experiments.ablation import make_ablation_hook


class _FakeSAE:
    """Minimal SAE test double: encode is identity, decode doubles.

    This isolates the hook's *mechanics* (pre-hook vs. post-hook, args vs.
    kwargs, decision-token slicing, feature zeroing) from real SAE math —
    the doubling makes it trivial to tell whether a tensor went through the
    encode/decode round trip at all.
    """

    def __init__(self):
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.rms_scale = None

    def parameters(self):
        return iter([self._param])

    def normalize_input(self, x):
        return x

    def denormalize_output(self, x):
        return x

    def encode(self, x):
        return x.clone()

    def decode(self, features):
        return features * 2


class _FakeLayer(torch.nn.Module):
    """Records exactly what it receives, then passes it straight through."""

    def __init__(self):
        super().__init__()
        self.received = None

    def forward(self, hidden_states=None):
        self.received = hidden_states
        return hidden_states


def test_ablation_hook_modifies_layer_input_not_output():
    """Regression test: the ablation hook must be a pre-hook.

    The SAE is trained on activations captured entering a layer (see
    activation_extractor.py's pre-hook fix), so the ablation intervention
    must patch the same tensor — the layer's *input* — not its output.
    A forward hook on the layer's output would intervene one layer
    downstream of where the SAE's features actually live.
    """
    sae = _FakeSAE()
    layer = _FakeLayer()
    handle = layer.register_forward_pre_hook(
        make_ablation_hook(sae, feature_indices=None, decision_token_only=False),
        with_kwargs=True,
    )

    x = torch.tensor([[[1.0, 2.0]]])  # (batch=1, seq=1, d_model=2)
    output = layer(x)
    handle.remove()

    # The layer must receive the SAE round-tripped tensor (encode/decode -> *2),
    # not the raw input — proving the hook patched the layer's *input*.
    assert torch.equal(layer.received, x * 2)
    assert torch.equal(output, x * 2)
    assert not torch.equal(layer.received, x)


def test_ablation_hook_handles_hidden_states_kwarg():
    """Some HF decoder layers are invoked with hidden_states as a kwarg."""
    sae = _FakeSAE()
    layer = _FakeLayer()
    handle = layer.register_forward_pre_hook(
        make_ablation_hook(sae, feature_indices=None, decision_token_only=False),
        with_kwargs=True,
    )

    x = torch.tensor([[[1.0, 2.0]]])
    layer(hidden_states=x)
    handle.remove()

    assert torch.equal(layer.received, x * 2)


def test_ablation_hook_decision_token_only_leaves_earlier_tokens_untouched():
    sae = _FakeSAE()
    layer = _FakeLayer()
    handle = layer.register_forward_pre_hook(
        make_ablation_hook(sae, feature_indices=None, decision_token_only=True),
        with_kwargs=True,
    )

    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (batch=1, seq=2, d_model=2)
    layer(x)
    handle.remove()

    received = layer.received
    assert torch.equal(received[:, 0, :], x[:, 0, :])
    assert torch.equal(received[:, 1, :], x[:, 1, :] * 2)


def test_ablation_hook_zeros_specified_features():
    class _IdentitySAE(_FakeSAE):
        def decode(self, features):
            return features  # no doubling — isolates feature zeroing

    sae = _IdentitySAE()
    layer = _FakeLayer()
    handle = layer.register_forward_pre_hook(
        make_ablation_hook(sae, feature_indices=[0], decision_token_only=False),
        with_kwargs=True,
    )

    x = torch.tensor([[[1.0, 2.0]]])
    layer(x)
    handle.remove()

    received = layer.received
    assert received[0, 0, 0].item() == 0.0
    assert received[0, 0, 1].item() == 2.0


def test_baseline_pass_rate_guard_arithmetic():
    """The guard must reject the shape of run that produced an empty report.

    Layer 12's first ablation run kept 1 of 100 prompts per contrast type
    because the prompt's decision token was a newline rather than a tool name.
    Every downstream statistic degenerated to 0.0/None while the report still
    looked complete, so the pass rate has to be checked before publishing.
    """
    degenerate = {
        f"contrast_{i}": {"n_tested": 1, "n_baseline_mismatches": 0, "n_unknown_baseline": 99}
        for i in range(34)
    }
    healthy = {
        f"contrast_{i}": {"n_tested": 88, "n_baseline_mismatches": 7, "n_unknown_baseline": 5}
        for i in range(34)
    }

    def pass_rate(per_contrast):
        kept = sum(v["n_tested"] for v in per_contrast.values())
        tried = sum(
            v["n_tested"] + v["n_baseline_mismatches"] + v["n_unknown_baseline"]
            for v in per_contrast.values()
        )
        return kept / tried

    assert pass_rate(degenerate) == pytest.approx(0.01)
    assert pass_rate(degenerate) < 0.2, "guard would not have caught the real failure"
    assert pass_rate(healthy) == pytest.approx(0.88)
    assert pass_rate(healthy) >= 0.2, "guard would reject a healthy run"


# ---------------------------------------------------------------------------
# Frequency-matched random control
# ---------------------------------------------------------------------------


def _rates_with_bins():
    """Synthetic firing-rate array with known bin populations.

    Bins (FIRING_RATE_BIN_EDGES): [0,1e-5) [1e-5,1e-4) [1e-4,1e-3) [1e-3,1e-2)
    [1e-2,0.1) [0.1,1.01). Indices 0-19 silent, 20-39 ultra-rare, 40-59 rare,
    60-79 alive-low, 80-99 alive-mid, 100-119 alive-high.
    """
    import numpy as np

    rates = np.zeros(120)
    rates[20:40] = 5e-5
    rates[40:60] = 5e-4
    rates[60:80] = 5e-3
    rates[80:100] = 5e-2
    rates[100:120] = 0.5
    return rates


def test_frequency_matched_sampler_matches_bins():
    import random as _random

    import numpy as np

    from kiji_inspector.experiments.ablation import (
        FIRING_RATE_BIN_EDGES,
        sample_frequency_matched,
    )

    rates = _rates_with_bins()
    contrastive = [85, 86, 105, 106]  # two alive-mid, two alive-high
    excluded = set(contrastive)
    rng = _random.Random(0)

    matched = sample_frequency_matched(rng, contrastive, rates, excluded)

    assert len(matched) == len(contrastive)
    assert len(set(matched)) == len(matched), "duplicates in matched sample"
    assert not (set(matched) & excluded), "excluded feature sampled"
    bin_of = np.digitize(rates, FIRING_RATE_BIN_EDGES) - 1
    for ci, mi in zip(contrastive, matched, strict=True):
        assert bin_of[mi] == bin_of[ci], f"feature {mi} not in same bin as {ci}"


def test_frequency_matched_sampler_falls_back_to_nearest_bin():
    import random as _random

    import numpy as np

    from kiji_inspector.experiments.ablation import (
        FIRING_RATE_BIN_EDGES,
        sample_frequency_matched,
    )

    rates = _rates_with_bins()
    # Exclude the ENTIRE alive-high bin (100-119) except the contrastive
    # feature itself, so its bin has no eligible pool.
    contrastive = [105]
    excluded = set(range(100, 120))
    rng = _random.Random(0)

    matched = sample_frequency_matched(rng, contrastive, rates, excluded)

    assert len(matched) == 1
    bin_of = np.digitize(rates, FIRING_RATE_BIN_EDGES) - 1
    # Nearest non-empty bin is alive-mid (one below), not a silent bin.
    assert bin_of[matched[0]] == bin_of[105] - 1


def test_frequency_matched_sampler_never_matches_active_to_silent():
    """Regression: the failure this sampler exists to fix.

    The legacy uniform draw paired highly active contrastive features with
    features that never fire, making the control a no-op. With 70% of the
    dictionary silent and all contrastive features alive, no matched feature
    may be near-silent.
    """
    import random as _random

    import numpy as np

    from kiji_inspector.experiments.ablation import sample_frequency_matched

    rng = _random.Random(0)
    n = 1000
    rates = np.zeros(n)
    rates[700:] = 0.05  # 70% silent, 30% alive
    contrastive = list(range(700, 710))
    excluded = set(contrastive)

    matched = sample_frequency_matched(rng, contrastive, rates, excluded)

    assert len(matched) == 10
    assert all(rates[m] >= 1e-3 for m in matched), "active target matched to silent feature"


def test_compute_firing_rates_fake_sae(tmp_path):
    import numpy as np

    from kiji_inspector.experiments.ablation import compute_firing_rates

    class _SparseSAE(_FakeSAE):
        """Feature j fires iff input dim j > 0.5; d_sae == d_model == 4."""

        d_sae = 4

        def encode(self, x):
            return (x > 0.5).to(x.dtype)

    # 8 rows: dim 0 always high (rate 1.0), dim 1 high in half (0.5),
    # dim 2 never (0.0), dim 3 in one row (0.125).
    data = np.zeros((8, 4), dtype=np.float32)
    data[:, 0] = 1.0
    data[:4, 1] = 1.0
    data[0, 3] = 1.0
    np.save(tmp_path / "shard_000000.npy", data)

    rates = compute_firing_rates(_SparseSAE(), tmp_path, chunk_size=8, device="cpu")

    np.testing.assert_allclose(rates, [1.0, 0.5, 0.0, 0.125])


# ---------------------------------------------------------------------------
# Metrics: BH correction, honest denominators, preserved raw counts
# ---------------------------------------------------------------------------


def _make_type(n_tested, deltas_contrastive, flips=2):
    return {
        "n_tested": n_tested,
        "n_baseline_mismatches": 0,
        "n_unknown_baseline": 0,
        "contrastive_flips": flips,
        "contrastive_directed_flips": 1,
        "random_flips": 1,
        "reconstruction_flips": 0,
        "contrastive_feature_indices": [1, 2, 3],
        "n_random_features": 3,
        "prob_deltas": {
            "contrastive": deltas_contrastive,
            "random": [0.0] * len(deltas_contrastive),
            "reconstruction": [0.0] * len(deltas_contrastive),
        },
    }


def test_metrics_bh_and_denominators():
    from kiji_inspector.experiments.ablation import compute_ablation_metrics

    per_contrast = {
        # 3 testable types (>=10 nonzero deltas), strong positive shifts
        **{
            f"strong_{i}": _make_type(50, [0.01 * (j + 1) for j in range(20)])
            for i in range(3)
        },
        # 2 excluded types: tested but too few nonzero deltas
        **{f"null_{i}": _make_type(25, [0.0] * 24 + [0.001]) for i in range(2)},
    }

    report = compute_ablation_metrics(per_contrast)
    agg = report["aggregate"]

    assert agg["wilcoxon_tested_types"] == 3
    assert sorted(agg["wilcoxon_excluded_types"]) == ["null_0", "null_1"]
    # Strong types are all significant, raw and BH-adjusted alike.
    assert agg["wilcoxon_significant_count"] == 3
    assert agg["wilcoxon_significant_count_bh"] == 3
    assert agg["wilcoxon_significant_rate_tested"] == pytest.approx(1.0)
    # Honest denominator counts the 2 excluded types as not-significant.
    assert agg["wilcoxon_significant_rate_all"] == pytest.approx(3 / 5)
    assert agg["fisher_combined_p_value"] is not None
    assert "mean_wilcoxon_p_value" not in agg

    strong = report["per_contrast_type"]["strong_0"]
    assert "wilcoxon_p_value_bh" in strong["contrastive_ablation"]
    assert strong["contrastive_ablation"]["wilcoxon_p_value_bh"] >= (
        strong["contrastive_ablation"]["wilcoxon_p_value"]
    )
    # Raw counts preserved for post-hoc reanalysis, not popped.
    assert strong["raw_counts"]["contrastive_flips"] == 2
    assert strong["raw_counts"]["contrastive_feature_indices"] == [1, 2, 3]


def test_metrics_random_and_recon_get_wilcoxon():
    from kiji_inspector.experiments.ablation import compute_ablation_metrics

    t = _make_type(50, [0.01] * 20)
    t["prob_deltas"]["random"] = [0.005 * (j + 1) for j in range(20)]
    t["prob_deltas"]["reconstruction"] = [0.0] * 20  # too few nonzero -> None

    report = compute_ablation_metrics({"only": t})
    info = report["per_contrast_type"]["only"]

    assert info["random_ablation"]["wilcoxon_p_value"] is not None
    assert info["reconstruction_baseline"]["wilcoxon_p_value"] is None


def test_compute_type_conditional_rates(tmp_path):
    """Rates must be computed on the type's anchor rows only (row 2*i)."""
    import numpy as np

    from kiji_inspector.analysis.shard_io import open_layer_shards
    from kiji_inspector.experiments.ablation import compute_type_conditional_rates

    class _SparseSAE(_FakeSAE):
        d_sae = 4

        def encode(self, x):
            return (x > 0.5).to(x.dtype)

    # 4 pairs -> 8 rows. Anchor rows (0,2,4,6) fire dim 0; contrast rows
    # (1,3,5,7) fire dim 1. Pair indices {0, 2} -> anchor rows {0, 4}, where
    # dim 2 fires only on row 4.
    data = np.zeros((8, 4), dtype=np.float32)
    data[0::2, 0] = 1.0
    data[1::2, 1] = 1.0
    data[4, 2] = 1.0
    np.save(tmp_path / "shard_000000.npy", data)

    memmaps, offsets = open_layer_shards(tmp_path)
    rates = compute_type_conditional_rates(
        _SparseSAE(), memmaps, offsets, np.array([0, 2]), chunk_size=2
    )

    # dim 0: fires on both anchor rows -> 1.0; dim 1: contrast-only -> 0.0;
    # dim 2: fires on one of the two anchors -> 0.5.
    np.testing.assert_allclose(rates, [1.0, 0.0, 0.5, 0.0])
