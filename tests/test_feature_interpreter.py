"""Tests for the memmap-backed Step 4a activation loader.

Covers ShardedRowView indexing semantics and
load_activations_from_shards dedup/alignment behavior.
"""

import json

import numpy as np
import pytest
import torch

from kiji_inspector.analysis.feature_interpreter import (
    collect_max_activating_examples,
    load_activations_from_shards,
)
from kiji_inspector.analysis.shard_io import ShardedRowView, open_layer_shards

D_MODEL = 4


def _write_shards(layer_dir, row_counts):
    """Write consecutive shard_*.npy files; row i (globally) is filled with i."""
    layer_dir.mkdir(parents=True, exist_ok=True)
    start = 0
    for shard_idx, n in enumerate(row_counts):
        rows = np.arange(start, start + n, dtype=np.float32)
        arr = np.repeat(rows[:, None], D_MODEL, axis=1)
        np.save(layer_dir / f"shard_{shard_idx:06d}.npy", arr)
        start += n
    return start


def _write_activations_dir(tmp_path, prompts, row_counts):
    """Create a Step-1-style activations dir: shards + metadata + prompts."""
    act_dir = tmp_path / "activations"
    _write_shards(act_dir, row_counts)
    with open(act_dir / "metadata.json", "w") as f:
        json.dump({"model": "test-model", "layer": "residual_0", "d_model": D_MODEL}, f)
    with open(act_dir / "prompts.json", "w") as f:
        json.dump(prompts, f)
    return act_dir


def _eager_reference(row_counts):
    """Dense equivalent of the shards for comparison."""
    total = sum(row_counts)
    rows = np.arange(total, dtype=np.float32)
    return np.repeat(rows[:, None], D_MODEL, axis=1)


class TestShardedRowView:
    @pytest.fixture
    def view(self, tmp_path):
        # 3 shards, final one partial (smaller), 10 rows total: 4 + 4 + 2
        layer_dir = tmp_path / "shards"
        _write_shards(layer_dir, [4, 4, 2])
        memmaps, cum = open_layer_shards(layer_dir)
        # Restrict to even global rows: [0, 2, 4, 6, 8]
        return ShardedRowView(memmaps, cum, np.array([0, 2, 4, 6, 8], dtype=np.int64))

    def test_len_shape_dtype(self, view):
        assert len(view) == 5
        assert view.shape == (5, D_MODEL)
        assert view.dtype == np.float32

    def test_scalar_int_returns_1d(self, view):
        row = view[1]
        assert row.shape == (D_MODEL,)
        assert np.all(row == 2.0)

    def test_negative_int(self, view):
        assert np.all(view[-1] == 8.0)
        assert np.all(view[-5] == 0.0)

    def test_out_of_range_int_raises(self, view):
        with pytest.raises(IndexError):
            view[5]
        with pytest.raises(IndexError):
            view[-6]

    def test_slice_across_shard_boundary(self, view):
        # global rows 2, 4, 6 span shards 0, 1 (boundary at 4)
        out = view[1:4]
        assert out.shape == (3, D_MODEL)
        assert np.array_equal(out[:, 0], np.array([2.0, 4.0, 6.0]))

    def test_slice_final_partial_shard(self, view):
        # global row 8 lives in the last (partial, 2-row) shard
        out = view[3:5]
        assert np.array_equal(out[:, 0], np.array([6.0, 8.0]))

    def test_empty_slice(self, view):
        out = view[2:2]
        assert out.shape == (0, D_MODEL)
        assert out.dtype == np.float32

    def test_stepped_and_negative_slices(self, view):
        ref = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        assert np.array_equal(view[::2][:, 0], ref[::2])
        assert np.array_equal(view[::-1][:, 0], ref[::-1])
        assert np.array_equal(view[-3:][:, 0], ref[-3:])

    def test_result_is_writable(self, view):
        out = view[0:2]
        out[0, 0] = 123.0  # must not raise (fresh copy, not a memmap view)

    def test_invalid_key_type(self, view):
        with pytest.raises(TypeError):
            view["foo"]
        with pytest.raises(TypeError):
            view[[0, 1]]


class TestLoadActivationsFromShards:
    def test_dedup_keeps_first_occurrence_order(self, tmp_path):
        # 10 rows: prompt "a" repeats at rows 0/3/9, "b" at 1/4, rest unique
        prompts = ["a", "b", "c", "a", "b", "d", "e", "f", "g", "a"]
        act_dir = _write_activations_dir(tmp_path, prompts, [4, 4, 2])

        unique_prompts, view = load_activations_from_shards(act_dir)

        assert unique_prompts == ["a", "b", "c", "d", "e", "f", "g"]
        assert view.shape == (7, D_MODEL)
        # Each unique prompt maps to its FIRST occurrence row
        expected_rows = [0, 1, 2, 5, 6, 7, 8]
        ref = _eager_reference([4, 4, 2])
        assert np.array_equal(view[0 : len(view)], ref[expected_rows])
        for i, row in enumerate(expected_rows):
            assert np.all(view[i] == float(row))

    def test_matches_eager_reference_when_all_unique(self, tmp_path):
        prompts = [f"p{i}" for i in range(10)]
        act_dir = _write_activations_dir(tmp_path, prompts, [4, 4, 2])

        unique_prompts, view = load_activations_from_shards(act_dir)

        assert unique_prompts == prompts
        assert np.array_equal(view[0:10], _eager_reference([4, 4, 2]))

    def test_prompt_shard_length_mismatch_raises(self, tmp_path):
        prompts = ["a", "b", "c"]  # 3 prompts vs 10 shard rows
        act_dir = _write_activations_dir(tmp_path, prompts, [4, 4, 2])

        with pytest.raises(ValueError, match="3 entries but shards"):
            load_activations_from_shards(act_dir)

    def test_missing_metadata_raises(self, tmp_path):
        act_dir = tmp_path / "activations"
        _write_shards(act_dir, [2])
        with pytest.raises(FileNotFoundError, match="metadata.json"):
            load_activations_from_shards(act_dir)


class TestCollectMaxActivatingExamplesSmoke:
    def test_streaming_topk_with_sharded_view(self, tmp_path):
        """End-to-end 4b smoke test on CPU with a tiny SAE and a ShardedRowView."""
        d_sae = 8
        prompts = [f"p{i}" for i in range(10)] + ["p0", "p1"]  # 2 duplicates
        act_dir = _write_activations_dir(tmp_path, prompts, [5, 5, 2])

        # Tiny identity-ish SAE: feature j reads coordinate j % D_MODEL
        w_enc = torch.zeros(D_MODEL, d_sae)
        for j in range(d_sae):
            w_enc[j % D_MODEL, j] = 1.0
        checkpoint = {
            "config": {"d_model": D_MODEL, "d_sae": d_sae, "dtype": "float32", "rms_scale": 1.0},
            "model_state_dict": {
                "W_enc": w_enc,
                "b_enc": torch.zeros(d_sae),
                "threshold": torch.zeros(d_sae),
                "W_dec": torch.zeros(d_sae, D_MODEL),
                "b_dec": torch.zeros(D_MODEL),
            },
        }
        ckpt_path = tmp_path / "sae.pt"
        torch.save(checkpoint, ckpt_path)

        unique_prompts, view = load_activations_from_shards(act_dir)
        assert len(unique_prompts) == 10

        results = collect_max_activating_examples(
            prompts=unique_prompts,
            activations=view,
            sae_checkpoint=str(ckpt_path),
            feature_indices=[0, 3],
            top_n=3,
            bottom_n=2,
        )

        assert set(results.keys()) == {0, 3}
        # Row i is filled with value i, so feature activation == row value and
        # the top prompts are the highest-index rows in descending order.
        top_prompts = [ex["prompt"] for ex in results[0]["top"]]
        assert top_prompts == ["p9", "p8", "p7"]
        assert results[0]["top"][0]["activation"] == pytest.approx(9.0)
        assert results[0]["max_activation"] == pytest.approx(9.0)
        # Row 0 activates at exactly 0 (JumpReLU threshold 0 keeps it at 0)
        assert results[0]["frac_nonzero"] == pytest.approx(9 / 10)
