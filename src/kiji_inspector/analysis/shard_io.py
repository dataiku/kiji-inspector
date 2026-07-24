"""Memmap-based access to Step 1 activation shards.

Shared by the contrastive-feature analysis (Step 3) and feature
interpretation (Step 4) so that neither path materializes the full
activation matrix in RAM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def open_layer_shards(
    layer_dir: Path,
) -> tuple[list[np.memmap], np.ndarray]:
    """Open all shard files in a layer dir as memmaps (no RAM materialization).

    Returns:
        (memmaps, cum_offsets) — ``cum_offsets[i]`` is the global row index of
        the first row in shard ``i``, and ``cum_offsets[-1]`` is the total row
        count across all shards.
    """
    shard_paths = sorted(layer_dir.glob("shard_*.npy"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.npy files in {layer_dir}")
    memmaps = [np.load(p, mmap_mode="r") for p in shard_paths]

    first_shape = memmaps[0].shape
    if len(first_shape) != 2:
        raise ValueError(f"Shard {shard_paths[0]} must be 2-D, got shape {first_shape}")

    expected_d_model = first_shape[1]
    expected_dtype = memmaps[0].dtype
    for path, memmap in zip(shard_paths[1:], memmaps[1:], strict=True):
        if memmap.ndim != 2:
            raise ValueError(f"Shard {path} must be 2-D, got shape {memmap.shape}")
        if memmap.shape[1] != expected_d_model:
            raise ValueError(
                f"Shard {path} has d_model={memmap.shape[1]}, expected {expected_d_model}"
            )
        if memmap.dtype != expected_dtype:
            raise ValueError(f"Shard {path} has dtype={memmap.dtype}, expected {expected_dtype}")

    cum = np.zeros(len(memmaps) + 1, dtype=np.int64)
    for i, m in enumerate(memmaps):
        cum[i + 1] = cum[i] + m.shape[0]
    return memmaps, cum


def gather_rows(
    memmaps: list[np.memmap],
    cum_offsets: np.ndarray,
    global_indices: np.ndarray,
) -> np.ndarray:
    """Gather rows at ``global_indices`` from a list of virtually-concatenated memmaps.

    Only the requested rows are paged in from disk; the full array is never
    materialized in RAM. The result is a fresh writable array.
    """
    d_model = memmaps[0].shape[1]
    total_rows = int(cum_offsets[-1])
    if len(global_indices) and (
        int(global_indices.min()) < 0 or int(global_indices.max()) >= total_rows
    ):
        raise IndexError(
            f"Activation row index out of range: requested "
            f"[{int(global_indices.min())}, {int(global_indices.max())}] but shards "
            f"hold {total_rows} rows. A shard file is likely missing or truncated, "
            "or this layer's shards don't match the layer the indices were built from."
        )
    out = np.empty((len(global_indices), d_model), dtype=memmaps[0].dtype)
    for shard_idx, m in enumerate(memmaps):
        lo, hi = int(cum_offsets[shard_idx]), int(cum_offsets[shard_idx + 1])
        mask = (global_indices >= lo) & (global_indices < hi)
        if mask.any():
            local = global_indices[mask] - lo
            out[mask] = m[local]
    return out


class ShardedRowView:
    """Read-only, row-indexable view over memmapped shards restricted to a
    subset of global row indices.

    Behaves like a 2-D array of shape ``(len(row_indices), d_model)`` for the
    access patterns the pipeline uses: ``len()``, ``.shape``, ``.dtype``, and
    integer/slice indexing. Every access returns a fresh writable ndarray
    gathered from the memmaps; only the requested rows are paged in.
    """

    def __init__(
        self,
        memmaps: list[np.memmap],
        cum_offsets: np.ndarray,
        row_indices: np.ndarray,
    ):
        self._memmaps = memmaps
        self._cum_offsets = cum_offsets
        self._row_indices = np.asarray(row_indices, dtype=np.int64)
        self._d_model = int(memmaps[0].shape[1])

    def __len__(self) -> int:
        return len(self._row_indices)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._row_indices), self._d_model)

    @property
    def dtype(self) -> np.dtype:
        return self._memmaps[0].dtype

    def __getitem__(self, key: int | slice) -> np.ndarray:
        if isinstance(key, (int, np.integer)):
            n = len(self._row_indices)
            idx = int(key)
            if idx < 0:
                idx += n
            if not 0 <= idx < n:
                raise IndexError(f"row index {int(key)} out of range for {n} rows")
            return gather_rows(self._memmaps, self._cum_offsets, self._row_indices[idx : idx + 1])[
                0
            ]
        if isinstance(key, slice):
            selected = self._row_indices[key]
            if len(selected) == 0:
                return np.empty((0, self._d_model), dtype=self.dtype)
            return gather_rows(self._memmaps, self._cum_offsets, selected)
        raise TypeError(f"ShardedRowView indices must be int or slice, not {type(key).__name__}")
