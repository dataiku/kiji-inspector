"""
Strip the multi-token-prediction (MTP) draft stack from a checkpoint.

Some checkpoints ship an auxiliary MTP draft stack
(``num_nextn_predict_layers > 0``) whose weights live under the
``mtp.`` prefix. The kiji-inspector pipeline never uses it, and the
extractor loads models with its own ``speculative_config``, so we run
against a stripped variant as a checkpoint-compatibility precaution.

The stripped variant is written next to the source as a new directory:
kept shard files are hardlinked (falling back to copy across
filesystems), so the variant costs almost no extra disk. Note that
hardlinked files share inodes — modifying weights in either directory
modifies both.

Only checkpoints whose MTP tensors live in their own shard(s) are
supported; a shard mixing MTP and non-MTP tensors would require a
multi-GB rewrite and is rejected

Usage:
    uv run python -m kiji_inspector.utils.strip_mtp SRC_DIR DST_DIR
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

MTP_PREFIX = "mtp."
INDEX_NAME = "model.safetensors.index.json"
# Repo metadata not worth carrying into the stripped variant.
EXCLUDED_REPO_FILES = {"README.md", ".gitattributes"}
# Config keys that describe the MTP stack (beyond num_nextn_predict_layers).
MTP_CONFIG_KEYS = ("mtp_hybrid_override_pattern", "mtp_layers_block_type")


def _read_safetensors_header(path: Path) -> dict:
    """Read the JSON header of a safetensors file (no tensor data loaded)."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(header_len))


def _tensor_bytes(path: Path, names: set[str]) -> int:
    """Exact data bytes occupied by *names* in the shard at *path*."""
    header = _read_safetensors_header(path)
    total = 0
    for name in names:
        begin, end = header[name]["data_offsets"]
        total += end - begin
    return total


def _link_or_copy(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def strip_mtp(src_dir: str | os.PathLike, dst_dir: str | os.PathLike) -> dict:
    """Write an MTP-free variant of *src_dir* to *dst_dir*.

    Returns a summary dict: ``tensors_removed``, ``shards_dropped``,
    ``bytes_saved``.

    Raises:
        ValueError: On a missing/invalid source, an existing destination, a
            checkpoint without MTP tensors (or with nothing else), or a shard
            that mixes MTP and non-MTP tensors.
    """
    src = Path(src_dir).expanduser().resolve()
    dst = Path(dst_dir).expanduser().resolve()

    if not src.is_dir():
        raise ValueError(f"Source model directory not found: {src}")
    index_path = src / INDEX_NAME
    if not index_path.is_file():
        raise ValueError(f"No {INDEX_NAME} in {src} — sharded safetensors expected.")
    if dst.exists():
        raise ValueError(f"Destination already exists, refusing to overwrite: {dst}")
    if src == dst or src in dst.parents:
        raise ValueError(f"Destination must not be inside the source directory: {dst}")

    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    mtp_weights = {k: v for k, v in weight_map.items() if k.startswith(MTP_PREFIX)}
    kept_weights = {k: v for k, v in weight_map.items() if not k.startswith(MTP_PREFIX)}

    if not mtp_weights:
        raise ValueError(f"No '{MTP_PREFIX}*' tensors in {index_path} — nothing to strip.")
    if not kept_weights:
        raise ValueError(f"All tensors in {index_path} are '{MTP_PREFIX}*' — refusing.")

    mtp_shards = set(mtp_weights.values())
    kept_shards = set(kept_weights.values())
    mixed = sorted(mtp_shards & kept_shards)
    if mixed:
        raise ValueError(
            f"Shard(s) mix MTP and non-MTP tensors: {', '.join(mixed)}. "
            "Rewriting mixed shards is not supported — this tool only drops "
            "shards that contain exclusively MTP tensors."
        )
    dropped_shards = sorted(mtp_shards)

    bytes_saved = sum(
        _tensor_bytes(src / shard, {k for k, v in mtp_weights.items() if v == shard})
        for shard in dropped_shards
    )

    # Build in a sibling temp dir (same filesystem, so hardlinks work), then
    # atomically rename into place. A failed run leaves nothing behind.
    entries = [e for e in sorted(src.iterdir()) if e.name not in EXCLUDED_REPO_FILES]
    tmp = Path(tempfile.mkdtemp(prefix=f".{dst.name}.tmp-", dir=dst.parent))
    try:
        for entry in entries:
            if entry.name in (INDEX_NAME, "config.json") or entry.name in dropped_shards:
                continue
            _link_or_copy(entry, tmp / entry.name)

        new_index = dict(index)
        new_index["weight_map"] = kept_weights
        metadata = dict(index.get("metadata") or {})
        if "total_size" in metadata:
            metadata["total_size"] = metadata["total_size"] - bytes_saved
        if metadata or "metadata" in index:
            new_index["metadata"] = metadata
        (tmp / INDEX_NAME).write_text(json.dumps(new_index, indent=2) + "\n")

        config = json.loads((src / "config.json").read_text())
        config["num_nextn_predict_layers"] = 0
        for key in MTP_CONFIG_KEYS:
            config.pop(key, None)
        (tmp / "config.json").write_text(json.dumps(config, indent=2) + "\n")

        missing = [s for s in sorted(kept_shards) if not (tmp / s).is_file()]
        if missing:
            raise RuntimeError(f"Kept shard(s) missing from output: {', '.join(missing)}")

        os.rename(tmp, dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    return {
        "tensors_removed": len(mtp_weights),
        "shards_dropped": len(dropped_shards),
        "bytes_saved": bytes_saved,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Strip the mtp.* draft stack from a sharded safetensors checkpoint."
    )
    parser.add_argument("src_dir", help="Source model directory (with safetensors index).")
    parser.add_argument("dst_dir", help="Destination directory (must not exist).")
    args = parser.parse_args(argv)

    try:
        summary = strip_mtp(args.src_dir, args.dst_dir)
    except ValueError as e:
        sys.exit(f"error: {e}")

    print(f"Stripped MTP stack: {args.src_dir} -> {args.dst_dir}")
    print(f"  tensors removed : {summary['tensors_removed']}")
    print(f"  shards dropped  : {summary['shards_dropped']}")
    print(f"  bytes saved     : {summary['bytes_saved']:,}")
    print(
        "  Note: kept shards are hardlinked where possible — they share inodes "
        "with the source; modifying weights in either directory modifies both."
    )


if __name__ == "__main__":
    main()
