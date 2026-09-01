#!/usr/bin/env python3
"""Record exactly which weights produced the published results.

``hf_hub_download`` resolves ``main`` unless told otherwise, so "the SAE for
layer 43" is not a reproducible reference: the repo can gain a corrected
checkpoint and nothing in the run output would say which version was read.
The base model shows this is not hypothetical --- the runs used ``d468880b``
and upstream ``main`` has since moved past it.

The loader now pins revisions (``core/registry.py``); this writes the matching
evidence: per-file SHA-256 for the SAE checkpoints and for every shard of the
stripped base checkpoint, so a reader can verify they hold the same bytes.

Checksums come from the Hugging Face cache, where a downloaded LFS file is
stored as a blob named by its own SHA-256 --- confirmed against a full
``sha256sum`` of a 5 GB shard before relying on it. Files that
``strip_mtp`` rewrote are not hardlinked into the cache and are hashed here.

Usage::

    python paper/steering_workshop/provenance.py \\
        --model-dir ~/models/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16-no-mtp \\
        --hf-cache /ephemeral/cache/huggingface/hub
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kiji_inspector.core.registry import (  # noqa: E402
    BASE_MODEL_REVISIONS,
    MODEL_REVISIONS,
)

OUT = Path(__file__).resolve().parent / "artifacts" / "provenance.json"
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16"
SAE_REPO = "575-lab/kiji-inspector-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
LAYERS = (6, 13, 20, 27, 34, 43)


def _cache_dir(cache: Path, repo: str) -> Path:
    return cache / ("models--" + repo.replace("/", "--"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(name: str) -> bool:
    return len(name) == 64 and set(name) <= set("0123456789abcdef")


def _blobs_by_inode(cache: Path, repo: str) -> dict[int, str]:
    """Map inode -> SHA-256, for the cached blobs that are named by one.

    Hugging Face stores an LFS file as a blob named by its SHA-256, which is
    why the shard checksums here cost nothing to collect. Small files are
    ordinary git objects named by a 40-hex SHA-1 instead, so those are filtered
    out and hashed directly rather than recorded under the wrong algorithm.
    """
    blobs = _cache_dir(cache, repo) / "blobs"
    if not blobs.is_dir():
        return {}
    return {
        b.stat().st_ino: b.name for b in blobs.iterdir() if b.is_file() and _is_sha256(b.name)
    }


def _checked_revision(cache: Path, repo: str, expected: str | None) -> str:
    """The revision in the cache, cross-checked against the pinned one."""
    ref = _cache_dir(cache, repo) / "refs" / "main"
    seen = ref.read_text().strip() if ref.is_file() else None
    if expected and seen and expected != seen:
        raise SystemExit(
            f"{repo}: registry pins {expected} but the cache holds {seen}.\n"
            "Re-download at the pinned revision, or update the pin deliberately."
        )
    resolved = expected or seen
    if not resolved:
        raise SystemExit(f"{repo}: no pinned revision and none in the cache.")
    return resolved


def sae_files(cache: Path) -> dict:
    revision = _checked_revision(cache, SAE_REPO, MODEL_REVISIONS.get(SAE_REPO))
    snapshot = _cache_dir(cache, SAE_REPO) / "snapshots" / revision
    by_inode = _blobs_by_inode(cache, SAE_REPO)
    files = {}
    for layer in LAYERS:
        rel = f"layer_{layer}/sae_checkpoints/sae_final.pt"
        path = snapshot / rel
        if not path.exists():
            continue
        resolved = path.resolve()
        files[rel] = {
            "sha256": by_inode.get(resolved.stat().st_ino) or _sha256(resolved),
            "bytes": resolved.stat().st_size,
        }
    return {"repo": SAE_REPO, "revision": revision, "files": files}


def base_model_files(model_dir: Path, cache: Path) -> dict:
    revision = _checked_revision(cache, BASE_MODEL, BASE_MODEL_REVISIONS.get(BASE_MODEL))
    by_inode = _blobs_by_inode(cache, BASE_MODEL)
    files, rewritten = {}, []
    for path in sorted(model_dir.iterdir()):
        if not path.is_file():
            continue
        stat = path.stat()
        # A file hardlinked into the cache came from the download untouched,
        # whether or not its blob name gave us a checksum for free.
        linked = stat.st_nlink > 1
        if not linked:
            rewritten.append(path.name)
        files[path.name] = {
            "sha256": by_inode.get(stat.st_ino) or _sha256(path),
            "bytes": stat.st_size,
            "source": "upstream" if linked else "strip_mtp",
        }
    return {
        "repo": BASE_MODEL,
        "revision": revision,
        "derivation": {
            "tool": "kiji_inspector.utils.strip_mtp",
            "what": "removes the multi-token-prediction draft stack; kept shards "
                    "are hardlinked from the download, so their bytes are upstream's",
            "rewritten": rewritten,
        },
        "files": files,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True, help="stripped base checkpoint")
    ap.add_argument("--hf-cache", type=Path, required=True, help="huggingface hub cache")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    record = {
        "note": "Weights behind the published results. Revisions are pinned in "
                "src/kiji_inspector/core/registry.py and loaded by default.",
        "sae": sae_files(args.hf_cache.expanduser()),
        "baseModel": base_model_files(args.model_dir.expanduser(), args.hf_cache.expanduser()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
