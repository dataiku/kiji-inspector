#!/bin/bash
set -euo pipefail

VLLM_DIR="$(uv run python - <<'PY'
import sys
import pathlib
try:
    import vllm
except ImportError:
    sys.stderr.write('Error: vllm is not installed in the current Python environment.\n')
    sys.exit(1)
vllm_path = pathlib.Path(vllm.__file__).resolve().parent
print(vllm_path)
PY
)"
PATCH_SRC_DIR="patches"
PATCH_DST_DIR="${VLLM_DIR}/patches"
PATCH_NAME="0001-patch-venv.mbox.patch"
PATCH_SRC_FILE="${PATCH_SRC_DIR}/${PATCH_NAME}"
PATCH_REL_FILE="patches/${PATCH_NAME}"

if [[ ! -d "$VLLM_DIR" ]]; then
  echo "Error: vllm directory not found: $VLLM_DIR" >&2
  exit 1
fi

if [[ ! -d "$PATCH_SRC_DIR" ]]; then
  echo "Error: patches directory not found: $PATCH_SRC_DIR" >&2
  exit 1
fi

if [[ ! -f "$PATCH_SRC_FILE" ]]; then
  echo "Error: patch file not found: $PATCH_SRC_FILE" >&2
  exit 1
fi

rm -rf "$PATCH_DST_DIR"
cp -r "$PATCH_SRC_DIR" "$PATCH_DST_DIR"

cd "$VLLM_DIR"

if patch -p1 --reverse --dry-run < "$PATCH_REL_FILE" >/dev/null 2>&1; then
  patch -p1 --reverse < "$PATCH_REL_FILE"
  echo "Patch reverted successfully."
elif patch -p1 --dry-run < "$PATCH_REL_FILE" >/dev/null 2>&1; then
  echo "Patch is not currently applied. Skipping revert."
else
  echo "Patch could not be reverted cleanly. The installed vllm version may differ from the patch." >&2
  exit 1
fi
