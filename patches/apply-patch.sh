#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_SRC_DIR="$SCRIPT_DIR"
PATCH_NAME="extract-hiddenstates-gemma3-nemotron.patch"
PATCH_SRC_FILE="${PATCH_SRC_DIR}/${PATCH_NAME}"

VLLM_DIR="$(uv run python - <<'PY'
import pathlib
import sys
try:
    import vllm
except ImportError:
    sys.stderr.write('Error: vllm is not installed in the current Python environment.\n')
    sys.exit(1)
vllm_path = pathlib.Path(vllm.__file__).resolve().parent
print(vllm_path)
PY
)"

PATCH_DST_DIR="${VLLM_DIR}/patches"
PATCH_REL_FILE="patches/${PATCH_NAME}"

if [[ ! -d "$VLLM_DIR" ]]; then
  echo "Error: vllm directory not found: $VLLM_DIR" >&2
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
  echo "Patch is already applied. Skipping."
  exit 0
fi

# `-N` ignores hunks that are already applied, which lets us complete
# partially applied patch states without failing on duplicate hunks.
if patch -p1 -N --dry-run < "$PATCH_REL_FILE" >/dev/null 2>&1; then
  patch -p1 -N < "$PATCH_REL_FILE"
  echo "Patch applied successfully."
else
  echo "Patch could not be applied cleanly. The installed vllm version may differ from the patch." >&2
  exit 1
fi
