#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_SRC_DIR="$SCRIPT_DIR"

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

if [[ ! -d "$VLLM_DIR" ]]; then
  echo "Error: vllm directory not found: $VLLM_DIR" >&2
  exit 1
fi

mapfile -t PATCH_FILES < <(find "$PATCH_SRC_DIR" -maxdepth 1 -type f -name '*.patch' | sort)

if [[ "${#PATCH_FILES[@]}" -eq 0 ]]; then
  echo "Error: no patch files found in $PATCH_SRC_DIR" >&2
  exit 1
fi

rm -rf "$PATCH_DST_DIR"
cp -r "$PATCH_SRC_DIR" "$PATCH_DST_DIR"

cd "$VLLM_DIR"

all_already_applied=true

for patch_file in "${PATCH_FILES[@]}"; do
  patch_name="$(basename "$patch_file")"
  patch_rel_file="patches/${patch_name}"

  if patch -p1 --reverse --dry-run < "$patch_rel_file" >/dev/null 2>&1; then
    echo "Patch already applied: $patch_name"
    continue
  fi

  all_already_applied=false

  set +e
  patch_output="$(patch -p1 -N < "$patch_rel_file" 2>&1)"
  patch_status=$?
  set -e

  printf '%s\n' "$patch_output"

  if [[ "$patch_status" -eq 0 ]]; then
    echo "Patch applied: $patch_name"
    continue
  fi

  if patch -p1 --reverse --dry-run < "$patch_rel_file" >/dev/null 2>&1; then
    echo "Patch applied with already-present hunks: $patch_name"
    find . -name '*.rej' -delete
    continue
  fi

  echo "Patch could not be applied cleanly: $patch_name" >&2
  exit 1
done

if [[ "$all_already_applied" == true ]]; then
  echo "All patches are already applied. Skipping."
else
  echo "All patches applied successfully."
fi
