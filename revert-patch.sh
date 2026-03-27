#!/bin/bash
set -euo pipefail

VLLM_DIR=".venv/lib/python3.12/site-packages/vllm"
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

if git apply --reverse --check "$PATCH_REL_FILE" >/dev/null 2>&1; then
  git apply --reverse "$PATCH_REL_FILE"
  echo "Patch reverted successfully."
elif git apply --check "$PATCH_REL_FILE" >/dev/null 2>&1; then
  echo "Patch is not currently applied. Skipping revert."
else
  echo "Patch could not be reverted cleanly. The installed vllm version may differ from the patch." >&2
  exit 1
fi
