#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/vllm-0.20.1-gemma4-hidden-states.patch"
EXPECTED_VERSION="0.20.1"

if [[ ! -f "${PATCH_FILE}" ]]; then
    echo "ERROR: patch file not found at ${PATCH_FILE}" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: '${PYTHON_BIN}' not found on PATH. Set PYTHON=... to override." >&2
    exit 1
fi

VLLM_DIR="$("${PYTHON_BIN}" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
if [[ -z "${VLLM_DIR}" || ! -d "${VLLM_DIR}" ]]; then
    echo "ERROR: could not locate the installed vllm package via ${PYTHON_BIN}." >&2
    echo "       Make sure your virtualenv is activated, or set PYTHON=/path/to/python." >&2
    exit 1
fi

VLLM_VERSION="$("${PYTHON_BIN}" -c 'import vllm; print(getattr(vllm, "__version__", "unknown"))')"

echo "Detected vllm install:"
echo "  python : ${PYTHON_BIN}"
echo "  path   : ${VLLM_DIR}"
echo "  version: ${VLLM_VERSION}"
echo "  patch  : ${PATCH_FILE}"

if [[ "${VLLM_VERSION}" != "${EXPECTED_VERSION}" ]]; then
    echo "WARNING: installed vllm version (${VLLM_VERSION}) does not match the patch's expected version (${EXPECTED_VERSION})." >&2
    if [[ "${FORCE:-0}" != "1" ]]; then
        echo "         Re-run with FORCE=1 to apply anyway." >&2
        exit 1
    fi
fi

if ! command -v patch >/dev/null 2>&1; then
    echo "ERROR: 'patch' utility not found on PATH. Install it (e.g. apt-get install patch)." >&2
    exit 1
fi

echo
echo "Dry-run check (no files modified yet)..."
if ! patch -p1 -d "${VLLM_DIR}" --dry-run --forward < "${PATCH_FILE}"; then
    echo "ERROR: dry-run failed. The patch does not apply cleanly to ${VLLM_DIR}." >&2
    echo "       If the patch is already applied, you should be safe to ignore." >&2
    exit 1
fi

echo
echo "Applying patch..."
patch -p1 -d "${VLLM_DIR}" --forward < "${PATCH_FILE}"

echo
echo "Done. Patch applied to ${VLLM_DIR}."
