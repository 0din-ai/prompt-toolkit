#!/usr/bin/env bash
#
# 0DIN Prompt Toolkit — Offline Installer
#
# Installs odin-prompt-toolkit from locally-bundled wheel files.
# Run this on the recipient machine (no internet required for the SDK itself;
# pip dependencies are fetched from PyPI unless a deps/ directory is present).
#
# Usage:
#   bash install.sh [--venv DIR] [--no-venv]
#
# Options:
#   --venv DIR    Create and install into a virtual environment at DIR
#                 (default: ./venv if not already inside a venv)
#   --no-venv     Skip venv creation, install into the current Python environment
#
# What this script does:
#   1. Checks Python >= 3.10
#   2. Creates a venv (unless --no-venv or already in one)
#   3. Installs the pure-Python SDK wheel with [onnx] extras
#      (fetches onnxruntime, sentence-transformers, numpy from PyPI)
#   4. Installs the native acceleration wheel if present in sdk/
#   5. Copies the bundled ONNX model to ~/.cache/odin-prompt-toolkit/models/v1/
#   6. Runs verify.py smoke test
#

set -euo pipefail

# ── colors ────────────────────────────────────────────────────────────────────
CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
RESET='\033[0m'

info()    { echo -e "${CYAN}  →  $*${RESET}"; }
success() { echo -e "${GREEN}  ✓  $*${RESET}"; }
warn()    { echo -e "${YELLOW}  ⚠  $*${RESET}"; }
error()   { echo -e "${RED}  ✗  $*${RESET}" >&2; exit 1; }

# ── configuration ─────────────────────────────────────────────────────────────
REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=10
MODEL_CACHE_DIR="${HOME}/.cache/odin-prompt-toolkit/models/v1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── argument parsing ──────────────────────────────────────────────────────────
VENV_DIR="${SCRIPT_DIR}/venv"
USE_VENV=true    # may be flipped below

while [[ $# -gt 0 ]]; do
    case $1 in
        --venv)    VENV_DIR="$2"; USE_VENV=true;  shift 2 ;;
        --no-venv) USE_VENV=false;                shift   ;;
        --help)
            sed -n '2,20p' "$0" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *) error "Unknown option: $1 (use --help for usage)" ;;
    esac
done

# If we're already inside a virtual environment, default to using it
if [[ -n "${VIRTUAL_ENV:-}" ]] && [[ "$USE_VENV" == "true" ]]; then
    warn "Already inside a virtual environment ($VIRTUAL_ENV) — skipping venv creation"
    USE_VENV=false
fi

# ── banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}║       0DIN Prompt Toolkit — Offline Installer            ║${RESET}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── step 1: check python ──────────────────────────────────────────────────────
info "Checking Python version..."

if ! command -v python3 &>/dev/null; then
    error "python3 not found. Install Python ${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}+ and re-run."
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if (( PYTHON_MAJOR < REQUIRED_PYTHON_MAJOR )) || \
   (( PYTHON_MAJOR == REQUIRED_PYTHON_MAJOR && PYTHON_MINOR < REQUIRED_PYTHON_MINOR )); then
    error "Python ${PYTHON_VERSION} found, but >= ${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR} required."
fi

success "Python ${PYTHON_VERSION}"

# ── step 2: locate SDK wheel ──────────────────────────────────────────────────
info "Locating SDK wheel..."

SDK_DIR="${SCRIPT_DIR}/sdk"
[[ -d "$SDK_DIR" ]] || error "sdk/ directory not found next to install.sh (expected: ${SDK_DIR})"

PURE_WHEEL=$(find "$SDK_DIR" -name "odin_prompt_toolkit-*-py3-none-any.whl" | sort -V | tail -1)
[[ -n "$PURE_WHEEL" ]] || error "No odin_prompt_toolkit-*-py3-none-any.whl found in ${SDK_DIR}"

success "SDK wheel: $(basename "$PURE_WHEEL")"

# Detect native wheel for this platform (best-effort)
PYTHON_TAG="cp${PYTHON_MAJOR}${PYTHON_MINOR}"
OS_RAW=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH_RAW=$(uname -m)

case "$OS_RAW" in
    linux)  PLAT_GLOB="manylinux*" ;;
    darwin) PLAT_GLOB="macosx*"    ;;
    mingw*|msys*|cygwin*) PLAT_GLOB="win*" ;;
    *) PLAT_GLOB="*" ;;
esac

case "$ARCH_RAW" in
    x86_64|amd64)   ARCH_GLOB="x86_64"  ;;
    aarch64|arm64)  ARCH_GLOB="arm64\|aarch64" ;;
    *)              ARCH_GLOB="$ARCH_RAW" ;;
esac

NATIVE_WHEEL=$(find "$SDK_DIR" \
    -name "odin_prompt_toolkit_native-*-${PYTHON_TAG}-*-${PLAT_GLOB}*.whl" 2>/dev/null \
    | grep -E "${ARCH_GLOB}" | sort -V | tail -1 || true)

if [[ -n "$NATIVE_WHEEL" ]]; then
    success "Native wheel: $(basename "$NATIVE_WHEEL")"
else
    warn "No native wheel found for this platform (${OS_RAW}/${ARCH_RAW}/${PYTHON_TAG}) — pure Python will be used"
fi

# ── step 3: set up python environment ────────────────────────────────────────
if [[ "$USE_VENV" == "true" ]]; then
    info "Creating virtual environment at ${VENV_DIR}..."

    if [[ -d "$VENV_DIR" ]]; then
        warn "Virtual environment already exists at ${VENV_DIR} — reusing it"
    else
        python3 -m venv "$VENV_DIR"
        success "Virtual environment created"
    fi

    PIP="${VENV_DIR}/bin/pip"
    PYTHON="${VENV_DIR}/bin/python"
else
    # Use whatever pip/python is active
    PYTHON="$(command -v python3)"
    PIP="$(command -v pip3 2>/dev/null || echo "$PYTHON -m pip")"
    # Normalize: always invoke as array-safe "$PYTHON -m pip" path
    PIP="$PYTHON -m pip"
fi

# Ensure pip is up to date (suppresses "new pip version available" noise)
$PYTHON -m pip install --quiet --upgrade pip

# ── step 4: check for offline deps/ folder ───────────────────────────────────
# If the operator has pre-staged pip wheels in deps/ (for fully air-gapped
# environments), use them. Otherwise pip fetches from PyPI as normal.
DEPS_DIR="${SCRIPT_DIR}/deps"
PIP_EXTRA_ARGS=()
if [[ -d "$DEPS_DIR" ]] && [[ -n "$(ls -A "$DEPS_DIR" 2>/dev/null)" ]]; then
    info "Found deps/ directory — using offline pip wheels"
    PIP_EXTRA_ARGS=(--no-index --find-links "$DEPS_DIR")
else
    info "No deps/ directory found — fetching dependencies from PyPI"
fi

# ── step 5: install SDK wheel with onnx extras ───────────────────────────────
info "Installing odin-prompt-toolkit[onnx]..."

$PYTHON -m pip install --quiet "${PIP_EXTRA_ARGS[@]}" "${PURE_WHEEL}[onnx]"
success "SDK installed (with onnxruntime + sentence-transformers)"

# ── step 6: install native wheel (optional) ──────────────────────────────────
if [[ -n "$NATIVE_WHEEL" ]]; then
    info "Installing native acceleration wheel..."
    $PYTHON -m pip install --quiet "${PIP_EXTRA_ARGS[@]}" "$NATIVE_WHEEL"
    success "Native acceleration installed (~653× faster)"
fi

# ── step 7: install ONNX model ───────────────────────────────────────────────
info "Installing ONNX model..."

MODEL_SRC="${SCRIPT_DIR}/model/v1"
REQUIRED_MODEL_FILES=("onnx/model_O4.onnx" "tokenizer.json" "config.json")

for f in "${REQUIRED_MODEL_FILES[@]}"; do
    [[ -f "${MODEL_SRC}/${f}" ]] || \
        error "Bundled model file missing: model/v1/${f}\n       Re-package the bundle with the model files included."
done

mkdir -p "${MODEL_CACHE_DIR}/onnx"
cp "${MODEL_SRC}/onnx/model_O4.onnx" "${MODEL_CACHE_DIR}/onnx/"
cp "${MODEL_SRC}/tokenizer.json"      "${MODEL_CACHE_DIR}/"
cp "${MODEL_SRC}/config.json"         "${MODEL_CACHE_DIR}/"

MODEL_SIZE=$(du -sh "${MODEL_CACHE_DIR}/onnx/model_O4.onnx" | cut -f1)
success "Model installed to ${MODEL_CACHE_DIR} (${MODEL_SIZE})"

# ── step 8: smoke test ────────────────────────────────────────────────────────
info "Running post-install verification..."

VERIFY_SCRIPT="${SCRIPT_DIR}/verify.py"
if [[ -f "$VERIFY_SCRIPT" ]]; then
    if $PYTHON "$VERIFY_SCRIPT"; then
        success "All verification checks passed"
    else
        error "Verification failed — installation may be incomplete.\n       Review the output above and contact your 0DIN representative."
    fi
else
    warn "verify.py not found — skipping smoke test"
fi

# ── done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}║                  Installation Complete!                     ║${RESET}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""

if [[ "$USE_VENV" == "true" ]]; then
    echo -e "${CYAN}Activate the virtual environment:${RESET}"
    echo "  source ${VENV_DIR}/bin/activate"
    echo ""
fi

echo -e "${CYAN}Quick test:${RESET}"
echo "  python -c \"from odin_prompt_toolkit import sign_text; print('odin-prompt-toolkit ready')\""
echo ""
