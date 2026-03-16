#!/usr/bin/env bash
#
# Build a offline bundle for odin-prompt-toolkit.
#
# Run this on a build machine (internet access OK). The output directory is
# designed to be copied to a USB drive and handed to a recipient who has no
# internet access during installation.
#
# Usage:
#   ./package-offline.sh [OPTIONS]
#
# Options:
#   --version VERSION        SDK version (default: read from packages/python/pyproject.toml)
#   --pure-wheel PATH        Path to odin_prompt_toolkit-*.whl  (default: auto-detect in packages/python/dist/)
#   --native-wheel PATH      Path to odin_prompt_toolkit_native-*.whl  (optional)
#   --model-source PATH      Local directory containing model v1 files  (default: download from HuggingFace)
#   --output-dir PATH        Where to write the bundle directory  (default: current directory)
#   --zip                    Also create a .zip archive of the bundle
#   --help                   Show this help message
#
# Output:
#   odin-prompt-toolkit-offline-v{VERSION}/   ← copy this whole directory to USB
#   odin-prompt-toolkit-offline-v{VERSION}.zip  ← only if --zip is passed
#
# Bundle layout (what the recipient receives):
#   install.sh              ← recipient runs this
#   verify.py               ← post-install smoke test
#   README.txt              ← quick-start instructions
#   sdk/
#     odin_prompt_toolkit-{VERSION}-py3-none-any.whl
#     odin_prompt_toolkit_native-{VERSION}-{platform}.whl  (if provided)
#   model/
#     v1/
#       onnx/model_O4.onnx       (~235 MB)
#       tokenizer.json
#       config.json
#

set -euo pipefail

# ── colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${RESET}    $*"; }
log_success() { echo -e "${GREEN}[OK]${RESET}      $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${RESET}    $*"; }
log_error()   { echo -e "${RED}[ERROR]${RESET}   $*"; }
die()         { log_error "$*"; exit 1; }

# ── defaults ──────────────────────────────────────────────────────────────────
VERSION=""
PURE_WHEEL=""
NATIVE_WHEEL=""
MODEL_SOURCE=""
OUTPUT_DIR="."
CREATE_ZIP=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HF_REPO="0dinai/jailbreak-embeddings-small"

# ── argument parsing ──────────────────────────────────────────────────────────
show_help() {
    sed -n '2,38p' "$0" | sed 's/^# //' | sed 's/^#//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --version)      VERSION="$2";       shift 2 ;;
        --pure-wheel)   PURE_WHEEL="$2";    shift 2 ;;
        --native-wheel) NATIVE_WHEEL="$2";  shift 2 ;;
        --model-source) MODEL_SOURCE="$2";  shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";    shift 2 ;;
        --zip)          CREATE_ZIP=true;    shift   ;;
        --help)         show_help ;;
        *) die "Unknown option: $1 (use --help for usage)" ;;
    esac
done

# ── auto-detect version ───────────────────────────────────────────────────────
if [[ -z "$VERSION" ]]; then
    PYPROJECT="${REPO_ROOT}/packages/python/pyproject.toml"
    [[ -f "$PYPROJECT" ]] || die "Cannot find pyproject.toml at $PYPROJECT"
    VERSION=$(grep '^version = ' "$PYPROJECT" | sed 's/version = "\(.*\)"/\1/')
    [[ -n "$VERSION" ]] || die "Could not extract version from $PYPROJECT"
    log_info "Auto-detected version: $VERSION"
fi

[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || \
    die "Invalid version format: $VERSION (expected X.Y.Z)"

# ── bundle paths ──────────────────────────────────────────────────────────────
BUNDLE_NAME="odin-prompt-toolkit-offline-v${VERSION}"
BUNDLE_DIR="${OUTPUT_DIR}/${BUNDLE_NAME}"
ZIP_PATH="${OUTPUT_DIR}/${BUNDLE_NAME}.zip"

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${CYAN}  0DIN Prompt Toolkit — Offline Packager v${VERSION}${RESET}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

# ── step 1: locate pure-python wheel ─────────────────────────────────────────
log_info "Step 1/5: Locating SDK wheel..."

if [[ -z "$PURE_WHEEL" ]]; then
    PYTHON_DIST="${REPO_ROOT}/packages/python/dist"
    PURE_WHEEL=$(find "$PYTHON_DIST" -name "odin_prompt_toolkit-${VERSION}-py3-none-any.whl" 2>/dev/null | head -1)
    if [[ -z "$PURE_WHEEL" ]]; then
        die "Pure Python wheel not found in ${PYTHON_DIST}.\n       Run 'make package-python' first, or pass --pure-wheel PATH."
    fi
fi

[[ -f "$PURE_WHEEL" ]] || die "Pure wheel not found: $PURE_WHEEL"
log_success "Pure Python wheel: $(basename "$PURE_WHEEL")"

# Optional native wheel
if [[ -n "$NATIVE_WHEEL" ]]; then
    [[ -f "$NATIVE_WHEEL" ]] || die "Native wheel not found: $NATIVE_WHEEL"
    log_success "Native wheel: $(basename "$NATIVE_WHEEL")"
else
    log_warn "No native wheel provided — recipients will use pure Python (slower but functional)"
fi

echo ""

# ── step 2: locate / download model ──────────────────────────────────────────
log_info "Step 2/5: Obtaining ONNX model..."

REQUIRED_MODEL_FILES=("onnx/model_O4.onnx" "tokenizer.json" "config.json")

if [[ -n "$MODEL_SOURCE" ]]; then
    [[ -d "$MODEL_SOURCE" ]] || die "Model source directory not found: $MODEL_SOURCE"

    for f in "${REQUIRED_MODEL_FILES[@]}"; do
        [[ -f "${MODEL_SOURCE}/${f}" ]] || \
            die "Required model file missing from --model-source: ${f}"
    done
    log_success "Using local model source: $MODEL_SOURCE"

else
    # Download from HuggingFace into a temp dir so we can validate before copying
    if ! command -v huggingface-cli &>/dev/null; then
        die "huggingface-cli not found. Install with: pip install huggingface_hub[cli]\n       Or provide a local model with --model-source PATH."
    fi

    TEMP_MODEL_DIR=$(mktemp -d)
    trap 'rm -rf "$TEMP_MODEL_DIR"' EXIT

    log_info "Downloading model from HuggingFace: ${HF_REPO} (~235 MB)..."
    huggingface-cli download "$HF_REPO" \
        --repo-type model \
        --local-dir "$TEMP_MODEL_DIR" \
        --quiet \
        v1/onnx/model_O4.onnx \
        v1/tokenizer.json \
        v1/config.json || die "Model download failed. Check: huggingface-cli whoami"

    # HuggingFace CLI may place files under v1/ or at root depending on version
    if [[ -d "${TEMP_MODEL_DIR}/v1" ]]; then
        MODEL_SOURCE="${TEMP_MODEL_DIR}/v1"
    else
        MODEL_SOURCE="$TEMP_MODEL_DIR"
    fi

    for f in "${REQUIRED_MODEL_FILES[@]}"; do
        [[ -f "${MODEL_SOURCE}/${f}" ]] || \
            die "Expected model file missing after download: ${f}"
    done

    MODEL_SIZE=$(du -sh "${MODEL_SOURCE}/onnx/model_O4.onnx" | cut -f1)
    log_success "Model downloaded (ONNX: ${MODEL_SIZE})"
fi

echo ""

# ── step 3: assemble bundle directory ────────────────────────────────────────
log_info "Step 3/5: Assembling bundle..."

# Clean any previous attempt
if [[ -d "$BUNDLE_DIR" ]]; then
    log_warn "Removing existing bundle directory: $BUNDLE_DIR"
    rm -rf "$BUNDLE_DIR"
fi

mkdir -p "${BUNDLE_DIR}/sdk"
mkdir -p "${BUNDLE_DIR}/model/v1/onnx"

# SDK wheels
cp "$PURE_WHEEL" "${BUNDLE_DIR}/sdk/"
if [[ -n "$NATIVE_WHEEL" ]]; then
    cp "$NATIVE_WHEEL" "${BUNDLE_DIR}/sdk/"
fi

# Model files
cp "${MODEL_SOURCE}/onnx/model_O4.onnx" "${BUNDLE_DIR}/model/v1/onnx/"
cp "${MODEL_SOURCE}/tokenizer.json"      "${BUNDLE_DIR}/model/v1/"
cp "${MODEL_SOURCE}/config.json"         "${BUNDLE_DIR}/model/v1/"

# Installer and verifier
cp "${SCRIPT_DIR}/install-offline.sh" "${BUNDLE_DIR}/install.sh"
cp "${SCRIPT_DIR}/verify.py"             "${BUNDLE_DIR}/verify.py"
chmod +x "${BUNDLE_DIR}/install.sh"
chmod +x "${BUNDLE_DIR}/verify.py"

# Quick-start README for recipient
cat > "${BUNDLE_DIR}/README.txt" << EOF
0DIN Prompt Toolkit — Offline Bundle v${VERSION}
========================================================

QUICK START
-----------
1. Copy this entire directory to the target machine.
2. Open a terminal and navigate to this directory.
3. (Optional) Activate an existing Python virtual environment first.
4. Run the installer:

       bash install.sh

5. When installation finishes, activate the environment (if a new venv
   was created) and test:

       source venv/bin/activate     # Linux / macOS
       venv\\Scripts\\activate.bat    # Windows (Git Bash / WSL)

       python verify.py

REQUIREMENTS
------------
- Python 3.10 or newer  (check: python3 --version)
- pip  (check: python3 -m pip --version)
- Internet access for pip dependencies (onnxruntime, sentence-transformers, numpy)
  If fully air-gapped, pre-download those wheels and place them in deps/
  before running install.sh — the installer will use them automatically.

CONTENTS
--------
  install.sh          Installation script
  verify.py           Post-install smoke test
  sdk/                Python wheel files
  model/v1/           ONNX model (~235 MB) and tokenizer

SUPPORT
-------
Contact your 0DIN representative with questions.
EOF

log_success "Bundle assembled: $BUNDLE_DIR"
echo ""

# ── step 4: checksum manifest ─────────────────────────────────────────────────
log_info "Step 4/5: Generating checksum manifest..."

MANIFEST="${BUNDLE_DIR}/MANIFEST.sha256"
(
    cd "$BUNDLE_DIR"
    find . -type f ! -name "MANIFEST.sha256" | sort | while read -r f; do
        if command -v sha256sum &>/dev/null; then
            sha256sum "$f"
        else
            shasum -a 256 "$f"
        fi
    done
) > "$MANIFEST"

log_success "Manifest written: MANIFEST.sha256 ($(wc -l < "$MANIFEST") files)"
echo ""

# ── step 5: optional zip ──────────────────────────────────────────────────────
log_info "Step 5/5: Finalizing..."

if [[ "$CREATE_ZIP" == "true" ]]; then
    if [[ -f "$ZIP_PATH" ]]; then
        rm -f "$ZIP_PATH"
    fi
    (cd "$OUTPUT_DIR" && zip -qr "$(basename "$ZIP_PATH")" "$(basename "$BUNDLE_DIR")")
    ZIP_SIZE=$(du -h "$ZIP_PATH" | cut -f1)
    log_success "Zip archive: $ZIP_PATH ($ZIP_SIZE)"
fi

BUNDLE_SIZE=$(du -sh "$BUNDLE_DIR" | cut -f1)

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
log_success "Offline installer bundle ready!"
echo ""
echo "  📦 Bundle:  $BUNDLE_DIR"
echo "  📏 Size:    $BUNDLE_SIZE"
if [[ "$CREATE_ZIP" == "true" ]]; then
    echo "  🗜️  Zip:     $ZIP_PATH ($ZIP_SIZE)"
fi
echo ""
echo "Next steps:"
echo "  1. Copy $BUNDLE_DIR to a USB drive (or secure file transfer)"
echo "  2. On the target machine: cd <bundle-dir> && bash install.sh"
echo "  3. Recipient runs: python verify.py"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
