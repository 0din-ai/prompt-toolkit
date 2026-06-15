#!/usr/bin/env bash
# Regenerate SusFactor golden vectors from the validated Rust implementation.
#
# Usage:
#   SUSFACTOR_MODEL_DIR=/path/to/cache/susfactor-v1 \
#     scripts/regenerate_susfactor_goldens.sh
#
# What it does:
#   1. Runs the Rust extraction example against the local model.
#   2. Writes rust_score values back into spec/test-vectors/susfactor_vectors.json.
#   3. Shows the git diff so you can review before committing.
#
# When to run this:
#   - After the model is retrained or re-exported.
#   - After the model is re-validated against Heimdall with a new ORT version.
#   - NEVER as an automatic step — always a human-reviewed, deliberate commit.
#
# Requirements:
#   - Rust toolchain with the `susfactor` feature available.
#   - Local copy of 0dinai/susfactor-e5-large-onnx at SUSFACTOR_MODEL_DIR.
#     The directory must contain:
#       onnx/model.onnx
#       onnx/model.onnx_data
#       tokenizer.json

set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
RESET='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo -e "${CYAN}═══════════════════════════════════════════════════════════${RESET}"
echo -e "${CYAN}  SusFactor Golden Vector Regeneration${RESET}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${RESET}"
echo ""

# ── Validate model dir ───────────────────────────────────────────────────────

if [[ -z "${SUSFACTOR_MODEL_DIR:-}" ]]; then
    echo -e "${RED}error: SUSFACTOR_MODEL_DIR is not set.${RESET}"
    echo ""
    echo "Point it at a local directory containing:"
    echo "  onnx/model.onnx"
    echo "  onnx/model.onnx_data"
    echo "  tokenizer.json"
    echo ""
    echo "Download from: https://huggingface.co/0dinai/susfactor-e5-large-onnx"
    exit 1
fi

for required in "onnx/model.onnx" "onnx/model.onnx_data" "tokenizer.json"; do
    if [[ ! -f "${SUSFACTOR_MODEL_DIR}/${required}" ]]; then
        echo -e "${RED}error: missing required file: ${SUSFACTOR_MODEL_DIR}/${required}${RESET}"
        exit 1
    fi
done

echo -e "Model dir: ${YELLOW}${SUSFACTOR_MODEL_DIR}${RESET}"
echo ""

# ── Run the Rust extraction example ─────────────────────────────────────────

echo -e "${CYAN}Building extraction example (--release for ORT speed)...${RESET}"
cargo build \
    --example extract_susfactor_goldens \
    --features susfactor \
    --release \
    --manifest-path "${ROOT_DIR}/packages/rust/Cargo.toml"

echo ""
echo -e "${CYAN}Running extraction example...${RESET}"
SUSFACTOR_MODEL_DIR="${SUSFACTOR_MODEL_DIR}" \
    cargo run \
    --example extract_susfactor_goldens \
    --features susfactor \
    --release \
    --manifest-path "${ROOT_DIR}/packages/rust/Cargo.toml"

echo ""

# ── Show diff for review ─────────────────────────────────────────────────────

FIXTURE="${ROOT_DIR}/spec/test-vectors/susfactor_vectors.json"
echo -e "${CYAN}Diff of ${FIXTURE}:${RESET}"
echo ""
git -C "${ROOT_DIR}" diff -- "${FIXTURE}" || true

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${RESET}"
echo -e "${YELLOW}  REVIEW THE DIFF ABOVE BEFORE COMMITTING.${RESET}"
echo ""
echo -e "  Commit message format:"
echo -e "    chore(susfactor): regenerate golden vectors [model: <sha/tag>]"
echo ""
echo -e "  Only commit if:"
echo -e "    • Expected labels look correct for all prompts."
echo -e "    • Near-boundary scores have been reviewed."
echo -e "    • The model change has been re-validated in Heimdall."
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "${GREEN}✅ Regeneration complete. Review, then commit.${RESET}"
