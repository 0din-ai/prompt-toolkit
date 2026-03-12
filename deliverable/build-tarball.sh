#!/usr/bin/env bash
#
# Build design partner deliverable tarball for odin-prompt-toolkit
#
# This script assembles a complete distribution package including:
# - Python SDK wheels (pure + native for multiple platforms)
# - ONNX model files (downloaded from HuggingFace or copied from local cache)
# - Example scripts
# - Signature pack (user-provided)
# - Documentation and installation scripts
#
# Usage:
#   ./build-tarball.sh [OPTIONS]
#
# Options:
#   --version VERSION    SDK version (default: read from pyproject.toml)
#   --model-source PATH  Local model directory (default: download from HuggingFace)
#   --sig-pack PATH      Path to signature pack JSON (default: none, creates placeholder)
#   --output-dir PATH    Output directory (default: current directory)
#   --keep-staging       Don't delete staging directory after build
#   --help               Show this help message
#
# Requirements:
#   - Python SDK wheels in ../packages/python/dist/
#   - Native wheels in ../packages/python-native/dist/ (optional)
#   - huggingface-cli installed (for model download)
#   - HuggingFace token set (for private repos): huggingface-cli login
#
# Output:
#   odin-prompt-toolkit-deliverable-v{VERSION}.tar.gz
#   odin-prompt-toolkit-deliverable-v{VERSION}.tar.gz.sha256
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

# Default values
VERSION=""
MODEL_SOURCE=""
SIG_PACK=""
OUTPUT_DIR="."
KEEP_STAGING=false

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_DIST="$REPO_ROOT/packages/python/dist"
NATIVE_DIST="$REPO_ROOT/packages/python-native/dist"
EXAMPLES_DIR="$REPO_ROOT/packages/python/examples"
INSTALL_MD="$REPO_ROOT/packages/python/INSTALL.md"

# HuggingFace model repo
HF_REPO="0dinai/jailbreak-embeddings-small"
HF_MODEL_PATH="v1"

# Functions
log_info() {
    echo -e "${CYAN}[INFO]${RESET} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${RESET} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${RESET} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${RESET} $*"
}

die() {
    log_error "$*"
    exit 1
}

show_help() {
    sed -n '2,27p' "$0" | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --model-source)
            MODEL_SOURCE="$2"
            shift 2
            ;;
        --sig-pack)
            SIG_PACK="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --keep-staging)
            KEEP_STAGING=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            die "Unknown option: $1 (use --help for usage)"
            ;;
    esac
done

# Auto-detect version from pyproject.toml if not specified
if [[ -z "$VERSION" ]]; then
    PYPROJECT="$REPO_ROOT/packages/python/pyproject.toml"
    if [[ ! -f "$PYPROJECT" ]]; then
        die "Could not find $PYPROJECT"
    fi
    VERSION=$(grep '^version = ' "$PYPROJECT" | sed 's/version = "\(.*\)"/\1/')
    if [[ -z "$VERSION" ]]; then
        die "Could not extract version from $PYPROJECT"
    fi
    log_info "Auto-detected version: $VERSION"
fi

# Validate version format
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    die "Invalid version format: $VERSION (expected: X.Y.Z)"
fi

# Staging directory
STAGING_DIR="$SCRIPT_DIR/staging"
DELIVERABLE_NAME="odin-prompt-toolkit-deliverable-v$VERSION"
DELIVERABLE_DIR="$STAGING_DIR/$DELIVERABLE_NAME"
TARBALL_NAME="$DELIVERABLE_NAME.tar.gz"
TARBALL_PATH="$OUTPUT_DIR/$TARBALL_NAME"

log_info "Building deliverable for odin-prompt-toolkit v$VERSION"
log_info "Output: $TARBALL_PATH"
echo ""

# Step 1: Validate prerequisites
log_info "Step 1/8: Validating prerequisites..."

# Check Python wheels exist
if [[ ! -d "$PYTHON_DIST" ]]; then
    die "Python dist directory not found: $PYTHON_DIST (run 'make package-python' first)"
fi

PURE_WHEEL=$(find "$PYTHON_DIST" -name "odin_prompt_toolkit-${VERSION}-py3-none-any.whl" | head -n 1)
if [[ -z "$PURE_WHEEL" ]]; then
    die "Python wheel not found: odin_prompt_toolkit-${VERSION}-py3-none-any.whl (run 'make package-python')"
fi
log_success "Found Python wheel: $(basename "$PURE_WHEEL")"

# Check native wheels (optional)
NATIVE_WHEELS=()
if [[ -d "$NATIVE_DIST" ]]; then
    while IFS= read -r wheel; do
        NATIVE_WHEELS+=("$wheel")
    done < <(find "$NATIVE_DIST" -name "odin_prompt_toolkit_native-${VERSION}-*.whl")
    
    if [[ ${#NATIVE_WHEELS[@]} -gt 0 ]]; then
        log_success "Found ${#NATIVE_WHEELS[@]} native wheel(s)"
    else
        log_warn "No native wheels found (pure Python only)"
    fi
else
    log_warn "Native dist directory not found: $NATIVE_DIST"
fi

# Check examples directory
if [[ ! -d "$EXAMPLES_DIR" ]]; then
    die "Examples directory not found: $EXAMPLES_DIR"
fi
log_success "Found examples directory"

# Check installation scripts
if [[ ! -f "$SCRIPT_DIR/install.sh" ]]; then
    die "install.sh not found: $SCRIPT_DIR/install.sh"
fi
if [[ ! -f "$SCRIPT_DIR/verify.py" ]]; then
    die "verify.py not found: $SCRIPT_DIR/verify.py"
fi
if [[ ! -f "$SCRIPT_DIR/README.md" ]]; then
    die "README.md not found: $SCRIPT_DIR/README.md"
fi
log_success "Found installation scripts and README"

echo ""

# Step 2: Clean and create staging directory
log_info "Step 2/8: Creating staging directory..."
rm -rf "$STAGING_DIR"
mkdir -p "$DELIVERABLE_DIR"/{sdk,model/v1,signatures,examples}
log_success "Created staging directory: $DELIVERABLE_DIR"
echo ""

# Step 3: Copy SDK wheels
log_info "Step 3/8: Copying SDK wheels..."
cp "$PURE_WHEEL" "$DELIVERABLE_DIR/sdk/"
log_success "Copied: $(basename "$PURE_WHEEL")"

for wheel in "${NATIVE_WHEELS[@]}"; do
    cp "$wheel" "$DELIVERABLE_DIR/sdk/"
    log_success "Copied: $(basename "$wheel")"
done

# Create requirements.txt
cat > "$DELIVERABLE_DIR/sdk/requirements.txt" << EOF
# Core dependencies for odin-prompt-toolkit v${VERSION}
numpy>=1.24.0,<2.0.0
EOF
log_success "Created requirements.txt"
echo ""

# Step 4: Copy/download model files
log_info "Step 4/8: Obtaining model files..."

if [[ -n "$MODEL_SOURCE" ]]; then
    # Use local model directory
    log_info "Using local model source: $MODEL_SOURCE"
    
    if [[ ! -d "$MODEL_SOURCE" ]]; then
        die "Model source directory not found: $MODEL_SOURCE"
    fi
    
    # Copy all files from model source
    cp -r "$MODEL_SOURCE"/* "$DELIVERABLE_DIR/model/v1/"
    log_success "Copied model files from local source"
    
else
    # Download from HuggingFace
    log_info "Downloading model from HuggingFace: $HF_REPO"
    
    # Check if huggingface-cli is available
    if ! command -v huggingface-cli &> /dev/null; then
        die "huggingface-cli not found (install: pip install huggingface_hub[cli])"
    fi
    
    # Download model files
    log_info "Downloading ONNX model (~235MB, this may take a few minutes)..."
    
    # Download files to temporary directory then move
    TEMP_MODEL_DIR=$(mktemp -d)
    trap 'rm -rf "$TEMP_MODEL_DIR"' EXIT
    
    if huggingface-cli download "$HF_REPO" \
        --repo-type model \
        --local-dir "$TEMP_MODEL_DIR" \
        --quiet \
        ${HF_MODEL_PATH}/onnx/model_O4.onnx \
        ${HF_MODEL_PATH}/tokenizer.json \
        ${HF_MODEL_PATH}/config.json; then
        
        # Move files to deliverable
        if [[ -d "$TEMP_MODEL_DIR/$HF_MODEL_PATH" ]]; then
            cp -r "$TEMP_MODEL_DIR/$HF_MODEL_PATH"/* "$DELIVERABLE_DIR/model/v1/"
        else
            # Files might be at root
            cp -r "$TEMP_MODEL_DIR"/* "$DELIVERABLE_DIR/model/v1/"
        fi
        
        log_success "Downloaded model files from HuggingFace"
    else
        die "Failed to download model from HuggingFace (check: huggingface-cli whoami)"
    fi
fi

# Verify critical model files exist
REQUIRED_MODEL_FILES=(
    "model/v1/onnx/model_O4.onnx"
    "model/v1/tokenizer.json"
    "model/v1/config.json"
)

for file in "${REQUIRED_MODEL_FILES[@]}"; do
    if [[ ! -f "$DELIVERABLE_DIR/$file" ]]; then
        die "Required model file missing: $file"
    fi
done

MODEL_SIZE=$(du -sh "$DELIVERABLE_DIR/model/v1/onnx/model_O4.onnx" | cut -f1)
log_success "Model validated (ONNX model size: $MODEL_SIZE)"
echo ""

# Step 5: Copy examples
log_info "Step 5/8: Copying examples..."
cp "$EXAMPLES_DIR"/*.py "$DELIVERABLE_DIR/examples/"
EXAMPLE_COUNT=$(find "$DELIVERABLE_DIR/examples" -name "*.py" | wc -l | tr -d ' ')
log_success "Copied $EXAMPLE_COUNT example scripts"
echo ""

# Step 6: Copy/create signature pack
log_info "Step 6/8: Handling signature pack..."

if [[ -n "$SIG_PACK" ]]; then
    if [[ ! -f "$SIG_PACK" ]]; then
        die "Signature pack not found: $SIG_PACK"
    fi
    cp "$SIG_PACK" "$DELIVERABLE_DIR/signatures/threat-feed-v1.json"
    log_success "Copied signature pack: $(basename "$SIG_PACK")"
else
    # Create placeholder signature pack
    cat > "$DELIVERABLE_DIR/signatures/threat-feed-v1.json" << 'EOF'
{
  "version": "v1",
  "created_at": "2024-03-09T00:00:00Z",
  "description": "Placeholder signature pack - replace with your own",
  "signatures": [
    {
      "id": "example-1",
      "signature": "0din-v1:0000000000000000000000000000000000000000000000000000000000000000",
      "metadata": {
        "category": "example",
        "note": "This is a placeholder - replace with real signatures"
      }
    }
  ]
}
EOF
    log_warn "Created placeholder signature pack (replace with real data)"
fi
echo ""

# Step 7: Copy documentation and scripts
log_info "Step 7/8: Copying documentation and scripts..."
cp "$SCRIPT_DIR/install.sh" "$DELIVERABLE_DIR/"
cp "$SCRIPT_DIR/verify.py" "$DELIVERABLE_DIR/"
cp "$SCRIPT_DIR/README.md" "$DELIVERABLE_DIR/"
cp "$INSTALL_MD" "$DELIVERABLE_DIR/INSTALL.md"

# Ensure scripts are executable
chmod +x "$DELIVERABLE_DIR/install.sh"
chmod +x "$DELIVERABLE_DIR/verify.py"

log_success "Copied documentation and scripts"
echo ""

# Step 8: Create tarball and checksum
log_info "Step 8/8: Creating tarball..."

# Create tarball from staging directory
(cd "$STAGING_DIR" && tar -czf "$TARBALL_NAME" "$DELIVERABLE_NAME")
mv "$STAGING_DIR/$TARBALL_NAME" "$TARBALL_PATH"

# Generate SHA256 checksum
CHECKSUM_FILE="$TARBALL_PATH.sha256"
if command -v sha256sum &> /dev/null; then
    (cd "$OUTPUT_DIR" && sha256sum "$(basename "$TARBALL_PATH")") > "$CHECKSUM_FILE"
elif command -v shasum &> /dev/null; then
    (cd "$OUTPUT_DIR" && shasum -a 256 "$(basename "$TARBALL_PATH")") > "$CHECKSUM_FILE"
else
    log_warn "sha256sum/shasum not found, skipping checksum generation"
    CHECKSUM_FILE=""
fi

TARBALL_SIZE=$(du -h "$TARBALL_PATH" | cut -f1)
log_success "Created tarball: $TARBALL_NAME ($TARBALL_SIZE)"

if [[ -n "$CHECKSUM_FILE" ]]; then
    CHECKSUM=$(cat "$CHECKSUM_FILE" | cut -d' ' -f1)
    log_success "SHA256: $CHECKSUM"
fi

# Cleanup staging directory
if [[ "$KEEP_STAGING" == "false" ]]; then
    rm -rf "$STAGING_DIR"
    log_info "Removed staging directory"
else
    log_info "Kept staging directory: $STAGING_DIR"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_success "Deliverable built successfully!"
echo ""
echo "  📦 Tarball: $TARBALL_PATH"
echo "  📏 Size:    $TARBALL_SIZE"
if [[ -n "$CHECKSUM_FILE" ]]; then
    echo "  🔐 SHA256:  $CHECKSUM"
fi
echo ""
echo "Next steps:"
echo "  1. Test the tarball: tar -tzf $TARBALL_PATH | head -20"
echo "  2. Extract and test install: tar -xzf $TARBALL_PATH && cd $DELIVERABLE_NAME && ./install.sh --online"
echo "  3. Distribute to design partner"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
