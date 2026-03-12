#!/bin/bash

# 0DIN Prompt Toolkit Design Partner Installation Script
# Installs the Python SDK, native acceleration, ONNX model, and signature pack

set -e

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
RESET='\033[0m'

# Configuration
REQUIRED_PYTHON_VERSION="3.10"
VENV_DIR="./venv"
CACHE_DIR="$HOME/.cache/odin-prompt-toolkit/models/v1"
MODEL_ONLINE_URL="https://huggingface.co/0dinai/jailbreak-embeddings-small/resolve/main"

# Modes
MODE="offline"  # default: expect bundled model
if [[ "$1" == "--online" ]]; then
    MODE="online"
fi

# Functions
error() {
    echo -e "${RED}ERROR: $1${RESET}" >&2
    exit 1
}

info() {
    echo -e "${CYAN}$1${RESET}"
}

success() {
    echo -e "${GREEN}✅ $1${RESET}"
}

warn() {
    echo -e "${YELLOW}⚠️  $1${RESET}"
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        error "python3 not found. Please install Python >= ${REQUIRED_PYTHON_VERSION}"
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    REQUIRED_MAJOR=$(echo "$REQUIRED_PYTHON_VERSION" | cut -d. -f1)
    REQUIRED_MINOR=$(echo "$REQUIRED_PYTHON_VERSION" | cut -d. -f2)
    
    if [[ "$PYTHON_MAJOR" -lt "$REQUIRED_MAJOR" ]] || \
       [[ "$PYTHON_MAJOR" -eq "$REQUIRED_MAJOR" && "$PYTHON_MINOR" -lt "$REQUIRED_MINOR" ]]; then
        error "Python ${PYTHON_VERSION} found, but >= ${REQUIRED_PYTHON_VERSION} required"
    fi
    
    success "Python ${PYTHON_VERSION} found"
}

detect_platform() {
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)
    
    case "$OS" in
        linux)
            PLATFORM="linux"
            ;;
        darwin)
            PLATFORM="macos"
            ;;
        mingw*|msys*|cygwin*)
            PLATFORM="windows"
            ;;
        *)
            warn "Unknown OS: $OS. Proceeding with pure Python installation."
            PLATFORM="unknown"
            ;;
    esac
    
    case "$ARCH" in
        x86_64|amd64)
            ARCH_NORMALIZED="x86_64"
            ;;
        aarch64|arm64)
            ARCH_NORMALIZED="aarch64"
            ;;
        *)
            warn "Unknown architecture: $ARCH. Proceeding with pure Python installation."
            ARCH_NORMALIZED="unknown"
            ;;
    esac
    
    info "Detected: $PLATFORM $ARCH_NORMALIZED"
}

create_venv() {
    if [[ -d "$VENV_DIR" ]]; then
        warn "Virtual environment already exists at $VENV_DIR"
        read -p "Remove and recreate? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            error "Installation aborted. Remove $VENV_DIR manually and retry."
        fi
    fi
    
    info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    success "Virtual environment created"
}

install_sdk() {
    info "Installing 0DIN Prompt Toolkit..."
    
    # Find the pure Python wheel
    PURE_WHEEL=$(find sdk/ -name "odin_prompt_toolkit-*-py3-none-any.whl" | head -1)
    if [[ -z "$PURE_WHEEL" ]]; then
        error "Pure Python wheel not found in sdk/"
    fi
    
    "$VENV_DIR/bin/pip" install --no-cache-dir "$PURE_WHEEL" > /dev/null 2>&1
    success "Pure Python SDK installed"
    
    # Try to install native wheel for this platform
    PYTHON_TAG="cp$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
    
    if [[ "$PLATFORM" == "linux" && "$ARCH_NORMALIZED" == "x86_64" ]]; then
        NATIVE_PATTERN="odin_prompt_toolkit_native-*-${PYTHON_TAG}-*-manylinux*_x86_64.whl"
    elif [[ "$PLATFORM" == "linux" && "$ARCH_NORMALIZED" == "aarch64" ]]; then
        NATIVE_PATTERN="odin_prompt_toolkit_native-*-${PYTHON_TAG}-*-manylinux*_aarch64.whl"
    elif [[ "$PLATFORM" == "macos" && "$ARCH_NORMALIZED" == "x86_64" ]]; then
        NATIVE_PATTERN="odin_prompt_toolkit_native-*-${PYTHON_TAG}-*-macosx*_x86_64.whl"
    elif [[ "$PLATFORM" == "macos" && "$ARCH_NORMALIZED" == "aarch64" ]]; then
        NATIVE_PATTERN="odin_prompt_toolkit_native-*-${PYTHON_TAG}-*-macosx*_arm64.whl"
    elif [[ "$PLATFORM" == "windows" && "$ARCH_NORMALIZED" == "x86_64" ]]; then
        NATIVE_PATTERN="odin_prompt_toolkit_native-*-${PYTHON_TAG}-*-win_amd64.whl"
    else
        info "No native wheel available for $PLATFORM $ARCH_NORMALIZED (using pure Python)"
        return
    fi
    
    NATIVE_WHEEL=$(find sdk/ -name "$NATIVE_PATTERN" 2>/dev/null | head -1)
    if [[ -n "$NATIVE_WHEEL" ]]; then
        "$VENV_DIR/bin/pip" install --no-cache-dir "$NATIVE_WHEEL" > /dev/null 2>&1
        success "Native acceleration installed (653× faster)"
    else
        info "Native wheel not found for this platform (using pure Python)"
    fi
}

install_model() {
    info "Setting up ONNX model..."
    
    mkdir -p "$CACHE_DIR/onnx"
    
    if [[ "$MODE" == "online" ]]; then
        info "Downloading model from HuggingFace (this may take a few minutes)..."
        
        if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
            error "curl or wget required for online mode. Use offline mode or install curl/wget."
        fi
        
        if command -v curl &> /dev/null; then
            curl -L -o "$CACHE_DIR/onnx/model_O4.onnx" "${MODEL_ONLINE_URL}/onnx/model_O4.onnx" || error "Failed to download model"
            curl -L -o "$CACHE_DIR/tokenizer.json" "${MODEL_ONLINE_URL}/tokenizer.json" || error "Failed to download tokenizer"
            curl -L -o "$CACHE_DIR/config.json" "${MODEL_ONLINE_URL}/config.json" || error "Failed to download config"
        else
            wget -O "$CACHE_DIR/onnx/model_O4.onnx" "${MODEL_ONLINE_URL}/onnx/model_O4.onnx" || error "Failed to download model"
            wget -O "$CACHE_DIR/tokenizer.json" "${MODEL_ONLINE_URL}/tokenizer.json" || error "Failed to download tokenizer"
            wget -O "$CACHE_DIR/config.json" "${MODEL_ONLINE_URL}/config.json" || error "Failed to download config"
        fi
        
        success "Model downloaded"
    else
        # Offline mode: copy from bundled files
        if [[ ! -d "model/v1" ]]; then
            error "Model directory not found. Run with --online to download, or ensure model/ directory exists."
        fi
        
        if [[ ! -f "model/v1/onnx/model_O4.onnx" ]]; then
            error "Model file not found at model/v1/onnx/model_O4.onnx"
        fi
        
        cp -r model/v1/* "$CACHE_DIR/"
        success "Model installed from bundle"
    fi
    
    # Verify model files exist
    if [[ ! -f "$CACHE_DIR/onnx/model_O4.onnx" ]]; then
        error "Model installation failed: model_O4.onnx not found"
    fi
    if [[ ! -f "$CACHE_DIR/tokenizer.json" ]]; then
        error "Model installation failed: tokenizer.json not found"
    fi
    if [[ ! -f "$CACHE_DIR/config.json" ]]; then
        error "Model installation failed: config.json not found"
    fi
}

install_signatures() {
    if [[ -f "signatures/threat-feed-v1.json" ]]; then
        info "Copying signature pack..."
        mkdir -p "$HOME/.odin-prompt-toolkit/signatures"
        cp signatures/threat-feed-v1.json "$HOME/.odin-prompt-toolkit/signatures/"
        success "Signature pack installed"
    else
        warn "No signature pack found (signatures/threat-feed-v1.json)"
    fi
}

run_verification() {
    if [[ -f "verify.py" ]]; then
        info "Running post-install verification..."
        if "$VENV_DIR/bin/python" verify.py; then
            success "Verification passed"
        else
            error "Verification failed. Installation may be incomplete."
        fi
    else
        warn "Verification script not found (verify.py)"
    fi
}

print_completion() {
    echo ""
    echo -e "${GREEN}╔═════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${GREEN}║                 Installation Complete!                     ║${RESET}"
    echo -e "${GREEN}╚═════════════════════════════════════════════════════════════╝${RESET}"
    echo ""
    echo -e "${CYAN}Activate the virtual environment:${RESET}"
    echo "  source $VENV_DIR/bin/activate"
    echo ""
    echo -e "${CYAN}Generate your first signature:${RESET}"
    echo "  python examples/basic_signature.py"
    echo ""
    echo -e "${CYAN}Documentation:${RESET}"
    echo "  See README.md and INSTALL.md for more information"
    echo ""
}

# Main installation flow
main() {
    echo -e "${CYAN}╔═════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${CYAN}║        0DIN Prompt Toolkit Design Partner Installation           ║${RESET}"
    echo -e "${CYAN}╚═════════════════════════════════════════════════════════════╝${RESET}"
    echo ""
    
    info "Installation mode: $MODE"
    if [[ "$MODE" == "online" ]]; then
        warn "Online mode: Will download ~235MB model from HuggingFace"
    fi
    echo ""
    
    check_python
    detect_platform
    create_venv
    install_sdk
    install_model
    install_signatures
    run_verification
    print_completion
}

main
