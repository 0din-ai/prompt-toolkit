#!/usr/bin/env bash
# Download the pre-built libtokenizers.a static library for daulet/tokenizers v1.27.0.
# Must be run before building or testing the Go SDK.
# Usage: bash scripts/download-libtokenizers.sh

set -euo pipefail

TOKENIZERS_VERSION="1.27.0"
BASE_URL="https://github.com/daulet/tokenizers/releases/download/v${TOKENIZERS_VERSION}"

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

# Normalize arch names to match daulet/tokenizers release naming
case "$ARCH" in
  x86_64)  ARCH="amd64" ;;
  aarch64) ARCH="arm64" ;;
  arm64)   ARCH="arm64" ;;
  *)
    echo "Unsupported architecture: $ARCH" >&2
    exit 1
    ;;
esac

PLATFORM="${OS}-${ARCH}"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/lib/${OS}_${ARCH}"
DEST="$DEST_DIR/libtokenizers.a"

if [ -f "$DEST" ]; then
  echo "libtokenizers.a already present at $DEST — skipping download."
  exit 0
fi

URL="${BASE_URL}/libtokenizers.${PLATFORM}.tar.gz"
echo "Downloading libtokenizers ${TOKENIZERS_VERSION} for ${PLATFORM}..."
echo "  URL:  $URL"
echo "  Dest: $DEST"

mkdir -p "$DEST_DIR"
curl -fsSL "$URL" | tar -xz -C "$DEST_DIR/"

if [ -f "$DEST" ]; then
  echo "Done. libtokenizers.a ready at $DEST"
else
  echo "Error: libtokenizers.a not found after extraction." >&2
  exit 1
fi
