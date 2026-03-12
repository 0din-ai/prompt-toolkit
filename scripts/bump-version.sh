#!/bin/bash

# Version bump script for 0DIN Prompt Toolkit
# Synchronizes version across all three language packages

set -e

# Colors for output
CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
RESET='\033[0m'

# Function to display usage
usage() {
    echo -e "${CYAN}Usage: $0 <new-version>${RESET}"
    echo ""
    echo "Examples:"
    echo "  $0 0.2.0"
    echo "  $0 1.0.0-beta.1"
    echo ""
    echo "This script will:"
    echo "  1. Update version in packages/rust/Cargo.toml"
    echo "  2. Update version in packages/python/pyproject.toml"
    echo "  3. Update version in packages/python/odin_prompt_toolkit/__init__.py"
    echo "  4. Update version in packages/python-native/pyproject.toml"
    echo "  5. Update version in packages/python-native/Cargo.toml"
    echo "  6. Update version in packages/typescript/package.json"
    echo "  7. Show git diff for review"
    echo "  8. Prompt for commit and tag"
    exit 1
}

# Check for version argument
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No version provided${RESET}"
    usage
fi

NEW_VERSION="$1"

# Validate version format (basic check)
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
    echo -e "${RED}Error: Invalid version format${RESET}"
    echo "Version must follow semantic versioning (e.g., 0.2.0, 1.0.0-beta.1)"
    exit 1
fi

# Find project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

echo -e "${CYAN}========================================${RESET}"
echo -e "${CYAN}0DIN Prompt Toolkit - Version Bump${RESET}"
echo -e "${CYAN}========================================${RESET}"
echo ""
echo -e "${YELLOW}New version: ${NEW_VERSION}${RESET}"
echo ""

# Get current versions
echo -e "${CYAN}Current versions:${RESET}"
RUST_VERSION=$(grep '^version = ' packages/rust/Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
PYTHON_VERSION=$(grep '^version = ' packages/python/pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
PYTHON_NATIVE_PY_VERSION=$(grep '^version = ' packages/python-native/pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
PYTHON_NATIVE_RS_VERSION=$(grep '^version = ' packages/python-native/Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
TS_VERSION=$(grep '"version":' packages/typescript/package.json | head -1 | sed 's/.*"version": "\(.*\)".*/\1/')

echo "  Rust:                  $RUST_VERSION"
echo "  Python:                $PYTHON_VERSION"
echo "  Python-native (py):    $PYTHON_NATIVE_PY_VERSION"
echo "  Python-native (rust):  $PYTHON_NATIVE_RS_VERSION"
echo "  TypeScript:            $TS_VERSION"
echo ""

# Check if versions are already synchronized
if [ "$RUST_VERSION" != "$PYTHON_VERSION" ] || \
   [ "$PYTHON_VERSION" != "$PYTHON_NATIVE_PY_VERSION" ] || \
   [ "$PYTHON_NATIVE_PY_VERSION" != "$PYTHON_NATIVE_RS_VERSION" ] || \
   [ "$PYTHON_NATIVE_RS_VERSION" != "$TS_VERSION" ]; then
    echo -e "${YELLOW}⚠️  Warning: Current versions are not synchronized!${RESET}"
    echo ""
fi

# Confirm with user
echo -e "${YELLOW}This will update all six package versions to ${NEW_VERSION}.${RESET}"
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Aborted.${RESET}"
    exit 1
fi

echo ""
echo -e "${CYAN}Updating versions...${RESET}"

# Update Rust (Cargo.toml)
echo "  Updating packages/rust/Cargo.toml..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/^version = \".*\"/version = \"$NEW_VERSION\"/" packages/rust/Cargo.toml
else
    # Linux
    sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" packages/rust/Cargo.toml
fi

# Update Python (pyproject.toml)
echo "  Updating packages/python/pyproject.toml..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/^version = \".*\"/version = \"$NEW_VERSION\"/" packages/python/pyproject.toml
else
    sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" packages/python/pyproject.toml
fi

# Update Python (__init__.py __version__)
echo "  Updating packages/python/odin_prompt_toolkit/__init__.py..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" packages/python/odin_prompt_toolkit/__init__.py
else
    sed -i "s/__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" packages/python/odin_prompt_toolkit/__init__.py
fi

# Update Python-native (pyproject.toml)
echo "  Updating packages/python-native/pyproject.toml..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/^version = \".*\"/version = \"$NEW_VERSION\"/" packages/python-native/pyproject.toml
else
    sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" packages/python-native/pyproject.toml
fi

# Update Python-native (Cargo.toml)
echo "  Updating packages/python-native/Cargo.toml..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/^version = \".*\"/version = \"$NEW_VERSION\"/" packages/python-native/Cargo.toml
else
    sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" packages/python-native/Cargo.toml
fi

# Update TypeScript (package.json)
echo "  Updating packages/typescript/package.json..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/\"version\": \".*\"/\"version\": \"$NEW_VERSION\"/" packages/typescript/package.json
else
    sed -i "s/\"version\": \".*\"/\"version\": \"$NEW_VERSION\"/" packages/typescript/package.json
fi

echo -e "${GREEN}✅ Version updated in all packages${RESET}"
echo ""

# Show diff
echo -e "${CYAN}Changes:${RESET}"
git diff packages/rust/Cargo.toml packages/python/pyproject.toml packages/python/odin_prompt_toolkit/__init__.py packages/python-native/pyproject.toml packages/python-native/Cargo.toml packages/typescript/package.json
echo ""

# Verify versions
echo -e "${CYAN}New versions:${RESET}"
NEW_RUST_VERSION=$(grep '^version = ' packages/rust/Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
NEW_PYTHON_VERSION=$(grep '^version = ' packages/python/pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
NEW_PYTHON_INIT_VERSION=$(grep '__version__ = ' packages/python/odin_prompt_toolkit/__init__.py | sed 's/__version__ = "\(.*\)"/\1/')
NEW_PYTHON_NATIVE_PY_VERSION=$(grep '^version = ' packages/python-native/pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
NEW_PYTHON_NATIVE_RS_VERSION=$(grep '^version = ' packages/python-native/Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
NEW_TS_VERSION=$(grep '"version":' packages/typescript/package.json | head -1 | sed 's/.*"version": "\(.*\)".*/\1/')

echo "  Rust:                        $NEW_RUST_VERSION"
echo "  Python (pyproject):          $NEW_PYTHON_VERSION"
echo "  Python (__init__):           $NEW_PYTHON_INIT_VERSION"
echo "  Python-native (pyproject):   $NEW_PYTHON_NATIVE_PY_VERSION"
echo "  Python-native (Cargo):       $NEW_PYTHON_NATIVE_RS_VERSION"
echo "  TypeScript:                  $NEW_TS_VERSION"
echo ""

# Verify all versions match
if [ "$NEW_RUST_VERSION" = "$NEW_VERSION" ] && \
   [ "$NEW_PYTHON_VERSION" = "$NEW_VERSION" ] && \
   [ "$NEW_PYTHON_INIT_VERSION" = "$NEW_VERSION" ] && \
   [ "$NEW_PYTHON_NATIVE_PY_VERSION" = "$NEW_VERSION" ] && \
   [ "$NEW_PYTHON_NATIVE_RS_VERSION" = "$NEW_VERSION" ] && \
   [ "$NEW_TS_VERSION" = "$NEW_VERSION" ]; then
    echo -e "${GREEN}✅ All versions synchronized successfully!${RESET}"
else
    echo -e "${RED}❌ Error: Version mismatch detected${RESET}"
    exit 1
fi

echo ""

# Prompt to commit and tag
echo -e "${YELLOW}Would you like to commit these changes and create a git tag?${RESET}"
read -p "Commit and tag? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${CYAN}Creating commit...${RESET}"
    git add packages/rust/Cargo.toml packages/python/pyproject.toml packages/python/odin_prompt_toolkit/__init__.py packages/python-native/pyproject.toml packages/python-native/Cargo.toml packages/typescript/package.json
    git commit -m "chore: bump version to $NEW_VERSION"
    
    echo -e "${CYAN}Creating tag v${NEW_VERSION}...${RESET}"
    git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"
    
    echo -e "${GREEN}✅ Committed and tagged!${RESET}"
    echo ""
    echo -e "${CYAN}Next steps:${RESET}"
    echo "  1. Review the commit: git show"
    echo "  2. Push to remote: git push && git push --tags"
    echo ""
else
    echo ""
    echo -e "${YELLOW}Changes staged but not committed.${RESET}"
    echo "To commit manually:"
    echo "  git add packages/rust/Cargo.toml packages/python/pyproject.toml packages/python/odin_prompt_toolkit/__init__.py packages/python-native/pyproject.toml packages/python-native/Cargo.toml packages/typescript/package.json"
    echo "  git commit -m 'chore: bump version to $NEW_VERSION'"
    echo "  git tag -a 'v$NEW_VERSION' -m 'Release v$NEW_VERSION'"
    echo "  git push && git push --tags"
    echo ""
fi
