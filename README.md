# signature-sdk

Multi-language SDK for LSH (Locality-Sensitive Hashing) signature generation for AI prompt similarity detection.

## Overview

This SDK provides a unified implementation of the LSH signature algorithm across three languages (Rust, Python, TypeScript), extracted and consolidated from:

- **Heimdall** (Rust) — canonical/authoritative implementation
- **Thor** (TypeScript) — frontend/Node.js implementation  
- **Research/signature_cli** (Python) — research/CLI implementation

## Project Status

✅ **Phases 1-6 Complete** — Production-ready SDK with comprehensive documentation!

- ✅ Phase 1-5: All three SDKs implemented and validated (61 tests passing)
- ✅ Phase 6: Docusaurus documentation site
- ✅ Phase 7: CI/CD pipeline and packaging

See [VALIDATION.md](VALIDATION.md) for the complete cross-language validation report.

## Quick Links

- **[Validation Report](VALIDATION.md)** — Cross-language validation (61 tests passing)
- **[Algorithm Specification](spec/SPEC.md)** — Formal algorithm definition
- **[Versioning Specification](spec/VERSIONING.md)** — Signature version registry and compatibility
- **[Test Vectors](spec/test-vectors/)** — Canonical test vectors (8 files, 124 cases)
- **[Implementation Plans](.opencode/plans/)** — Detailed implementation phases

## Packages

| Language   | Package      | Status      | Tests      | Path           |
|------------|--------------|-------------|------------|----------------|
| Rust       | `signature-sdk`   | ✅ Ready    | 43 passing | [packages/rust/](packages/rust/) |
| Python     | `signature-sdk`   | ✅ Ready    | 11 passing | [packages/python/](packages/python/) |
| TypeScript | `@0din/signature-sdk`  | ✅ Ready    | 7 passing  | [packages/typescript/](packages/typescript/) |

**Total**: 61 tests passing across all languages

## Signature Versions

| Version | Provider | Model                        | Dimensions | Status |
|---------|----------|------------------------------|------------|--------|
| V0      | OpenAI   | text-embedding-3-large       | 1536       | ✅ Stable |
| V1      | ONNX     | multilingual-e5-small (tuned)| 384        | ✅ Stable |
| Latest  | →V1      | —                            | —          | Alias  |

**⚠️ Important:** V0 and V1 signatures use different embedding spaces and are **not comparable**.

## Features

- ✅ **SimHash LSH** — Random hyperplane LSH (Charikar 2002)
- ✅ **Deterministic** — Same input always produces same signature (via SplitMix64 PRNG)
- ✅ **Multi-family hashing** — 3 independent hash families for robustness
- ✅ **Band splitting** — 16 bands for LSH indexing and ANN search
- ✅ **Canonical SHA256** — Deterministic embedding deduplication
- ✅ **CM-LSH** — Confidence Matrix LSH (Rust, Python; optional, higher accuracy)
- ✅ **ONNX embeddings** — Local, API-free embedding generation (Rust, Python)
- ✅ **OpenAI embeddings** — API-based embedding generation (all languages)
- ✅ **Cross-language parity** — Identical signatures across all three languages
- ✅ **Comprehensive testing** — 61 tests validating 124 test cases

## Installation

> **Note:** These packages are currently for internal use only and are not published to public registries (crates.io, PyPI, npm). Use git dependencies as shown below.

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
signature-sdk = { git = "https://github.com/0din-ai/signature-sdk", branch = "main" }

# With CM-LSH feature (optional, higher accuracy)
signature-sdk = { git = "https://github.com/0din-ai/signature-sdk", branch = "main", features = ["cm-lsh"] }

# With ONNX embeddings (local, API-free)
signature-sdk = { git = "https://github.com/0din-ai/signature-sdk", branch = "main", features = ["onnx"] }

# With OpenAI embeddings
signature-sdk = { git = "https://github.com/0din-ai/signature-sdk", branch = "main", features = ["openai"] }

# All features
signature-sdk = { git = "https://github.com/0din-ai/signature-sdk", branch = "main", features = ["cm-lsh", "onnx", "openai"] }
```

Or via command line:

```bash
cargo add signature-sdk --git https://github.com/0din-ai/signature-sdk --branch main
```

### Python

```bash
# Install from git (core features only)
pip install "signature-sdk @ git+https://github.com/0din-ai/signature-sdk#subdirectory=packages/python"

# With all optional features (CM-LSH, ONNX, OpenAI)
pip install "signature-sdk[all] @ git+https://github.com/0din-ai/signature-sdk#subdirectory=packages/python"

# With specific features
pip install "signature-sdk[cm-lsh] @ git+https://github.com/0din-ai/signature-sdk#subdirectory=packages/python"
pip install "signature-sdk[onnx] @ git+https://github.com/0din-ai/signature-sdk#subdirectory=packages/python"
pip install "signature-sdk[openai] @ git+https://github.com/0din-ai/signature-sdk#subdirectory=packages/python"
```

Add to `requirements.txt`:

```txt
signature-sdk[all] @ git+https://github.com/0din-ai/signature-sdk#subdirectory=packages/python
```

### TypeScript

Add to your `package.json`:

```json
{
  "dependencies": {
    "@0din/signature-sdk": "github:0din-ai/signature-sdk#main"
  }
}
```

Or via command line:

```bash
npm install github:0din-ai/signature-sdk#main

# With yarn
yarn add github:0din-ai/signature-sdk#main

# With pnpm
pnpm add github:0din-ai/signature-sdk#main
```

**Note:** For monorepos using npm workspaces, you may need to adjust the path:

```bash
npm install "github:0din-ai/signature-sdk#main" --workspace=typescript
```

## Quick Start

### Rust

```rust
use signature_sdk::{simhash_lsh_multi, normalize_vector, LshConfig};

let vector = vec![0.5, 0.5, 0.5, 0.5]; // Your embedding
let normalized = normalize_vector(&vector);
let families = simhash_lsh_multi(&normalized, &LshConfig::default());

println!("Signature: {}", families[0].signature);
```

See [packages/rust/README.md](packages/rust/README.md) for full documentation.

### Python

```python
from signature_sdk import simhash_lsh_multi, normalize_vector

vector = [0.5, 0.5, 0.5, 0.5]  # Your embedding
normalized = normalize_vector(vector)
families = simhash_lsh_multi(normalized)

print(f"Signature: {families[0].signature}")
```

See [packages/python/README.md](packages/python/README.md) for full documentation.

### TypeScript

```typescript
import { simhashLshMulti, normalizeVector } from '@0din/signature-sdk';

const vector = [0.5, 0.5, 0.5, 0.5]; // Your embedding
const normalized = normalizeVector(vector);
const families = simhashLshMulti(normalized);

console.log(`Signature: ${families[0].signature}`);
```

See [packages/typescript/README.md](packages/typescript/README.md) for full documentation.

## Examples

Each language includes runnable example files demonstrating common use cases:

### Rust Examples

```bash
# Basic signature generation
cargo run --example basic_signature

# Similarity comparison
cargo run --example similarity_comparison

# Duplicate detection with bands
cargo run --example duplicate_detection

# CM-LSH (requires feature flag)
cargo run --example cm_lsh_example --features cm-lsh
```

See [packages/rust/examples/](packages/rust/examples/) for source code.

### Python Examples

```bash
# Basic signature generation
python packages/python/examples/basic_signature.py

# Similarity comparison
python packages/python/examples/similarity_comparison.py

# Duplicate detection with bands
python packages/python/examples/duplicate_detection.py

# CM-LSH
python packages/python/examples/cm_lsh_example.py
```

See [packages/python/examples/](packages/python/examples/) for source code.

### TypeScript Examples

```bash
# Basic signature generation
npx ts-node packages/typescript/examples/basic_signature.ts

# Similarity comparison
npx ts-node packages/typescript/examples/similarity_comparison.ts

# Duplicate detection with bands
npx ts-node packages/typescript/examples/duplicate_detection.ts
```

See [packages/typescript/examples/](packages/typescript/examples/) for source code.

## Algorithm Overview

**SimHash via Random Hyperplane LSH:**

1. Embed text using OpenAI or ONNX provider
2. L2-normalize embedding to unit length
3. For each of 256 bits:
   - Generate deterministic random hyperplane via SplitMix64
   - Compute dot product with normalized embedding
   - Bit = 1 if dot > 0, else 0
4. Pack bits into 64-character hex string
5. Split into 16 bands for LSH indexing

**Default configuration:**
- 3 independent hash families (improves recall)
- 256 bits per signature (64 hex chars)
- 16 bands (4 hex chars each)

## Signature Format

```
0din-v{N}:<hex_signature>
```

Examples:
- `0din-v0:a3f9c2e1b8d4f7a2...` (V0: OpenAI)
- `0din-v1:7f2c8a9d3e1b5f4c...` (V1: ONNX)

## Documentation

- **[spec/SPEC.md](spec/SPEC.md)** — Complete algorithm specification with pseudocode
- **[spec/VERSIONING.md](spec/VERSIONING.md)** — Version registry, compatibility, migration guide
- **[models/v1/config.json](models/v1/config.json)** — ONNX model metadata
- **[.opencode/plans/](.opencode/plans/)** — Implementation plans for all phases

## Development

### Project Structure

```
sig-sdk/
├── spec/                  # Formal specifications
│   ├── SPEC.md           # Algorithm spec
│   ├── VERSIONING.md     # Version registry
│   └── test-vectors/     # Cross-language test vectors
├── models/               # Model metadata
│   └── v1/
│       └── config.json
├── packages/             # Language implementations
│   ├── rust/            # Rust SDK (canonical)
│   ├── python/          # Python SDK
│   └── typescript/      # TypeScript SDK
├── docs/                 # Docusaurus documentation
└── .opencode/plans/      # Implementation plans
```

### Running Tests

```bash
# Test all languages (runs all 61 tests)
make test

# Test individual languages
make test-rust      # 43 tests
make test-python    # 11 tests
make test-typescript # 7 tests

# Generate test vectors from canonical Rust implementation
make generate-vectors

# Install dependencies for all packages
make install

# Run linters
make lint

# Format code
make fmt

# Full CI pipeline (clean, install, lint, test)
make ci

# Display all available commands
make help
```

### Building Documentation

```bash
cd docs && npm run build
```

## Roadmap

- [x] **Phase 1:** Specification & test vectors ✅
- [x] **Phase 2:** Rust SDK (canonical) ✅
- [x] **Phase 3:** Python SDK ✅
- [x] **Phase 4:** TypeScript SDK ✅
- [x] **Phase 5:** Cross-language validation ✅
- [x] **Phase 6:** Docusaurus documentation ✅
- [x] **Phase 7:** CI/CD & packaging ✅

## Contributing

See [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for development setup, coding standards, and pull request guidelines.

**Quick start:**

```bash
# Clone and install
git clone https://github.com/0din-ai/signature-sdk.git
cd sig-sdk
make install

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests
make test

# Run linters
make lint
```

## License

MIT

## References

- Charikar, M. (2002). "Similarity estimation techniques from rounding algorithms." STOC.
- Gong, Y., Lazebnik, S. (2011). "Iterative Quantization: A Procrustean Approach to Learning Binary Codes." CVPR.
- SplitMix64: https://prng.di.unimi.it/splitmix64.c

## Support

For issues or questions, please contact the 0DIN team.
