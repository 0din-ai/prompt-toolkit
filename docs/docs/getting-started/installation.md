---
sidebar_position: 1
---

# Installation

Install 0din-sig for your preferred language. All three implementations provide identical functionality and signatures.

## Rust

Add `odin-sig` to your `Cargo.toml`:

```toml
[dependencies]
odin-sig = { git = "https://github.com/0din/sig-sdk" }
```

### Feature Flags

```toml
[dependencies]
odin-sig = { 
  git = "https://github.com/0din/sig-sdk",
  features = ["openai", "onnx", "cm-lsh"]
}
```

| Feature | Default | Description |
|---------|---------|-------------|
| `openai` | ✅ Yes | OpenAI API embedding provider (V0 signatures) |
| `onnx` | ✅ Yes | Local ONNX embedding provider (V1 signatures) |
| `cm-lsh` | ❌ No | Confidence Matrix LSH (experimental) |

### Verify Installation

```bash
cargo build
```

## Python

Install via pip with git dependency:

```bash
pip install "0din-sig @ git+https://github.com/0din/sig-sdk#subdirectory=packages/python"
```

### Optional Dependencies

```bash
# Core LSH only (no embeddings)
pip install "0din-sig[dev] @ git+https://github.com/0din/sig-sdk#subdirectory=packages/python"

# With OpenAI support
pip install "0din-sig[openai] @ git+https://github.com/0din/sig-sdk#subdirectory=packages/python"

# With ONNX support
pip install "0din-sig[onnx] @ git+https://github.com/0din/sig-sdk#subdirectory=packages/python"

# With CM-LSH support
pip install "0din-sig[cm-lsh] @ git+https://github.com/0din/sig-sdk#subdirectory=packages/python"

# All features
pip install "0din-sig[all] @ git+https://github.com/0din/sig-sdk#subdirectory=packages/python"
```

### Requirements

- Python 3.10 or higher
- NumPy (automatically installed)

### Verify Installation

```python
import odin_sig
print(odin_sig.__version__)
```

## TypeScript

Install via npm with git dependency:

```bash
npm install "github:0din/sig-sdk#main" --workspace=typescript
```

Or with Yarn:

```bash
yarn add "github:0din/sig-sdk#main"
```

### Requirements

- Node.js 18 or higher
- TypeScript 4.5 or higher (if using TypeScript)

### Verify Installation

```typescript
import { simhashLshMulti } from '@0din/sig';
console.log('0din-sig installed successfully');
```

## Development Installation

If you're contributing to the SDK or need the latest unreleased changes:

### Clone Repository

```bash
git clone https://github.com/0din/sig-sdk.git
cd sig-sdk
```

### Install All Dependencies

```bash
make install
```

This installs dependencies for all three languages:
- Rust: `cargo fetch`
- Python: `pip install -e ".[dev,all]"`
- TypeScript: `npm install`

### Run Tests

```bash
make test
```

Runs all 61 tests across the three implementations.

## Next Steps

- **[Quick Start](./quick-start)** — Generate your first signature
- **[Configuration](./configuration)** — Configure embedding providers and LSH parameters
- **[Examples](https://github.com/0din/sig-sdk/tree/main/packages/rust/examples)** — See runnable examples

## Troubleshooting

### Rust: ONNX Model Download Fails

If the ONNX model fails to download on first use:

1. Check internet connection
2. Manually download from [HuggingFace](https://huggingface.co/intfloat/multilingual-e5-small)
3. Place in `~/.cache/odin-sig/models/v1/`

### Python: NumPy Import Error

Ensure you have a compatible NumPy version:

```bash
pip install "numpy>=1.24"
```

### TypeScript: Module Not Found

If imports fail, ensure your `tsconfig.json` includes:

```json
{
  "compilerOptions": {
    "moduleResolution": "node",
    "esModuleInterop": true
  }
}
```

## Registry Publishing (Coming Soon)

The SDK is currently available via git dependencies only. Publication to official registries is planned:

- 🔲 **crates.io** (Rust)
- 🔲 **PyPI** (Python)
- 🔲 **npm** (TypeScript)

Follow the [GitHub repository](https://github.com/0din/sig-sdk) for updates.
