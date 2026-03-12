---
sidebar_position: 1
---

# Installation

Install signature-sdk for your preferred language. All three implementations provide identical functionality and signatures.

## Rust

Add `signature-sdk` to your `Cargo.toml`:

```toml
[dependencies]
signature-sdk = { git = "https://github.com/0din-ai/signature-sdk" }
```

### Feature Flags

```toml
[dependencies]
signature-sdk = { 
  git = "https://github.com/0din-ai/signature-sdk",
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
pip install "signature-sdk @ git+https://github.com/0din-ai/signature-sdk#subdirectory=packages/python"
```

### Optional Dependencies

```bash
# Core LSH only (no embeddings)
pip install "signature-sdk[dev] @ git+https://github.com/0din-ai/signature-sdk#subdirectory=packages/python"

# With OpenAI support
pip install "signature-sdk[openai] @ git+https://github.com/0din-ai/signature-sdk#subdirectory=packages/python"

# With ONNX support
pip install "signature-sdk[onnx] @ git+https://github.com/0din-ai/signature-sdk#subdirectory=packages/python"

# With CM-LSH support
pip install "signature-sdk[cm-lsh] @ git+https://github.com/0din-ai/signature-sdk#subdirectory=packages/python"

# All features
pip install "signature-sdk[all] @ git+https://github.com/0din-ai/signature-sdk#subdirectory=packages/python"
```

### Requirements

- Python 3.10 or higher
- NumPy (automatically installed)

### Verify Installation

```python
import signature_sdk
print(signature_sdk.__version__)
```

## TypeScript

Install via npm with git dependency:

```bash
npm install "github:0din-ai/signature-sdk#main" --workspace=typescript
```

Or with Yarn:

```bash
yarn add "github:0din-ai/signature-sdk#main"
```

### Requirements

- Node.js 18 or higher
- TypeScript 4.5 or higher (if using TypeScript)

### Verify Installation

```typescript
import { simhashLshMulti } from '@0din/signature-sdk';
console.log('signature-sdk installed successfully');
```

## Development Installation

If you're contributing to the SDK or need the latest unreleased changes:

### Clone Repository

```bash
git clone https://github.com/0din-ai/signature-sdk.git
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
- **[Examples](https://github.com/0din-ai/signature-sdk/tree/main/packages/rust/examples)** — See runnable examples

## Troubleshooting

### Rust: ONNX Model Download Fails

If the ONNX model fails to download on first use:

1. Check internet connection
2. Manually download from [HuggingFace](https://huggingface.co/intfloat/multilingual-e5-large)
3. Place in `~/.cache/signature-sdk/models/v1/`

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

Follow the [GitHub repository](https://github.com/0din-ai/signature-sdk) for updates.
