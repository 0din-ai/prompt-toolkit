---
sidebar_position: 1
---

# Installation

Install odin-prompt-toolkit for your preferred language. All three implementations provide identical functionality and signatures.

## Rust

Add `odin-prompt-toolkit` to your `Cargo.toml`:

```toml
[dependencies]
odin-prompt-toolkit = { git = "https://github.com/0din-ai/prompt-toolkit" }
```

### Feature Flags

```toml
[dependencies]
odin-prompt-toolkit = { 
  git = "https://github.com/0din-ai/prompt-toolkit",
  features = ["openai", "onnx", "cm-lsh", "susfactor", "threatfeed"]
}
```

| Feature | Default | Description |
|---------|---------|-------------|
| `openai` | ✅ Yes | OpenAI API embedding provider (V0 signatures) |
| `onnx` | ✅ Yes | Local ONNX embedding provider (V1 signatures) |
| `cm-lsh` | ❌ No | Confidence Matrix LSH (experimental, higher accuracy) |
| `susfactor` | ❌ No | Jailbreak/prompt-injection classifier ([SusFactor](../concepts/susfactor)) |
| `threatfeed` | ❌ No | Threat feed sync and similarity lookup |

:::note ONNX Runtime build requirement

The `onnx` feature uses [ONNX Runtime](https://onnxruntime.ai/) via the `ort`
crate with `download-binaries`, which fetches a prebuilt native ONNX Runtime
library **at build time**. This means building the `onnx` feature requires
network access (or a pre-cached binary). For air-gapped or offline builds,
supply your own ONNX Runtime library via the `ORT_DYLIB_PATH` environment
variable. A prebuilt binary must exist for your target triple.

:::

### Verify Installation

```bash
cargo build
```

## Python

Install via pip with git dependency:

```bash
pip install "odin-prompt-toolkit @ git+https://github.com/0din-ai/prompt-toolkit#subdirectory=packages/python"
```

### Optional Dependencies

```bash
# Core LSH only (no embeddings)
pip install "odin-prompt-toolkit @ git+https://github.com/0din-ai/prompt-toolkit#subdirectory=packages/python"

# With OpenAI support
pip install "odin-prompt-toolkit[openai] @ git+https://github.com/0din-ai/prompt-toolkit#subdirectory=packages/python"

# With ONNX support (V1 embeddings + SusFactor ONNX backend)
pip install "odin-prompt-toolkit[onnx] @ git+https://github.com/0din-ai/prompt-toolkit#subdirectory=packages/python"

# With CM-LSH support
pip install "odin-prompt-toolkit[cm-lsh] @ git+https://github.com/0din-ai/prompt-toolkit#subdirectory=packages/python"

# With SusFactor (PyTorch backend)
pip install "odin-prompt-toolkit[susfactor] @ git+https://github.com/0din-ai/prompt-toolkit#subdirectory=packages/python"

# With Threat Feed integration
pip install "odin-prompt-toolkit[threatfeed] @ git+https://github.com/0din-ai/prompt-toolkit#subdirectory=packages/python"

# All features
pip install "odin-prompt-toolkit[all] @ git+https://github.com/0din-ai/prompt-toolkit#subdirectory=packages/python"
```

### Requirements

- Python 3.10 or higher
- NumPy (automatically installed)

### Verify Installation

```python
import odin_prompt_toolkit
print(odin_prompt_toolkit.__version__)
```

## TypeScript

Install via npm with git dependency:

```bash
npm install "github:0din-ai/odin-prompt-toolkit#main" --workspace=typescript
```

Or with Yarn:

```bash
yarn add "github:0din-ai/odin-prompt-toolkit#main"
```

### Requirements

- Node.js 18 or higher
- TypeScript 4.5 or higher (if using TypeScript)

### Verify Installation

```typescript
import { simhashLshMulti } from '@0din/odin-prompt-toolkit';
console.log('odin-prompt-toolkit installed successfully');
```

## Development Installation

If you're contributing to the SDK or need the latest unreleased changes:

### Clone Repository

```bash
git clone https://github.com/0din-ai/prompt-toolkit.git
cd prompt-toolkit
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

Runs all 384 tests across the three implementations.

## Next Steps

- **[Quick Start](./quick-start)** — Generate your first signature
- **[Configuration](./configuration)** — Configure embedding providers and LSH parameters
- **[Examples](https://github.com/0din-ai/prompt-toolkit/tree/main/packages/rust/examples)** — See runnable examples

## Troubleshooting

### Rust: ONNX Model Download Fails

If the ONNX model fails to download on first use:

1. Check internet connection
2. Manually download from [HuggingFace](https://huggingface.co/0dinai/0din-jailbreak-embeddings-small)
3. Place in `~/.cache/odin-prompt-toolkit/models/v1/`

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

Follow the [GitHub repository](https://github.com/0din-ai/prompt-toolkit) for updates.
