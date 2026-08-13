---
sidebar_position: 1
---

# Installation

Install odin-prompt-toolkit for your preferred language. All implementations provide identical SusFactor classification; Rust, Python, and TypeScript also include LSH signatures and the threat feed.

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
| `susfactor` | ❌ No | Jailbreak/prompt-injection classifier ([SusFactor](../concepts/susfactor.md)) |
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
pip install "0din-prompt-toolkit @ git+https://github.com/0din-ai/prompt-toolkit#subdirectory=packages/python"
```

### Optional Dependencies

```bash
# Core LSH only (no embeddings)
pip install "0din-prompt-toolkit @ git+https://github.com/0din-ai/prompt-toolkit#subdirectory=packages/python"

# With OpenAI support
pip install "0din-prompt-toolkit[openai] @ git+https://github.com/0din-ai/prompt-toolkit#subdirectory=packages/python"

# With ONNX support (V1 embeddings + SusFactor ONNX backend)
pip install "0din-prompt-toolkit[onnx] @ git+https://github.com/0din-ai/prompt-toolkit#subdirectory=packages/python"

# With CM-LSH support
pip install "0din-prompt-toolkit[cm-lsh] @ git+https://github.com/0din-ai/prompt-toolkit#subdirectory=packages/python"

# With SusFactor (PyTorch backend)
pip install "0din-prompt-toolkit[susfactor] @ git+https://github.com/0din-ai/prompt-toolkit#subdirectory=packages/python"

# With Threat Feed integration
pip install "0din-prompt-toolkit[threatfeed] @ git+https://github.com/0din-ai/prompt-toolkit#subdirectory=packages/python"

# All features
pip install "0din-prompt-toolkit[all] @ git+https://github.com/0din-ai/prompt-toolkit#subdirectory=packages/python"
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
npm install "github:0din-ai/prompt-toolkit#main" --workspace=typescript
```

Or with Yarn:

```bash
yarn add "github:0din-ai/prompt-toolkit#main"
```

### Requirements

- Node.js 18 or higher
- TypeScript 4.5 or higher (if using TypeScript)

### Verify Installation

```typescript
import { simhashLshMulti } from '@0din/prompt-toolkit';
console.log('odin-prompt-toolkit installed successfully');
```

## Go

The Go SDK provides SusFactor jailbreak classification. It requires two native shared libraries (ORT and libtokenizers) installed separately — see [Go + Docker Integration](../guides/go-docker-integration.md) for a production-ready setup.

```bash
go get github.com/0din-ai/prompt-toolkit/packages/go@main
```

### Native Dependencies

The Go SDK uses CGo and links against:

| Library | Version | Purpose |
|---------|---------|---------|
| `libonnxruntime.so` | 1.26.0 | ONNX Runtime inference |
| `libtokenizers.a` | 1.27.0 | HuggingFace tokenizers |

**Linux (CI / Docker):**

```bash
# ORT v1.26.0
curl -fsSL https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-x64-1.26.0.tgz \
  | tar -xz -C /opt/
sudo cp /opt/onnxruntime-linux-x64-1.26.0/lib/libonnxruntime.so.1.26.0 /usr/local/lib/libonnxruntime.so
sudo ldconfig

# libtokenizers v1.27.0
curl -fsSL https://github.com/daulet/tokenizers/releases/download/v1.27.0/libtokenizers.linux-amd64.tar.gz \
  | tar -xz -C /tmp/
sudo cp /tmp/libtokenizers.a /usr/local/lib/libtokenizers.a
```

Then build with:

```bash
CGO_ENABLED=1 CGO_LDFLAGS="-L/usr/local/lib" \
  ORT_LIB_PATH=/usr/local/lib/libonnxruntime.so \
  go build ./...
```

### Requirements

- Go 1.22 or higher
- CGo enabled (`CGO_ENABLED=1`)
- ORT v1.26.0 shared library
- libtokenizers v1.27.0 static library

### Verify Installation

```go
package main

import (
    "context"
    "fmt"
    "github.com/0din-ai/prompt-toolkit/packages/go/susfactor"
)

func main() {
    clf, err := susfactor.NewClassifier(context.Background(),
        susfactor.WithModelDir("/path/to/susfactor-v1"),
    )
    if err != nil {
        panic(err)
    }
    defer clf.Close()
    fmt.Println("Go SDK ready")
}
```

---

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

This installs dependencies for all four languages:
- Rust: `cargo fetch`
- Python: `pip install -e ".[dev,all]"`
- TypeScript: `npm install`
- Go: native libraries must be installed separately (see [Go section](#go) above)

### Run Tests

```bash
make test
```

Runs all 400+ tests across the four implementations.

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
