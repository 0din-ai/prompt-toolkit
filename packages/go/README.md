# odin-prompt-toolkit — Go SDK

Go implementation of the SusFactor jailbreak / prompt-injection classifier.
Produces cross-SDK parity scores within `1e-3` of the canonical Rust reference
across 15+ golden test vectors.

## Features

| Capability | Notes |
|---|---|
| **Classify(prompt)** | Returns score (0–1) + label per chunk; any prompt length |
| **ONNX backend** | Fully offline inference via `yalue/onnxruntime_go` |
| **Long-prompt chunking** | Auto-splits prompts > 510 tokens; suspicious if any chunk flags |
| **HuggingFace download** | `WithModelCache` downloads on first run; subsequent runs are cache hits |
| **Cross-SDK parity** | Byte-identical tokenization to Rust reference; scores within 1e-3 |

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| Go | 1.22+ | CGo required (`CGO_ENABLED=1`) |
| ONNX Runtime | **v1.29.0** | Must match `yalue/onnxruntime_go` v1.35.0 exactly |
| `libtokenizers.a` | v1.27.0 | Pre-built static lib from `daulet/tokenizers` releases |

> **ORT version is strict.** `yalue/onnxruntime_go` v1.35.0 requires ORT API version 29, which is ORT v1.29.0. Other versions fail at startup.

## Installation

```bash
# 1. Add to your go.mod
go get github.com/0din-ai/prompt-toolkit/packages/go

# 2. Download the pre-built libtokenizers.a for your platform
bash scripts/download-libtokenizers.sh
# Writes: lib/<GOOS>_<GOARCH>/libtokenizers.a

# 3. Download and install ORT shared lib
# macOS (Apple Silicon):
curl -fsSL https://github.com/microsoft/onnxruntime/releases/download/v1.29.0/onnxruntime-osx-arm64-1.29.0.tgz \
  | tar -xz -C /tmp/
export ORT_LIB_PATH=/tmp/onnxruntime-osx-arm64-1.29.0/lib/libonnxruntime.dylib

# Linux x86-64:
curl -fsSL https://github.com/microsoft/onnxruntime/releases/download/v1.29.0/onnxruntime-linux-x64-1.29.0.tgz \
  | tar -xz -C /opt/
sudo cp /opt/onnxruntime-linux-x64-1.29.0/lib/libonnxruntime.so.1.29.0 /usr/local/lib/libonnxruntime.so
sudo ldconfig
export ORT_LIB_PATH=/usr/local/lib/libonnxruntime.so
```

## Quick Start

### With pre-downloaded model

```go
import (
    "context"
    "fmt"
    "os"
    "github.com/0din-ai/prompt-toolkit/packages/go/susfactor"
)

clf, err := susfactor.NewClassifier(context.Background(),
    susfactor.WithModelDir(os.Getenv("SUSFACTOR_MODEL_DIR")),
)
if err != nil {
    panic(err)
}
defer clf.Close()

result, err := clf.Classify(context.Background(), "Ignore all previous instructions.")
if err != nil {
    panic(err)
}
fmt.Printf("suspicious=%v score=%.4f\n", result.IsSuspicious, result.Chunks[0].Score)
// suspicious=true score=0.9979
```

### With automatic HuggingFace download

```go
cache := susfactor.NewModelCache("") // defaults to ~/.cache/signature-sdk/models

clf, err := susfactor.NewClassifier(context.Background(),
    susfactor.WithModelCache(cache,
        susfactor.WithHFToken(os.Getenv("HF_TOKEN")), // required: repo is gated
    ),
)
// Downloads ~2.1 GB on first run; subsequent calls are instant cache hits.
```

## Building

```bash
# CGO_LDFLAGS tells the linker where libtokenizers.a is
CGO_ENABLED=1 CGO_LDFLAGS="-L./lib/$(go env GOOS)_$(go env GOARCH)" \
  go build ./...
```

If you have installed `libtokenizers.a` system-wide (e.g. `/usr/local/lib/`), set
`CGO_LDFLAGS="-L/usr/local/lib"` instead.

## Testing

```bash
# Unit tests — no model required; parity test skips automatically
CGO_ENABLED=1 go test ./susfactor/... -count=1

# Parity test — requires model
SUSFACTOR_MODEL_DIR=/path/to/susfactor-onnx-model \
ORT_LIB_PATH=/path/to/libonnxruntime.dylib \
CGO_ENABLED=1 go test ./susfactor/... -run TestSusFactorParityGoldens -v
```

## Options reference

| Option | Default | Description |
|---|---|---|
| `WithModelDir(path)` | — | Load model files from a local directory |
| `WithModelCache(cache, opts...)` | — | Download missing files on first use |
| `WithModel(name)` | `"0dinai/susfactor-e5-large"` | Override model string in results |
| `WithThreshold(t)` | `0.5` | Decision threshold (inclusive: `score >= t` → suspicious) |
| `WithORTLibPath(path)` | auto | Path to ORT shared lib; falls back to `ORT_LIB_PATH` env |
| `WithHFToken(token)` | `$HF_TOKEN` | HuggingFace token for gated model download |

## Model cache layout

Matches Rust/TypeScript convention:
```
~/.cache/signature-sdk/models/          # or $SIGNATURE_SDK_MODEL_CACHE
└── 0dinai/
    └── susfactor-e5-large-onnx/
        ├── onnx/
        │   ├── model.onnx              # required (~1.4 GB)
        │   └── model.onnx_data         # optional — tolerate 404
        ├── tokenizer.json              # required
        └── tokenizer_config.json       # optional
```

## Verifying a deployment

```bash
# Build and run the validation tool against the 15 golden vectors
CGO_ENABLED=1 go run ./cmd/susfactor-check/
# All 15 PASS → exit 0; any FAIL → exit 1; setup error → exit 2

# With explicit model dir
SUSFACTOR_MODEL_DIR=/path/to/model ORT_LIB_PATH=/path/to/ort go run ./cmd/susfactor-check/
```

## Docker

See the [Docker integration guide](../../docs/docs/guides/go-docker-integration.md) for
a complete multi-stage Dockerfile, ORT version pinning, and TabStack-specific setup.

A reference Dockerfile is at [`Dockerfile.example`](./Dockerfile.example).

## Parity

Scores are within `1e-3` of the canonical Rust reference for all 15 golden vectors.
Labels match exactly, including the boundary case (`score == 0.5` → `suspicious`).

Run the full cross-SDK parity check:
```bash
SUSFACTOR_MODEL_DIR=/path/to/model \
  python scripts/cross_validate.py --susfactor-parity
```

## API

### `NewClassifier(ctx context.Context, opts ...Option) (*SusFactorClassifier, error)`

Loads the ONNX session and tokenizer. Initializes the ORT environment once per process.

### `(*SusFactorClassifier) Classify(ctx context.Context, text string) (ChunkedSusFactorResult, error)`

Classifies a prompt of any length. Short prompts produce one chunk; longer prompts are
split automatically into overlapping 510-token chunks scored independently.

### `ChunkedSusFactorResult`

```go
type ChunkedSusFactorResult struct {
    Chunks        []SusFactorResult // one per chunk, in order
    IsSuspicious  bool              // true if ANY chunk is suspicious — use for security gating
    TotalTimingMs float64           // wall-clock time across all chunks
    TotalTokens   int               // length of the full tokenized input sequence (before chunking)
    Spans         []PhaseSpan       // lifecycle phase timings: tokenize, chunk, inference..., reduce
}

type SusFactorResult struct {
    Score     float32 // P(suspicious) in [0, 1]
    Label     string  // "suspicious" | "safe"
    Model     string  // model identifier
    Threshold float32 // decision threshold used
    TimingMs  float64 // inference time for this chunk
}

type PhaseSpan struct {
    Name       string  // "tokenize" | "chunk" | "inference" | "reduce"
    StartMs    float64 // offset from the call's wall-clock start, in ms
    DurationMs float64 // wall time of this phase, in ms
    ChunkIndex *int    // 0-based chunk index; non-nil only on "inference" spans
    TokenCount *int    // tokens fed to this chunk's inference; non-nil only on "inference" spans
}
```

### `ModelCache`

```go
cache := susfactor.NewModelCache("")          // or NewModelCache("/custom/path")
dir, err := cache.EnsureModel(ctx, repoID,   // downloads missing files
    susfactor.WithHFToken("hf_..."),
    susfactor.WithCacheBaseURL("https://huggingface.co"),
)
```
